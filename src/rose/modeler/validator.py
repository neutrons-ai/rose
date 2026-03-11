"""
Validate LLM-generated YAML against the ROSE model schema.

The validator checks structural correctness without building the
full refl1d experiment.  This makes it fast enough to run in the
LLM retry loop.
"""

from __future__ import annotations

import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)

#: Required keys for each layer entry.
_LAYER_REQUIRED = {"name", "rho"}

#: Allowed keys inside a layer's ``fit`` block.
_FIT_KEYS = {"thickness", "rho", "irho", "interface"}

#: Allowed keys in the ``experiment`` section.
_EXPERIMENT_KEYS = {
    "q_min",
    "q_max",
    "q_points",
    "dq_over_q",
    "relative_error",
    "step_interfaces",
    "data_file",
}

#: Required keys in the ``optimization`` section.
_OPT_REQUIRED = {"param", "param_values"}

#: Allowed keys in the ``optimization`` section.
_OPT_KEYS = {
    "param",
    "param_values",
    "parameters_of_interest",
    "num_realizations",
    "mcmc_steps",
    "entropy_method",
}


def validate_model_yaml(yaml_text: str) -> list[str]:
    """Validate a YAML string as a ROSE model description.

    Returns a list of error strings.  An empty list means the YAML
    is valid.

    Args:
        yaml_text: Raw YAML string to validate.

    Returns:
        List of human-readable error messages (empty if valid).
    """
    errors: list[str] = []

    # Parse YAML
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return [f"Invalid YAML syntax: {exc}"]

    if not isinstance(data, dict):
        return ["Top-level YAML must be a mapping (dict)"]

    # -- top-level keys ------------------------------------------------
    _ALLOWED_TOP_KEYS = {
        "name",
        "description",
        "layers",
        "experiment",
        "optimization",
    }
    for key in data:
        if key not in _ALLOWED_TOP_KEYS:
            errors.append(f"Unknown top-level key: '{key}'")

    # -- layers --------------------------------------------------------
    if "layers" not in data:
        errors.append("Missing required key: 'layers'")
    else:
        layers = data["layers"]
        if not isinstance(layers, list) or len(layers) < 2:
            errors.append("'layers' must be a list with at least 2 entries")
        else:
            errors.extend(_validate_layers(layers))

    # -- experiment (optional) -----------------------------------------
    if "experiment" in data:
        errors.extend(_validate_experiment(data["experiment"]))

    # -- optimization --------------------------------------------------
    if "optimization" not in data:
        errors.append("Missing required key: 'optimization'")
    else:
        errors.extend(
            _validate_optimization(data["optimization"], data.get("layers", []))
        )

    return errors


def _validate_layers(layers: list[Any]) -> list[str]:
    """Validate the layers list."""
    errors: list[str] = []
    names_seen: set[str] = set()

    for i, layer in enumerate(layers):
        prefix = f"layers[{i}]"
        if not isinstance(layer, dict):
            errors.append(f"{prefix}: must be a mapping")
            continue

        # Required keys
        for key in _LAYER_REQUIRED:
            if key not in layer:
                errors.append(f"{prefix}: missing required key '{key}'")

        # Name uniqueness
        name = layer.get("name")
        if name is not None:
            if name in names_seen:
                errors.append(f"{prefix}: duplicate layer name '{name}'")
            names_seen.add(name)

        # Numeric fields
        for key in ("rho", "thickness", "interface"):
            val = layer.get(key)
            if val is not None:
                try:
                    float(val)
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.{key}: must be a number")

        # Fit ranges
        if "fit" in layer:
            fit = layer["fit"]
            if not isinstance(fit, dict):
                errors.append(f"{prefix}.fit: must be a mapping")
            else:
                for key, bounds in fit.items():
                    if key not in _FIT_KEYS:
                        errors.append(
                            f"{prefix}.fit.{key}: unknown fit key "
                            f"(allowed: {_FIT_KEYS})"
                        )
                    if not isinstance(bounds, list) or len(bounds) != 2:
                        errors.append(f"{prefix}.fit.{key}: must be [min, max]")
                    else:
                        try:
                            lo, hi = float(bounds[0]), float(bounds[1])
                            if lo >= hi:
                                errors.append(f"{prefix}.fit.{key}: min must be < max")
                        except (TypeError, ValueError):
                            errors.append(f"{prefix}.fit.{key}: bounds must be numbers")
    return errors


def _validate_experiment(experiment: Any) -> list[str]:
    """Validate the experiment section."""
    errors: list[str] = []
    if not isinstance(experiment, dict):
        errors.append("'experiment' must be a mapping")
        return errors

    for key in experiment:
        if key not in _EXPERIMENT_KEYS:
            errors.append(f"experiment.{key}: unknown key")

    # Numeric checks
    for key in ("q_min", "q_max", "dq_over_q", "relative_error"):
        val = experiment.get(key)
        if val is not None:
            try:
                v = float(val)
                if v <= 0:
                    errors.append(f"experiment.{key}: must be positive")
            except (TypeError, ValueError):
                errors.append(f"experiment.{key}: must be a number")

    q_points = experiment.get("q_points")
    if q_points is not None:
        try:
            if int(q_points) < 1:
                errors.append("experiment.q_points: must be >= 1")
        except (TypeError, ValueError):
            errors.append("experiment.q_points: must be an integer")

    return errors


def _validate_optimization(optimization: Any, layers: list[Any]) -> list[str]:
    """Validate the optimization section."""
    errors: list[str] = []
    if not isinstance(optimization, dict):
        errors.append("'optimization' must be a mapping")
        return errors

    for key in _OPT_REQUIRED:
        if key not in optimization:
            errors.append(f"optimization: missing required key '{key}'")

    for key in optimization:
        if key not in _OPT_KEYS:
            errors.append(f"optimization.{key}: unknown key")

    layer_names = {l.get("name") for l in layers if isinstance(l, dict)}

    # param must reference a known layer
    param = optimization.get("param")
    if param and isinstance(param, str):
        parts = param.rsplit(" ", 1)
        if len(parts) == 2:
            layer_name, prop = parts
            if layer_name not in layer_names:
                errors.append(
                    f"optimization.param: layer '{layer_name}' not found "
                    f"in layers (available: {sorted(layer_names)})"
                )
            if prop not in ("thickness", "rho", "irho", "interface"):
                errors.append(
                    f"optimization.param: property '{prop}' must be one of "
                    "thickness, rho, interface"
                )
        else:
            errors.append(
                f"optimization.param: expected format "
                f"'<layer_name> <property>', got '{param}'"
            )

    # parameters_of_interest — validate references
    _VALID_PROPERTIES = {"thickness", "rho", "irho", "interface"}
    poi = optimization.get("parameters_of_interest")
    if poi is not None:
        if not isinstance(poi, list):
            errors.append("optimization.parameters_of_interest: must be a list")
        else:
            for i, entry in enumerate(poi):
                if not isinstance(entry, str):
                    errors.append(
                        f"optimization.parameters_of_interest[{i}]: must be a string"
                    )
                    continue
                parts = entry.rsplit(" ", 1)
                if len(parts) == 2:
                    lname, prop = parts
                    if lname not in layer_names:
                        errors.append(
                            f"optimization.parameters_of_interest[{i}]: "
                            f"layer '{lname}' not found in layers "
                            f"(available: {sorted(layer_names)})"
                        )
                    if prop not in _VALID_PROPERTIES:
                        errors.append(
                            f"optimization.parameters_of_interest[{i}]: "
                            f"property '{prop}' must be one of "
                            f"{sorted(_VALID_PROPERTIES)}"
                        )
                else:
                    errors.append(
                        f"optimization.parameters_of_interest[{i}]: "
                        f"expected format '<layer_name> <property>', "
                        f"got '{entry}'"
                    )

    # param_values
    pv = optimization.get("param_values")
    if pv is not None:
        if not isinstance(pv, list) or len(pv) < 2:
            errors.append("optimization.param_values: must be a list with >= 2 values")
        else:
            for i, v in enumerate(pv):
                try:
                    float(v)
                except (TypeError, ValueError):
                    errors.append(f"optimization.param_values[{i}]: must be a number")

    return errors
