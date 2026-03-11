"""
Build refl1d Experiment objects from YAML or JSON model descriptions.

Model files describe a layer stack declaratively — no code execution
is involved.  Supported formats:

- **YAML** (``.yaml`` / ``.yml``) — human-readable, ideal for
  hand-authored or LLM-generated models (Use-case 2).
- **JSON** (``.json``) — machine-friendly, can be exported from
  other tools or the refl1d GUI.

YAML schema example::

    name: Layer A on B
    layers:
      - name: air
        rho: 0.0
        thickness: 0
        interface: 5
      - name: layer_A
        rho: 3.0
        thickness: 40
        interface: 8
        fit:
          thickness: [10, 100]
          rho: [1.0, 5.0]
          interface: [1, 15]
      - name: layer_B
        rho: 4.5
        thickness: 50
        interface: 5
      - name: Si
        rho: 2.07
    experiment:
      q_min: 0.008
      q_max: 0.2
      q_points: 50
      dq_over_q: 0.025
      relative_error: 0.10
      step_interfaces: false
      data_file: null
    optimization:
      param: layer_B thickness
      param_values: [20, 40, 60, 80]
      parameters_of_interest: [layer_A thickness]
      num_realizations: 5
      mcmc_steps: 3000
      entropy_method: kdn
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from refl1d.names import SLD, Experiment, FitProblem, Parameter, QProbe

logger = logging.getLogger(__name__)

#: Maximum allowed model file size (1 MB).
MAX_MODEL_FILE_SIZE = 1 * 1024 * 1024

#: Default values for the experiment section.
EXPERIMENT_DEFAULTS: dict[str, object] = {
    "q_min": 0.008,
    "q_max": 0.2,
    "q_points": 50,
    "dq_over_q": 0.025,
    "relative_error": 0.10,
    "step_interfaces": None,
    "data_file": None,
}

#: Default values for the optimization section.
OPTIMIZATION_DEFAULTS: dict[str, object] = {
    "num_realizations": 3,
    "mcmc_steps": 2000,
    "entropy_method": "kdn",
}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def load_model_description(model_file: str | Path) -> dict[str, Any]:
    """Load and validate a YAML or JSON model description.

    Args:
        model_file: Path to a ``.yaml``, ``.yml``, or ``.json`` file.

    Returns:
        Parsed dict with ``"layers"`` and optional ``"name"`` /
        ``"experiment"`` keys.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On unsupported extension, oversized file, or
            missing ``layers`` key.
    """
    model_path = Path(model_file).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suffix = model_path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError(
            f"Unsupported model file format '{suffix}'. Use .yaml, .yml, or .json"
        )

    file_size = model_path.stat().st_size
    if file_size > MAX_MODEL_FILE_SIZE:
        raise ValueError(
            f"Model file too large ({file_size} bytes); "
            f"max is {MAX_MODEL_FILE_SIZE} bytes"
        )

    raw = model_path.read_text(encoding="utf-8")

    desc = json.loads(raw) if suffix == ".json" else yaml.safe_load(raw)

    if not isinstance(desc, dict) or "layers" not in desc:
        raise ValueError(
            "Model file must contain a 'layers' key describing the "
            "sample stack (top to bottom)"
        )

    _validate_layers(desc["layers"])
    if "experiment" in desc:
        _validate_experiment(desc["experiment"])
    if "optimization" in desc:
        _validate_optimization(desc["optimization"])
    return desc


def load_experiment(
    model_file: str | Path,
    q: np.ndarray,
    dq: np.ndarray,
    reflectivity: np.ndarray | None = None,
    errors: np.ndarray | None = None,
) -> Experiment:
    """Build a refl1d ``Experiment`` from a YAML/JSON model file.

    Args:
        model_file: Path to the model description file.
        q: Momentum transfer values.
        dq: Q resolution values.
        reflectivity: Optional measured reflectivity data.
        errors: Optional reflectivity error bars.

    Returns:
        A configured refl1d ``Experiment`` with fit ranges set.
    """
    desc = load_model_description(model_file)
    return build_experiment(desc, q, dq, reflectivity, errors)


def build_experiment(
    desc: dict[str, Any],
    q: np.ndarray,
    dq: np.ndarray,
    reflectivity: np.ndarray | None = None,
    errors: np.ndarray | None = None,
) -> Experiment:
    """Build a refl1d ``Experiment`` from an already-parsed model dict.

    This is the core builder — useful when the description comes from
    an LLM or API rather than a file on disk.

    Args:
        desc: Model description dict with a ``"layers"`` key.
        q: Momentum transfer values.
        dq: Q resolution values.
        reflectivity: Optional measured reflectivity data.
        errors: Optional reflectivity error bars.

    Returns:
        A configured refl1d ``Experiment``.
    """
    layers_desc = desc["layers"]
    experiment_opts = desc.get("experiment", {})

    probe = QProbe(q, dq, R=reflectivity, dR=errors)
    probe.intensity = Parameter(value=1, name="intensity")

    # Build layer stack: first layer is the incident medium (top),
    # last layer is the substrate (bottom, semi-infinite).
    slabs = []
    for layer in layers_desc:
        material = SLD(
            layer["name"],
            rho=layer.get("rho", 0.0),
            irho=layer.get("irho", 0.0),
        )
        thickness = layer.get("thickness", 0)
        interface = layer.get("interface", 0)
        slabs.append(material(thickness, interface))

    # Stack them: slab0 | slab1 | ... | substrate
    sample = slabs[0]
    for slab in slabs[1:]:
        sample = sample | slab

    # Create experiment
    step_interfaces = experiment_opts.get("step_interfaces", None)
    experiment = Experiment(
        sample=sample,
        probe=probe,
        step_interfaces=step_interfaces,
    )

    # Apply fit ranges
    for layer in layers_desc:
        fit = layer.get("fit", {})
        if not fit:
            continue
        layer_name = layer["name"]
        for param_key, bounds in fit.items():
            lo, hi = bounds
            if param_key == "thickness":
                sample[layer_name].thickness.range(lo, hi)
            elif param_key == "interface":
                sample[layer_name].interface.range(lo, hi)
            elif param_key == "rho":
                sample[layer_name].material.rho.range(lo, hi)
            elif param_key == "irho":
                sample[layer_name].material.irho.range(lo, hi)
            else:
                raise ValueError(
                    f"Unknown fit parameter '{param_key}' for layer "
                    f"'{layer_name}'. Supported: thickness, interface, "
                    f"rho, irho"
                )

    return experiment


def inspect_model(
    model_file: str | Path,
    q: np.ndarray | None = None,
    dq: np.ndarray | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Load a model and return metadata about its parameters.

    Args:
        model_file: Path to the YAML/JSON model file.
        q: Optional Q values (defaults to a standard grid).
        dq: Optional dQ values (defaults to 0.025 * Q).

    Returns:
        Dict with ``"variable"`` and ``"fixed"`` keys, each a list of
        dicts with ``name``, ``value``, ``bounds``, ``fixed`` fields.
    """
    if q is None:
        q = np.logspace(np.log10(0.008), np.log10(0.2), 50)
    if dq is None:
        dq = 0.025 * q

    experiment = load_experiment(model_file, q, dq)
    problem = FitProblem(experiment)

    variable = []
    for param in problem.parameters:
        variable.append(
            {
                "name": param.name,
                "value": param.value,
                "bounds": param.bounds,
                "fixed": False,
            }
        )

    fixed = []
    models = problem._models
    if models:
        struct = models[0].parameters()["sample"]["layers"]
        for layer_dict in struct:
            for _key, param in layer_dict.items():
                if isinstance(param, dict):
                    for _sk, sv in param.items():
                        if isinstance(sv, Parameter) and sv.fixed:
                            fixed.append(
                                {
                                    "name": sv.name,
                                    "value": sv.value,
                                    "bounds": sv.bounds,
                                    "fixed": True,
                                }
                            )
                elif isinstance(param, Parameter) and param.fixed:
                    fixed.append(
                        {
                            "name": param.name,
                            "value": param.value,
                            "bounds": param.bounds,
                            "fixed": True,
                        }
                    )

    return {"variable": variable, "fixed": fixed}


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------


def _validate_layers(layers: list) -> None:
    """Check that each layer dict has the required fields."""
    if not isinstance(layers, list) or len(layers) < 2:
        raise ValueError(
            "Model must have at least 2 layers (incident medium + substrate)"
        )
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"Layer {i} must be a dict, got {type(layer).__name__}")
        if "name" not in layer:
            raise ValueError(f"Layer {i} is missing required 'name' field")

        fit = layer.get("fit", {})
        for key, bounds in fit.items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(
                    f"Layer '{layer['name']}' fit.{key} must be "
                    f"a [min, max] list, got {bounds}"
                )
            if bounds[0] >= bounds[1]:
                raise ValueError(
                    f"Layer '{layer['name']}' fit.{key}: "
                    f"min ({bounds[0]}) must be less than max ({bounds[1]})"
                )


def _validate_experiment(experiment: object) -> None:
    """Validate the ``experiment`` section of the model description."""
    if not isinstance(experiment, dict):
        raise ValueError("'experiment' must be a mapping")
    allowed = set(EXPERIMENT_DEFAULTS)
    unknown = set(experiment) - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys in 'experiment': {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )
    if "q_min" in experiment and "q_max" in experiment:  # noqa: SIM102
        if experiment["q_min"] >= experiment["q_max"]:
            raise ValueError("experiment.q_min must be less than experiment.q_max")
    if "q_points" in experiment:
        qp = experiment["q_points"]
        if not isinstance(qp, int) or qp < 5 or qp > 1000:
            raise ValueError("experiment.q_points must be an integer in 5–1000")
    if "dq_over_q" in experiment and experiment["dq_over_q"] <= 0:
        raise ValueError("experiment.dq_over_q must be positive")
    if "relative_error" in experiment and experiment["relative_error"] <= 0:
        raise ValueError("experiment.relative_error must be positive")


def _validate_optimization(optimization: object) -> None:
    """Validate the ``optimization`` section of the model description."""
    if not isinstance(optimization, dict):
        raise ValueError("'optimization' must be a mapping")
    allowed = {
        "param",
        "param_values",
        "parameters_of_interest",
        "num_realizations",
        "mcmc_steps",
        "entropy_method",
    }
    unknown = set(optimization) - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys in 'optimization': {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )
    if "param" in optimization and not isinstance(optimization["param"], str):
        raise ValueError("optimization.param must be a string")
    if "param_values" in optimization:
        pv = optimization["param_values"]
        if not isinstance(pv, list) or not pv:
            raise ValueError("optimization.param_values must be a non-empty list")
    if "num_realizations" in optimization:
        nr = optimization["num_realizations"]
        if not isinstance(nr, int) or nr < 1 or nr > 100:
            raise ValueError(
                "optimization.num_realizations must be an integer in 1–100"
            )
    if "mcmc_steps" in optimization:
        ms = optimization["mcmc_steps"]
        if not isinstance(ms, int) or ms < 100 or ms > 100_000:
            raise ValueError(
                "optimization.mcmc_steps must be an integer in 100–100,000"
            )
    if "entropy_method" in optimization and optimization["entropy_method"] not in (
        "mvn",
        "kdn",
    ):
        raise ValueError("optimization.entropy_method must be 'mvn' or 'kdn'")
