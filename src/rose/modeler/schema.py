"""
Query input for use-case 2.

The user provides a plain-text description of their sample and
scientific question.  The LLM infers everything else — layer stack,
SLD values, fit ranges, optimisation target, and instrument settings.

Example (text file)::

    Polystyrene thin film (~50 nm) on a gold adhesion layer
    deposited on a silicon substrate, measured in air.
    Find the optimal gold layer thickness to maximise sensitivity
    to changes in the polymer film thickness.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class PlanQuery(BaseModel):
    """Top-level query model for use-case 2.

    The user provides a free-text description of their sample and
    hypothesis.  The LLM generates the complete ROSE model YAML.

    Attributes:
        description: Plain-text description of the sample, hypothesis,
            and what to optimise.
    """

    description: str = Field(min_length=10, max_length=10000)


def load_query(path: str) -> PlanQuery:
    """Load a query from a text or YAML file.

    Supports two formats:

    - **Plain text** (``.txt``): The entire file content is the
      description.
    - **YAML** (``.yaml`` / ``.yml``): Must contain a top-level
      ``description`` key.

    Args:
        path: Path to the query file.

    Returns:
        Validated :class:`PlanQuery`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: On unsupported file format or missing description.
    """
    query_path = Path(path)
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")

    text = query_path.read_text(encoding="utf-8").strip()
    suffix = query_path.suffix.lower()

    if suffix == ".txt":
        return PlanQuery(description=text)

    if suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
        if not isinstance(data, dict) or "description" not in data:
            raise ValueError(
                "YAML query file must contain a top-level 'description' key"
            )
        return PlanQuery(description=data["description"])

    raise ValueError(
        f"Unsupported query file format '{suffix}'. Use .txt, .yaml, or .yml"
    )
