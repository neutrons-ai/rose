"""
Neutron SLD database for common reflectometry materials.

Provides SLD lookup by material name, chemical formula, or alias.
Uses the ``periodictable`` package (bundled with refl1d) for
accurate neutron scattering length density computation.

Units: SLD values are in 10⁻⁶ Å⁻².

Example::

    >>> from rose.modeler.sld_database import get_sld, lookup_material
    >>> get_sld("silicon")
    2.074...
    >>> get_sld("D2O")
    6.335...
    >>> mat = lookup_material("gold")
    >>> mat.name, mat.sld
    ('Au', 4.66...)
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field

from periodictable import formula as pt_formula
from periodictable import neutron_sld

logger = logging.getLogger(__name__)

# -- Density table for compounds without elemental periodictable data ------

DENSITIES: dict[str, float] = {
    # Solvents
    "D2O": 1.107,
    "H2O": 1.00,
    "C7H8": 0.867,  # toluene
    "C7D8": 0.943,  # d-toluene
    "C2H6O": 0.789,  # ethanol
    "C4H8O": 0.889,  # THF
    "C3H6O": 0.784,  # acetone
    "C6H12": 0.779,  # cyclohexane
    # Oxides
    "SiO2": 2.20,
    "Al2O3": 3.98,
    "TiO2": 4.23,
    "Fe2O3": 5.24,
    "Cr2O3": 5.22,
    # Polymers
    "C8H8": 1.05,  # polystyrene
    "C8D8": 1.12,  # d-polystyrene
    "C5H8O2": 1.18,  # PMMA
    "C2H4O": 1.21,  # PEO
    "C2H6OSi": 0.97,  # PDMS
}

# -- Aliases: common names → chemical formulas ----------------------------

ALIASES: dict[str, str] = {
    # Substrates
    "silicon": "Si",
    "si": "Si",
    "silicon wafer": "Si",
    "quartz": "SiO2",
    "fused silica": "SiO2",
    "silica": "SiO2",
    "sapphire": "Al2O3",
    "alumina": "Al2O3",
    "germanium": "Ge",
    "ge": "Ge",
    # Solvents
    "heavy water": "D2O",
    "deuterated water": "D2O",
    "d2o": "D2O",
    "water": "H2O",
    "light water": "H2O",
    "h2o": "H2O",
    "toluene": "C7H8",
    "d-toluene": "C7D8",
    "deuterated toluene": "C7D8",
    "ethanol": "C2H6O",
    "thf": "C4H8O",
    "tetrahydrofuran": "C4H8O",
    "acetone": "C3H6O",
    "cyclohexane": "C6H12",
    # Metals
    "gold": "Au",
    "au": "Au",
    "copper": "Cu",
    "cu": "Cu",
    "nickel": "Ni",
    "ni": "Ni",
    "chromium": "Cr",
    "cr": "Cr",
    "titanium": "Ti",
    "ti": "Ti",
    "platinum": "Pt",
    "pt": "Pt",
    "silver": "Ag",
    "ag": "Ag",
    "iron": "Fe",
    "fe": "Fe",
    "aluminum": "Al",
    "aluminium": "Al",
    "al": "Al",
    # Oxides
    "silicon oxide": "SiO2",
    "silicon dioxide": "SiO2",
    "sio2": "SiO2",
    "aluminum oxide": "Al2O3",
    "titanium dioxide": "TiO2",
    "tio2": "TiO2",
    # Polymers
    "polystyrene": "C8H8",
    "ps": "C8H8",
    "deuterated polystyrene": "C8D8",
    "d-polystyrene": "C8D8",
    "dps": "C8D8",
    "pmma": "C5H8O2",
    "poly(methyl methacrylate)": "C5H8O2",
    "peo": "C2H4O",
    "poly(ethylene oxide)": "C2H4O",
    "pdms": "C2H6OSi",
    "polydimethylsiloxane": "C2H6OSi",
    # Special
    "air": "N2",  # approximate: 78% N₂
    "vacuum": "N2",
}

# SLD overrides for special cases where periodictable isn't accurate
_SLD_OVERRIDES: dict[str, float] = {
    "air": 0.0,
    "vacuum": 0.0,
    "N2": 0.0,  # when used as air proxy
}


# ------------------------------------------------------------------
# Data class
# ------------------------------------------------------------------


@dataclass
class Material:
    """A material with its neutron SLD.

    Attributes:
        name: Display name or formula.
        formula: Chemical formula string.
        density: Density in g/cm³ (may be ``None`` for elements).
        sld: Computed SLD in 10⁻⁶ Å⁻².
        aliases: Known aliases for this material.
    """

    name: str
    formula: str
    density: float | None = None
    sld: float = 0.0
    aliases: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def resolve_formula(name: str) -> str:
    """Resolve a material name or alias to a chemical formula.

    Args:
        name: Material name, alias, or formula.

    Returns:
        Chemical formula string.

    Raises:
        ValueError: If the name cannot be resolved.
    """
    key = name.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    # Try as-is (may be a formula already)
    try:
        pt_formula(name)
        return name
    except Exception as err:
        raise ValueError(
            f"Unknown material '{name}'. Use a chemical formula or a known alias."
        ) from err


def get_density(formula_str: str) -> float:
    """Get density for a formula from the density table or periodictable.

    Args:
        formula_str: Chemical formula.

    Returns:
        Density in g/cm³.

    Raises:
        ValueError: If density cannot be determined.
    """
    if formula_str in DENSITIES:
        return DENSITIES[formula_str]
    # Try periodictable (works for elements)
    try:
        f = pt_formula(formula_str)
        if hasattr(f, "density") and f.density is not None and f.density > 0:
            return float(f.density)
    except Exception:
        pass
    raise ValueError(
        f"Density not found for '{formula_str}'. "
        "Add it to DENSITIES or provide an explicit SLD."
    )


def compute_sld(formula_str: str, density: float | None = None) -> float:
    """Compute neutron SLD for a chemical formula.

    Args:
        formula_str: Chemical formula (e.g. ``"Si"``, ``"SiO2"``).
        density: Density override in g/cm³.  If ``None``, looked up
            from the density table or ``periodictable``.

    Returns:
        SLD in 10⁻⁶ Å⁻².

    Raises:
        ValueError: If the formula is invalid or density unavailable.
    """
    # Check overrides first
    if formula_str in _SLD_OVERRIDES:
        return _SLD_OVERRIDES[formula_str]

    if density is None:
        density = get_density(formula_str)

    f = pt_formula(formula_str)
    sld_real, _sld_imag, _sld_incoh = neutron_sld(f, density=density)
    return float(sld_real)


def get_sld(name_or_formula: str, density: float | None = None) -> float:
    """Get SLD for a material by name, alias, or formula.

    This is the main user-facing function.

    Args:
        name_or_formula: Material name (e.g. ``"silicon"``),
            alias (e.g. ``"heavy water"``), or formula (``"D2O"``).
        density: Optional density override in g/cm³.

    Returns:
        SLD in 10⁻⁶ Å⁻².

    Example:
        >>> get_sld("silicon")
        2.074...
        >>> get_sld("D2O")
        6.335...
        >>> get_sld("gold")
        4.66...
    """
    # Special cases
    key = name_or_formula.strip().lower()
    if key in ("air", "vacuum"):
        return 0.0

    formula_str = resolve_formula(name_or_formula)
    return compute_sld(formula_str, density=density)


def lookup_material(name: str) -> Material:
    """Look up a material and return a :class:`Material` object.

    Args:
        name: Material name, alias, or formula.

    Returns:
        Populated ``Material`` with SLD computed.

    Raises:
        ValueError: If the material cannot be resolved.
    """
    key = name.strip().lower()
    if key in ("air", "vacuum"):
        return Material(name=key, formula="", sld=0.0, aliases=["air", "vacuum"])

    formula_str = resolve_formula(name)
    density = None
    with contextlib.suppress(ValueError):
        density = get_density(formula_str)
    sld = compute_sld(formula_str, density=density)

    # Collect aliases for this formula
    aliases = [k for k, v in ALIASES.items() if v == formula_str]

    return Material(
        name=formula_str,
        formula=formula_str,
        density=density,
        sld=sld,
        aliases=aliases,
    )


def list_materials() -> list[Material]:
    """Return a list of all known materials.

    Returns:
        List of :class:`Material` objects for every unique formula
        in the alias table.
    """
    seen: set[str] = set()
    materials: list[Material] = []
    for formula_str in sorted(set(ALIASES.values())):
        if formula_str in seen:
            continue
        seen.add(formula_str)
        try:
            mat = lookup_material(formula_str)
            materials.append(mat)
        except ValueError:
            logger.debug("Skipping '%s': density not available", formula_str)
    return materials
