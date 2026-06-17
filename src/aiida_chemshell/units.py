"""Unit conversion utilities."""


class UnitsConverter:
    """Unit conversion utility."""

    ANGSTROM = 0.529177

    @classmethod
    def angstrom_to_bohr(cls, val: float) -> float:
        """Convert Angstrom to Bohr (a.u.)."""
        return val * 1.8897259886

    @classmethod
    def bohr_to_angstrom(cls, val: float) -> float:
        """Convert bohr (a.u.) to Angstrom."""
        return val * UnitsConverter.ANGSTROM
