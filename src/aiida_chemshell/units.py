"""Unit conversion utilities."""


class UnitsConverter:
    """Unit conversion utility."""

    ANGSTROM = 0.529177
    EV = 27.2114

    @classmethod
    def angstrom_to_bohr(cls, val: float) -> float:
        """Convert Angstrom to Bohr (a.u.)."""
        return val * 1.8897259886

    @classmethod
    def bohr_to_angstrom(cls, val: float) -> float:
        """Convert bohr (a.u.) to Angstrom."""
        return val * UnitsConverter.ANGSTROM

    @classmethod
    def hartree_to_ev(cls, val: float) -> float:
        """Convert Hartree (a.u.) to eV."""
        return val * UnitsConverter.EV

    @classmethod
    def ev_to_hartree(cls, val: float) -> float:
        """Convert eV to Hartree (a.u.)."""
        return val / UnitsConverter.EV
