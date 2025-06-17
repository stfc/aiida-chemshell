from enum import Enum, auto 


class ChemShellTheory(Enum):
    """Enum fr the ChemShell theory interfaces."""
    CASTEP    = auto() 
    CP2K      = auto() 
    DFTBP     = auto()
    FHI_AIMS  = auto() 
    GAMESS_UK = auto() 
    GAUSSIAN  = auto() 
    LSDALTON  = auto() 
    MNDO      = auto() 
    MOLPRO    = auto() 
    NWCHEM    = auto() 
    ORCA      = auto()
    PYSCF     = auto() 
    TURBOMOLE = auto()  