from enum import Enum, auto 


class ChemShellQMTheory(Enum):
    """Enum fr the ChemShell theory interfaces."""
    NONE      = auto() 
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

class ChemShellMMTheory(Enum):
    """ Enum for the ChemShell MM theory interfaces. """
    NONE      = auto() 
    DL_POLY   = auto() 
    GULP      = auto() 
    NAMD      = auto() 