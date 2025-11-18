"""CLF-ULTRA ChemShell + Janus workflow."""

from aiida.engine import ToContext, WorkChain
from aiida.orm import Code, Dict, Float, Int, List, SinglefileData, StructureData
from aiida.plugins.factories import CalculationFactory

ChemShellCalculation = CalculationFactory("chemshell")


class CLFULTRAOptimisationWorkChain(WorkChain):
    """
    CLF-ULTRA Geometry Optimisation WorkChain.

    This workchain will perform a QM or QM/MM based geometry optimisation
    on a given structure and generate etxXYZ formatted trajectory files for
    every step in the optimisation. It uses the NWChem and DL_POLY backends
    and the B3LYP dft functional for the QM calculations.
    """

    @classmethod
    def define(cls, spec) -> None:
        """Define the AiiDA process specification for the WorkChain."""
        super().define(spec)
        spec.input(
            "structure",
            valid_type=(SinglefileData, StructureData),
            required=True,
            help="The structure on which to perform the geometry optimisation",
        )
        spec.input("chemshell", valid_type=Code, required=True, help="")

        spec.input(
            "force_field_file",
            valid_type=SinglefileData,
            required=False,
            help="File defining the MM force field for QM/MM calculation.",
        )
        spec.input(
            "qm_region",
            valid_type=List,
            required=False,
            help=(
                "A list of atom indexes to apply the QM portion of a QM/MM "
                "calculation to."
            ),
        )

        spec.output("final_energy", valid_type=Float)
        spec.output("natms", valid_type=Int)

        spec.outline(cls.optimise, cls.generate_xyz_files, cls.result)
        return

    def optimise(self):
        """Perform the Geometry Optimisation Task."""
        inputs = {
            "qm_parameters": Dict(
                {
                    "theory": "NWChem",
                    "method": "dft",
                    "functional": "b3lyp",
                    "basis": "cc-pvtz",
                    # Add D3 correction
                }
            ),
            "optimisation_parameters": Dict({"save_path": True}),
            "structure": self.inputs.structure,
            "code": self.inputs.chemshell,
            "metadata": {
                "options": {
                    "resources": {"num_mpiprocs_per_machine": 4, "num_machines": 1},
                    "withmpi": True,
                }
            },
        }
        future = self.submit(ChemShellCalculation, **inputs)
        return ToContext(optimise=future)

    def generate_xyz_files(self):
        """Convert the paths to individual extended XYZ files."""
        trjp = self.ctx.optimise.outputs.trajectory_path.get_content(mode="r")
        trjf = self.ctx.optimise.outputs.trajectory_force.get_content(mode="r")
        trjp = trjp.split("\n")
        trjf = trjf.split("\n")
        natms = int(trjp[0])
        self.out("natms", natms)
        return

    def result(self) -> None:
        """Report the final results."""
        self.out("final_energy", self.ctx.optimise.outputs.energy)
        return
