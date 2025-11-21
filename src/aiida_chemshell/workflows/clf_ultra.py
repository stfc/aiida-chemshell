"""CLF-ULTRA ChemShell + Janus workflow."""

from aiida.engine import ToContext, WorkChain, calcfunction
from aiida.orm import (
    Code,
    Dict,
    Float,
    FolderData,
    List,
    SinglefileData,
    Str,
    StructureData,
)
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
        spec.input("janus", valid_type=Code, required=True, help="")

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

        spec.output("final_qm_energy", valid_type=Float)
        # spec.output("mlip_outputs", valid_type=Dict)
        spec.output("trajectory_files", valid_type=FolderData)
        spec.output("energy_difference", valid_type=Float)

        spec.outline(
            cls.optimise,
            cls.generate_xyz_files,
            cls.extract_final_structure,
            cls.mlip_sp,
            cls.calculate_energy_difference,
            cls.result,
        )
        return

    def optimise(self):
        """Perform the Geometry Optimisation Task."""
        inputs = {
            "qm_parameters": Dict(
                {
                    "theory": "NWChem",  # NWChem
                    "method": "dft",  # DFT
                    "functional": "B3LYP",  # B3LYP
                    "basis": "cc-pvtz",  # cc=pvtz
                    # "d3": True # Add D3 correction
                }
            ),
            "optimisation_parameters": Dict({"save_path": True}),
            "structure": self.inputs.structure,
            "code": self.inputs.chemshell,
            "metadata": {
                "options": {
                    "resources": {"num_mpiprocs_per_machine": 8, "num_machines": 1},
                    # "withmpi": True,
                }
            },
        }
        future = self.submit(ChemShellCalculation, **inputs)
        return ToContext(optimise=future)

    def generate_xyz_files(self):
        """Convert the paths to individual extended XYZ filesn for each step."""
        inputs = {
            "path": self.ctx.optimise.outputs.trajectory_path,
            "force": self.ctx.optimise.outputs.trajectory_force,
            "code": self.inputs.chemshell,
            "metadata": {
                "options": {
                    "resources": {"num_mpiprocs_per_machine": 2, "num_machines": 1},
                    # "withmpi": True,
                }
            },
        }
        from aiida_chemshell.calculations.splt_trajectory import SplitTrajectory

        future = self.submit(SplitTrajectory, **inputs)
        return ToContext(split_trajectory=future)

    def extract_final_structure(self):
        """Extract the optimised structure from the XYZ trajectory folder."""
        self.ctx.final_structure = extract_final_structure(
            self.ctx.split_trajectory.outputs.trajectory_folder
        )
        return

    def mlip_sp(self):
        """Run a single point energy with Janus-Core on the optimised structure."""
        structure = self.ctx.final_structure
        from aiida_mlip.helpers.help_load import load_model

        model = load_model(None, "mace_mp")
        inputs = {
            "metadata": {"options": {"resources": {"num_machines": 1}}},
            "code": self.inputs.janus,
            # "arch": Str(model.architecture),
            "struct": structure,
            "model": model,
            "device": Str("cpu"),
            # "calc_kwargs": Dict({"dispersion": True}),
        }
        mlip_sp = CalculationFactory("mlip.sp")
        future = self.submit(mlip_sp, **inputs)
        return ToContext(mlip_sp=future)

    def calculate_energy_difference(self):
        """Calculate the difference in final energies."""
        self.ctx.energy_difference = calculate_difference(
            self.ctx.optimise.outputs.energy, self.ctx.mlip_sp.outputs.results_dict
        )
        return

    def result(self) -> None:
        """Report the final results."""
        self.out("final_qm_energy", self.ctx.optimise.outputs.energy)
        self.out(
            "trajectory_files", self.ctx.split_trajectory.outputs.trajectory_folder
        )
        # self.out(
        #     "mlip_outputs",
        #     self.ctx.mlip_sp.outputs.results_dict
        # )
        self.out("energy_difference", self.ctx.energy_difference)
        return


@calcfunction
def extract_final_structure(folder: FolderData) -> StructureData:
    """Extract the final optimised structure from a folder of xyz files."""
    structure_file = folder.list_object_names()[-1]
    structure = StructureData()
    structure._parse_xyz(folder.get_object_content(structure_file, mode="r") + "\n")
    return structure


@calcfunction
def calculate_difference(qm_energy: Float, ml_results: Dict) -> Float:
    """Calculate the difference between the final energies."""
    ml_energy = ml_results["info"]["mace_mp_energy"]
    diff = qm_energy - ml_energy
    return Float(
        diff,
        label="Final Energy Difference",
        description=(
            "Difference in final energies between DFT calculation and "
            "MACE_mp energy calculation using Janus-core."
        ),
    )
