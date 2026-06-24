"""Workflows for geometry optimisation based taks."""

from aiida.common.exceptions import MissingEntryPointError
from aiida.engine import ToContext, WorkChain
from aiida.orm import ArrayData, Bool, Code, Dict, Float, SinglefileData
from aiida.plugins.factories import CalculationFactory

from aiida_chemshell.calculations.base import ChemShellCalculation
from aiida_chemshell.workflows.isolated_atoms import IsolatedAtomicEnergiesWorkChain


class GeometryOptimisationWorkChain(WorkChain):
    """Geometry optimisation calculation with extended optional calculation options."""

    @classmethod
    def define(cls, spec) -> None:
        """Define the AiiDA process specification for the WorkChain."""
        super().define(spec)

        ## Inputs ##
        spec.expose_inputs(ChemShellCalculation, namespace="chemsh")

        spec.input(
            "vibrational_analysis",
            valid_type=Bool,
            required=False,
            help="Calculate vibrational modes of resulting structure. (default=True)",
        )

        ## Outputs ##
        spec.output(
            "final_energy",
            valid_type=Float,
            required=True,
            help="The final energy for the optimised structure.",
        )
        spec.output(
            "optimised_structure",
            valid_type=SinglefileData,
            required=True,
            help="The final optimised geometry of the given structure.",
        )
        spec.output(
            "vibrational_energies",
            valid_type=Dict,
            required=False,
            help="The calculated thermochemical properties of the optimised structure",
        )
        spec.output(
            "vibrational_modes",
            valid_type=ArrayData,
            required=False,
            help="The calculated vibrational modes for the optimised structure.",
        )

        # Optional inputs/outputs for using the results to fine tune a MLIP model
        # using the janus-core project via the aiida-mlip plugin.
        try:
            CalculationFactory("mlip.train")
        except MissingEntryPointError:
            pass
        else:
            from aiida_mlip.data.model import ModelData

            spec.input(
                "mlip_model",
                valid_type=ModelData,
                required=False,
                help="The MLIP foundation model to apply fine-tuning to.",
            )
            spec.input(
                "mlip_code",
                valid_type=Code,
                required=False,
                help="The Janus-core AiiDA code instance.",
            )
            spec.output(
                "fine_tuned_model", valid_type=ModelData, required=False, help=""
            )
            spec.output(
                "fine_tuned_model_compiled",
                valid_type=SinglefileData,
                required=False,
                help="",
            )

        ## Workflow ##
        spec.outline(
            cls.optimise,
            cls.energy,
            cls.isolated_atom_energies,
            cls.generate_mlip_training_inputs,
            cls.train_mlip,
            cls.result,
        )

        return

    def optimise(self):
        """Perform the geometry optimisation."""
        inputs = self.exposed_inputs(ChemShellCalculation, namespace="chemsh")
        if "qm_parameters" not in inputs:
            inputs["qm_parameters"] = Dict(
                {
                    "theory": "NWChem",
                    "method": "dft",
                    "functional": "B3LYP",
                    "basis": "cc-pvdz",
                }
            )
        if "force_field_file" in inputs:
            if "mm_parameters" not in inputs:
                inputs["mm_parameters"] = Dict({"theory": "DL_POLY"})
            if "qmmm_parameters" not in inputs:
                inputs["qmmm_parameters"] = Dict({"qm_region": []})
        elif "mm_parameters" in inputs:
            return None
        if "optimisation_parameters" not in inputs:
            inputs["optimisation_parameters"] = Dict({})
        if "mlip_model" in self.inputs:
            inputs["optimisation_parameters"]["save_path"] = True

        future = self.submit(ChemShellCalculation, **inputs)
        future.label = ChemShellCalculation.default_process_label(future)
        future.description = (
            f"Geometry optimisation step from WorkChainNode pk: {self.node.pk}"
        )
        return ToContext(optimise=future)

    def energy(self):
        """Perform a single point energy calculation on the optimised structure."""
        if self.inputs.get("vibrational_analysis", False):
            inputs = {
                "code": self.exposed_inputs(ChemShellCalculation, namespace="chemsh")[
                    "code"
                ],
                "metadata": self.exposed_inputs(
                    ChemShellCalculation, namespace="chemsh"
                )["metadata"],
                "structure": self.ctx.optimise.outputs.optimised_structure,
                "qm_parameters": self.ctx.optimise.inputs.qm_parameters,
            }
            if "force_field_file" in self.ctx.optimise.inputs:
                inputs["force_field_file"] = self.ctx.optimise.inputs.force_field_file
                inputs["mm_parameters"] = self.ctx.optimise.inputs.mm_parameters
                inputs["qmmm_parameters"] = self.ctx.optimise.inputs.qmmm_parameters
            inputs["optimisation_parameters"] = (
                self.ctx.optimise.inputs.optimisation_parameters.get_dict()
            )
            inputs["optimisation_parameters"]["thermal"] = True
            future = self.submit(ChemShellCalculation, **inputs)
            future.label = ChemShellCalculation.default_process_label(future)
            future.description = (
                f"Vibrational frequency calculation step from WorkChainNode "
                f"pk: {self.node.pk}"
            )
            return ToContext(energy=future)
        return None

    def isolated_atom_energies(self):
        """Calculate isolated atomic energies for all species in the input structure."""
        if "mlip_model" in self.inputs:
            inputs = {
                "structure": self.inputs.chemsh.structure,
                "code": self.inputs.chemsh.code,
                "qm_parameters": self.ctx.optimise.inputs.qm_parameters,
            }
            future = self.submit(IsolatedAtomicEnergiesWorkChain, **inputs)
            future.label = "Isolated Atomic Energy WorkChain"
            future.description = (
                f"Isolated atom energies extracted from Node: "
                f"{self.inputs.structure.pk} for ChemShell optimisation "
                f"WorkChain: {self.node.pk} to be used for MLIP fine-tuning."
            )
            return ToContext(isolated_atoms=future)
        return None

    def generate_mlip_training_inputs(self):
        """Convert the optimisation path files to Janus compatible inputs."""
        if "mlip_model" in self.inputs:
            inputs = {
                "path": self.ctx.optimise.outputs.trajectory_path,
                "force": self.ctx.optimise.outputs.trajectory_force,
                "energies": self.ctx.optimise.outputs.optimisation_path,
                "atom_energies": self.ctx.isolated_atoms.outputs.atom_energies,
                "code": self.inputs.chemsh.code,
                "metadata": {
                    "options": {
                        "resources": {"num_mpiprocs_per_machine": 2, "num_machines": 1},
                        # "withmpi": True,
                    }
                },
            }
            from aiida_chemshell.calculations.file_conversion import (
                CreateJanusTrainingInputsCalcJob,
            )

            future = self.submit(CreateJanusTrainingInputsCalcJob, **inputs)
            future.label = "Generate MLIP training data set from geometry optimisation."
            future.description = (
                f"Data extraction step from WorkChainNode pk: {self.node.pk}"
            )
            return ToContext(create_mlip_inputs=future)
        return None

    def train_mlip(self):
        """Train a given MLIP model."""
        try:
            mlip_train_calc = CalculationFactory("mlip.train")
        except MissingEntryPointError:
            pass
        else:
            pass
            if "mlip_model" in self.inputs:
                # This needs to be properly addressed within aiida-mlip
                computer = self.inputs.get("mlip_code", None).computer
                work_dir = computer.get_workdir()
                with open("train.xyz", mode="wb") as f:
                    f.write(self.ctx.create_mlip_inputs.outputs.training_input.content)
                with open("test.xyz", mode="wb") as f:
                    f.write(self.ctx.create_mlip_inputs.outputs.test_input.content)
                with open("valid.xyz", mode="wb") as f:
                    f.write(
                        self.ctx.create_mlip_inputs.outputs.validation_input.content
                    )
                with computer.get_transport() as transport:
                    from pathlib import Path

                    transport.putfile(
                        Path("train.xyz").absolute(), f"{work_dir}/train.xyz"
                    )
                    transport.putfile(
                        Path("test.xyz").absolute(), f"{work_dir}/test.xyz"
                    )
                    transport.putfile(
                        Path("valid.xyz").absolute(), f"{work_dir}/valid.xyz"
                    )

                import yaml
                from aiida_mlip.data.config import JanusConfigfile

                from aiida_chemshell.utils import generate_default_mlip_fine_tune_config

                config_dict = generate_default_mlip_fine_tune_config()
                config_dict["train_file"] = f"{work_dir}/train.xyz"
                config_dict["test_file"] = f"{work_dir}/test.xyz"
                config_dict["valid_file"] = f"{work_dir}/valid.xyz"
                config_dict["name"] = "ChemShell_Workflow_Test"

                with open("mlip_config.yml", "w+") as f:
                    yaml.dump(config_dict, f, default_flow_style=False)

                mlip_inputs = {
                    "mlip_config": JanusConfigfile(Path("mlip_config.yml").absolute()),
                    "code": self.inputs.get("mlip_code", None),
                    "fine_tune": True,
                    "foundation_model": self.inputs.get("mlip_model", None),
                    "metadata": {"options": {"resources": {"num_machines": 1}}},
                }
                # Submit the mlip training job
                future = self.submit(mlip_train_calc, **mlip_inputs)
                future.label = "MLIP Fine-Tuning."
                future.description = (
                    f"MLIP fine-tuning step from WorkChainNode pk: {self.node.pk}"
                )
                return ToContext(mlip_training=future)
        return None

    def result(self):
        """Extract the final workflow results."""
        if "mlip_model" in self.inputs:
            self.out("optimised_structure", self.ctx.optimise.outputs.trajectory_path)
            self.out("fine_tuned_model", self.ctx.mlip_training.outputs.model)
            self.out(
                "fine_tuned_model_compiled",
                self.ctx.mlip_training.outputs.compiled_model,
            )
        else:
            self.out(
                "optimised_structure", self.ctx.optimise.outputs.optimised_structure
            )
        if self.inputs.get("vibrational_analysis", False):
            self.out("final_energy", self.ctx.energy.outputs.energy)
            self.out(
                "vibrational_energies", self.ctx.energy.outputs.vibrational_energies
            )
            self.out("vibrational_modes", self.ctx.energy.outputs.vibrational_modes)
        else:
            self.out("final_energy", self.ctx.optimise.outputs.energy)
        return
