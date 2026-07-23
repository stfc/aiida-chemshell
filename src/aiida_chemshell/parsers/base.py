"""Defines the calculation parsers for the ChemShell AiiDA plugin."""

import json
from pathlib import Path

import numpy
from aiida.common import ModificationNotAllowed
from aiida.engine import ExitCode
from aiida.orm import ArrayData, Dict, Float, SinglefileData, TrajectoryData
from aiida.parsers.parser import Parser

from aiida_chemshell.calculations.base import ChemShellCalculation


class ChemShellParser(Parser):
    """AiiDA parser plugin for ChemShell calculations."""

    def parse(self, **kwargs):
        """Parse the output of a ChemShell calculation."""
        retrieved_tmp_folder = Path(kwargs.get("retrieved_temporary_folder", ""))

        if ChemShellCalculation.FILE_STDOUT not in self.retrieved.list_object_names():
            return self.exit_codes.ERROR_STDOUT_NOT_FOUND
        results_path = retrieved_tmp_folder / ChemShellCalculation.FILE_RESULTS
        if not (results_path).exists():
            return self.exit_codes.ERROR_RESULTS_FILE_NOT_FOUND

        # Read the 'json' formatted results file
        with open(results_path, "rb") as f:
            results = json.loads(f.read())

        # Extract the final energy
        try:
            self.out("energy", Float(results["energy"][0], label="Final SCF Energy"))
        except (KeyError, ValueError):
            return self.exit_codes.ERROR_MISSING_FINAL_ENERGY
        except ModificationNotAllowed as e:
            raise e

        # Extract gradients/hessian if they are requested
        if "calculation_parameters" in self.node.inputs:
            array_desc = "1st and/or 2nd derivatives calculated with ChemShell"
            if self.node.inputs.calculation_parameters.get("gradients", False):
                try:
                    gradients = numpy.array(results["gradients"])
                except (KeyError, ValueError):
                    return self.exit_codes.ERROR_MISSING_GRADIENTS
                else:
                    grad_data = ArrayData(
                        label="Energy Derivative Arrays",
                        description=array_desc,
                    )
                    grad_data.set_array("gradients", gradients)
                    self.out("gradients", grad_data)
            if self.node.inputs.calculation_parameters.get("hessian", False):
                try:
                    hessian = numpy.array(results["hessian"])
                except (KeyError, ValueError):
                    return self.exit_codes.ERROR_MISSING_GRADIENTS
                else:
                    if "gradients" in self.outputs:
                        self.outputs["gradients"].set_array("hessian", hessian)
                    else:
                        grad_data = ArrayData(
                            label="Energy Derivative Arrays",
                            description=array_desc,
                        )
                        grad_data.set_array("hessian", hessian)
                        self.out("gradients", grad_data)

        # If the calculation was a geometry optimisation, store the optimised structure
        if "optimisation_parameters" in self.node.inputs:
            dl_find_path = retrieved_tmp_folder / ChemShellCalculation.FILE_DLFIND
            if self.node.inputs.optimisation_parameters.get("thermal", False):
                self.parse_vibrational_analysis(
                    self.retrieved.get_object_content(
                        ChemShellCalculation.FILE_STDOUT, "r"
                    )
                )
            elif self.node.inputs.optimisation_parameters.get("neb", "no") in [
                "free",
                "frozen",
                "perpendicular",
            ]:
                self.parse_xyz_path(retrieved_tmp_folder / "nebpath.xyz", "neb_path")
                self.parse_neb_info(retrieved_tmp_folder / "nebinfo")
            elif dl_find_path.exists():
                descrip = "Optimised structure from a ChemShell optimisation"
                input_pk = self.node.inputs.structure.pk
                descrip += f" of node {input_pk}"
                if isinstance(self.node.inputs.structure, SinglefileData):
                    input_fname = self.node.inputs.structure.filename
                    descrip += f" ({input_fname})"
                # Store the optimised structure file
                with open(dl_find_path, "rb") as f:
                    self.out(
                        "optimised_structure",
                        SinglefileData(
                            file=f,
                            filename=ChemShellCalculation.FILE_DLFIND,
                            label="CJSON Structure File",
                            description=descrip,
                        ),
                    )
                self.parse_optimisation_path(
                    self.retrieved.get_object_content(
                        ChemShellCalculation.FILE_STDOUT, "r"
                    )
                )
            else:
                return self.exit_codes.ERROR_MISSING_OPTIMISED_STRUCTURE_FILE

            if self.node.inputs.optimisation_parameters.get("save_path", False):
                trj_path = retrieved_tmp_folder / ChemShellCalculation.FILE_TRJPTH
                trj_frc_path = retrieved_tmp_folder / ChemShellCalculation.FILE_TRJFRC
                if trj_path.exists():
                    self.parse_xyz_path(trj_path, "trajectory_path")
                    with open(trj_frc_path, "rb") as f:
                        self.out(
                            "trajectory_force",
                            SinglefileData(
                                file=f,
                                filename=ChemShellCalculation.FILE_TRJFRC.replace(
                                    "/", "_"
                                ),
                                label="ChemShell optimisation trajectory.",
                            ),
                        )
                else:
                    return self.exit_codes.ERROR_MISSING_OPTIMISED_STRUCTURE_FILE

        return ExitCode(0)

    def parse_vibrational_analysis(self, stdout: str) -> None:
        """Extract the vibrational analysis from ChemShell output log."""
        read = False
        energies = {}
        modes = []
        for line in stdout.split("\n"):
            if read:
                line_vals = line.split()
                if "Temperature:" in line:
                    energies[f"Temperature / {line_vals[2]}"] = float(line_vals[1])
                elif "E_electronic" in line:
                    energies[f"E_electronic correction / {line_vals[7]}"] = float(
                        line_vals[6]
                    )
                elif "total ZPE" in line:
                    energies[f"ZPE / {line_vals[3]}"] = float(line_vals[2])
                elif "total E vib" in line:
                    energies[f"Enthalpy / {line_vals[4]}"] = float(line_vals[3])
                elif "total S vib" in line:
                    energies[f"Entropy / {line_vals[4]}"] = float(line_vals[3])
                elif "Mode" in line or "total" in line:
                    pass
                else:
                    # All remaining lines should be part of the modes table
                    modes.append(numpy.array([float(x) for x in line_vals[2:]]))

            if "Thermochemical analysis" in line:
                read = True
            elif "total S vib" in line:
                read = False
        self.out("vibrational_energies", Dict(energies))
        modes = numpy.asarray(modes)
        modes_data_node = ArrayData(
            label="Vibrational Modes",
            description="Calculated vibrational modes for the system.",
        )
        modes_data_node.set_array("Modes", modes)
        self.out("vibrational_modes", modes_data_node)
        return

    def parse_optimisation_path(self, stdout: str) -> None:
        """Extract per step values from the optimisation job."""
        energies = []
        for line in stdout.split("\n"):
            if "Energy calculation finished" in line:
                energies.append(float(line.split()[-1]))

        results = ArrayData(
            label="Optimisation Path Properties",
            description="Values calculated at each step of an optimisation.",
        )
        results.set_array("energies", numpy.asarray(energies))
        self.out("optimisation_path", results)
        return

    def parse_xyz_path(self, file_path: Path, output_link: str) -> None:
        """Parse the NEB pathway into an AiiDA TrajectoryData node."""
        with open(file_path) as f:
            lines = f.readlines()
        natoms = int(lines[0])
        symbols = []
        positions = []
        step = 0
        i = 2
        while i < len(lines):
            step_positions = numpy.zeros((natoms, 3), dtype=float)
            for atm_index, atm_line in enumerate(lines[i : i + natoms]):
                line = atm_line.split()
                if step == 0:
                    symbols.append(line[0])
                step_positions[atm_index][0] = float(line[1])
                step_positions[atm_index][1] = float(line[2])
                step_positions[atm_index][2] = float(line[3])
            positions.append(step_positions)
            step += 1
            i += natoms + 2
        path = TrajectoryData()
        path.set_trajectory(symbols=symbols, positions=numpy.asarray(positions))
        path.label = "ChemShell (DL_FIND) optimisation path."
        path.description = "Path taken for a ChemShell Optimisation or NEB calculation."
        self.out(output_link, path)
        return

    def parse_neb_info(self, file_path: Path) -> None:
        """Parse the NEB info file into an AiiDA ArrayData noe."""
        output = ArrayData(
            label="Step information from an NEB calculation.",
            description=(
                "Calculated step values for an ChemShell NEB calculation from node: "
                f"{self.node.pk}"
            ),
        )
        with open(file_path) as f:
            lines = f.readlines()
        length = []
        energy = []
        work = []
        mass = []
        for line in lines[1:]:
            vals = line.split()
            length.append(float(vals[0]))
            energy.append(float(vals[1]))
            work.append(float(vals[2]))
            mass.append(float(vals[3]))
        output.set_array("path_length", numpy.asarray(length))
        output.set_array("energy", numpy.asarray(energy))
        output.set_array("work", numpy.asarray(work))
        output.set_array("effective_mass", numpy.asarray(mass))
        self.out("neb_info", output)
        return
