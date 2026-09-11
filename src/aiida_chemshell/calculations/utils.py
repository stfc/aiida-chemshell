"""A collection of smaller utility AiiDA CalcFunctions."""

import re
from io import BytesIO

from aiida.engine import calcfunction
from aiida.orm import (
    ArrayData,
    Dict,
    Float,
    SinglefileData,
    StructureData,
    TrajectoryData,
)


@calcfunction
def create_dictionary(atoms, energies) -> Dict:
    """Collate a series of isolated atom energies into a dictionary output."""
    if len(atoms) != len(energies):
        raise ValueError(
            f"Mismatched lengths: Got {len(atoms)} atoms and {len(energies)} energies."
        )
    return Dict(dict(zip(atoms, energies, strict=False)))


@calcfunction
def extract_structures_from_xyz(file: SinglefileData) -> dict[str, StructureData]:
    """
    Split a multi-frame XYZ file into individual StructureData nodes.

    Each frame in the (optionally extended) XYZ file is parsed into its own
    StructureData node. Extended XYZ 'Lattice=' and 'pbc=' fields on the comment
    line are honoured when present; frames without them are treated as
    non-periodic.

    Parameters
    ----------
    file : SinglefileData
        A SinglefileData node wrapping an XYZ (or extended XYZ) file that may
        contain one or more frames.

    Returns
    -------
    dict[str, StructureData]
        A mapping of output link labels to StructureData nodes, one per frame.
        Keys are of the form '<filename>_frame_<index>' (zero-indexed, with
        spaces in the filename replaced by underscores). As a calcfunction
        return value, this dictionary is registered as a namespace of output
        nodes.

    Raises
    ------
    Exception
        If the file is not valid XYZ, is truncated, or contains a malformed
        atom line.
    """
    with file.open(mode="r") as f:
        lines = f.readlines()

    structures = {}
    line_count = len(lines)
    i = 0
    frame_idx = 0

    while i < line_count:
        line = lines[i].strip()
        try:
            natoms = int(line)
        except ValueError as e:
            raise Exception("Invalid XYZ format detected.") from e
        if (i + 2 + natoms) > line_count:
            raise Exception("XYZ file truncation detected.")

        # Create the base StructureData object
        structure = StructureData(pbc=(False, False, False))

        # Read the comment line
        i += 1
        line = lines[i].strip()
        cell = None
        if "Lattice=" in line:
            match = re.search(r'Lattice="([^"]+)"', line)
            if match:
                lat_vals = [float(x) for x in match.group(1).split()]
                if len(lat_vals) == 9:
                    cell = [lat_vals[0:3], lat_vals[3:6], lat_vals[6:9]]
            pbc = [True, True, True]
            if "pbc=" in line:
                match_pbc = re.search(r'pbc="([^"]+)"', line)
                if match_pbc:
                    pbc_vals = match_pbc.group(1).split()
                    if len(pbc_vals) == 3:
                        # Robust check: converts 'T', 'True', or '1' to True
                        pbc = [val.upper() in ["T", "TRUE", "1"] for val in pbc_vals]

            # Assign the parse cell parameters to the StructureData object
            structure.cell = cell
            structure.pbc = pbc

        i += 1
        for atmi in range(natoms):
            atom_line = lines[i + atmi].strip().split()
            if len(atom_line) < 4:
                raise Exception(
                    f"Invalid atom entry in xyz file: {line[i + atmi].strip()}"
                )

            structure.append_atom(
                position=[
                    float(atom_line[1]),
                    float(atom_line[2]),
                    float(atom_line[3]),
                ],
                symbols=atom_line[0],
            )

        structures[
            f"{file.filename.replace(' ', '_').strip('.xyz')}_frame_{frame_idx}"
        ] = structure
        frame_idx += 1
        i += natoms

    return structures


@calcfunction
def combine_into_extended_xyz(
    **kwargs: StructureData | TrajectoryData | Float | ArrayData,
) -> SinglefileData:
    """
    Collate a batch of single-point results into one extended XYZ file.

    The keyword nodes are sorted by type into three parallel lists: structures
    (from StructureData, or each frame of a TrajectoryData), energies (Float)
    and, optionally, force arrays (ArrayData). They are then written out as an
    extended XYZ file with an 'Energy=' label per frame and, when forces are
    supplied, a 'force' property column. The lists are paired positionally, so
    the caller is responsible for passing matched sets of nodes (equal numbers
    of structures, energies and, if used, arrays).

    Parameters
    ----------
    **kwargs : StructureData | TrajectoryData | Float | ArrayData
        Arbitrary keyword arguments whose values are the result nodes to
        combine. Only the values are used; the keys are ignored. Forces are read
        from the 'gradients' array of any ArrayData node.

    Returns
    -------
    SinglefileData
        A SinglefileData node wrapping the combined
        'chemshell_batch_workchain.extxyz' file.

    Raises
    ------
    ValueError
        If an unsupported node type is passed, if no structures are provided, or
        if the numbers of structures, energies and (when given) arrays do not
        match.
    """
    structures: list[StructureData] = []
    energies: list[Float] = []
    arrays: list[ArrayData] = []

    # Extract individual structures and SP calculation results.
    for _key, node in kwargs.items():
        if isinstance(node, TrajectoryData):
            for tindex in range(node.numsteps):
                structures.append(node.get_step_structure(tindex))
        elif isinstance(node, StructureData):
            structures.append(node)
        elif isinstance(node, Float):
            energies.append(node)
        elif isinstance(node, ArrayData):  # Ensure after TrajectoryData !!
            arrays.append(node)
        else:
            raise ValueError(
                f"Invalid input of type {node} to 'combine_into_extended_xyz' "
                "calcfunction."
            )
    if not structures:
        raise ValueError("No StructureData nodes were provided.")
    if len(structures) != len(energies):
        raise ValueError(
            f"Mismatched input lengths: received {len(structures)} structures and "
            f"{len(energies)} energies."
        )
    if arrays:
        if len(arrays) != len(energies):
            raise ValueError(
                f"Mismatched input lengths: received {len(arrays)} arrays and "
                f"{len(energies)} energies."
            )

    xyz_lines = []

    for i in range(len(structures)):
        structure = structures[i]
        energy = energies[i]

        xyz_lines.append(str(len(structure.sites)))

        header_line = f"Energy={energy.value}"
        if arrays:
            forces = arrays[i].get_array("gradients")
            header_line += " Properties=species:S:1:pos:R:3:force:R:3"
        else:
            forces = None
            header_line += " Properties=species:S:1:pos:R:3"
        xyz_lines.append(header_line)

        for j, site in enumerate(structure.sites):
            kind = structure.get_kind(site.kind_name)
            symbol = kind.symbols[0]
            x, y, z = site.position
            site_line = f"{symbol:<4} {x:15.8f} {y:15.8f} {z:15.8f}"
            if forces is not None:
                fx, fy, fz = forces[j]
                site_line += f" {fx:15.8f} {fy:15.8f} {fz:15.8f}"
            xyz_lines.append(site_line)

    content = "\n".join(xyz_lines)
    stream = BytesIO(content.encode("utf-8"))

    return SinglefileData(file=stream, filename="chemshell_batch_workchain.extxyz")
