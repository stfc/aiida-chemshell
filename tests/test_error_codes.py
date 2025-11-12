"""Tests for error code generation on failed job."""

from aiida.orm import Dict

from aiida_chemshell.calculations.base import ChemShellCalculation


def test_qm_theory_validation(generate_calcjob, generate_inputs):
    """Test the validation of qm theory input."""
    inputs = generate_inputs(qm={"theory": "INVALID"}, structure_fname="butanol.cjson")
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(
            f"Wrong error code generated from QM theory input validation: {str(e)}"
        ) from e
    else:
        raise AssertionError("No error caught when providing invalid QM theory key.")


def test_mm_theory_validation(generate_calcjob, generate_inputs):
    """Test the validation of mm theory input."""
    inputs = generate_inputs(
        mm={"theory": "INVALID"}, structure_fname="butanol.cjson", ff_fname="butanol.ff"
    )
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(
            f"Wrong error code generated from MM theory input validation: {str(e)}"
        ) from e
    else:
        raise AssertionError("No error caught when providing invalid MM theory key.")


def test_structure_validation(generate_calcjob, get_test_data_file):
    """Test error catching for when no input structure is provided."""
    # Test if no structure is given
    inputs = Dict(
        {
            "qm_parameters": {"theory": "NWChem"},
        }
    )
    try:
        tmp_pth, calc_info = generate_calcjob(ChemShellCalculation, inputs)
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(
            f"Wrong error caught during no structure file validation: {str(e)}"
        ) from e
    else:
        raise AssertionError("No error caught when providing no structure file.")

    # Test if structure file is of wrong format
    inputs = Dict(
        {
            "qm_theory": {"theory": "NWChem"},
            "structure": get_test_data_file("butanol.ff"),
        }
    )
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError as e:
        assert (
            "Structure file must be either an '.xyz', '.pun' or "
            "'.cjson' formatted structure file."
        ) in str(e)
    except Exception as e:
        raise AssertionError(
            f"Wrong error caught during invalid structure file type: {str(e)}"
        ) from e
    else:
        raise AssertionError(
            "No error caught when providing invalid structure file type."
        )

    # Test if given raw string not SinglefileData object
    inputs = Dict({"qm_theory": {"theory": "NWChem"}, "structure": "test.pdb"})
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(
            f"Wrong error caught when provided invalid SinglefileData object: {str(e)}"
        ) from e
    else:
        raise AssertionError(
            "No error caught when provided invalid SinglefileData object."
        )


def test_sp_calculation_input_validation(generate_calcjob, generate_inputs):
    """Test the parameter validation for the basic sp calculation inputs."""
    inputs = generate_inputs(sp={"gadents": True})
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError as e:
        assert "parameter keys are invalid" in str(e)
    except Exception as e:
        raise AssertionError(
            f"Incorrect error caught during single point parameter validation: {str(e)}"
        ) from e
    else:
        raise AssertionError(
            "No error caught during single point parameter validation."
        )

    # Test gradient input must be bool
    inputs = generate_inputs(sp={"gradients": "true"})
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError as e:
        assert "must be a Boolean value" in str(e)
    except Exception as e:
        raise AssertionError(
            f"Incorrect error caught during single point parameter validation: {str(e)}"
        ) from e
    else:
        raise AssertionError(
            "No error caught during single point parameter validation."
        )

    inputs = generate_inputs(sp={"hessian": "false"})
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError as e:
        assert "must be a Boolean value" in str(e)
    except Exception as e:
        raise AssertionError(
            f"Incorrect error caught during single point parameter validation: {str(e)}"
        ) from e
    else:
        raise AssertionError(
            "No error caught during single point parameter validation."
        )


def test_optimisation_input_validation(generate_calcjob, generate_inputs):
    """Test the parameter validation for the basic sp calculation inputs."""
    inputs = generate_inputs(opt={"mincycle": 5, "maxcycle": 100})
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError as e:
        assert "mincycle" in str(e)
    except Exception as e:
        raise AssertionError(
            f"Wrong error caught during optimisation parameter validation: {str(e)}"
        ) from e
    else:
        raise AssertionError(
            "No error caught during optimisation parameter validation."
        )


def test_qm_input_validation(generate_calcjob, generate_inputs):
    """Test the parameter validation for the QM inputs."""
    inputs = generate_inputs(
        qm={"theory": "NWChem", "method": "HF", "basis": "3-21G", "chrg": -1.0}
    )
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError as e:
        assert "chrg" in str(e)
    except Exception as e:
        raise AssertionError(
            f"Wrong error caught during QM parameter validation: {str(e)}"
        ) from e
    else:
        raise AssertionError("No error caught during QM parameter validation.")
    inputs = generate_inputs(
        qm={
            "theory": "NWChem",
            "method": "HF",
            "basis": "3-21G",
            "charge": -1.0,
            "direct": True,
            "diis": True,
            "mult": "1",
        }
    )
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError as e:
        assert "mult" in str(e)
        assert "must be of type float" in str(e)
    except Exception as e:
        raise AssertionError(
            f"Wrong error caught during QM parameter validation: {str(e)}"
        ) from e
    else:
        raise AssertionError("No error caught during QM parameter validation.")

    inputs = generate_inputs(qm={"theory": "NWChem", "method": "KF"})
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError as e:
        assert "method key ('KF') is not valid" in str(e)
    except Exception as e:
        raise AssertionError(
            f"Wrong error caught during QM parameter validation: {str(e)}"
        ) from e
    else:
        raise AssertionError("No error caught during QM parameter validation.")

    inputs = generate_inputs(qm={"theory": "NWChem", "scftype": "KF"})
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError as e:
        assert "The 'scftype' parameter must be" in str(e)
    except Exception as e:
        raise AssertionError(
            f"Wrong error caught during QM parameter validation: {str(e)}"
        ) from e
    else:
        raise AssertionError("No error caught during QM parameter validation.")
