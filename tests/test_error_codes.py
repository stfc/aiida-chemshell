"""Tests for error code generation on failed job."""

from aiida.orm import Dict

from aiida_chemshell.calculations import ChemShellCalculation


def test_qm_theory_validation(generate_calcjob, generate_inputs):
    """Test the validation of qm theory input."""
    inputs = generate_inputs(qm={"theory": "INVALID"}, structure_fname="butanol.cjson")
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(
            "Incorrect error code generated from QM theory input validation"
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
            "Incorrect error code generated from MM theory input validation"
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
            "Incorrect error caught during no structure file validation:"
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
            "Structure file must be either an '.xyz', '.pun' or \
                        '.cjson' formatted structure file."
            in str(e)
        )
    except Exception as e:
        raise AssertionError(
            "Incorrect error caught during invalid structure file type"
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
            "Incorrect error caught when provided invalid SinglefileData object"
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
            "Incorrect error caught during single point parameter \
                             validation"
        ) from e
    else:
        raise AssertionError(
            "No error caught during single point parameter validation."
        )

    # Test gradient input must be bool
    inputs = generate_inputs(sp={"gradients": "true", "hessian": "false"})
    try:
        generate_calcjob(ChemShellCalculation, inputs)
    except ValueError as e:
        assert "must be a Boolean value" in str(e)
    except Exception as e:
        raise AssertionError(
            "Incorrect error caught during single point parameter \
                             validation"
        ) from e
    else:
        raise AssertionError(
            "No error caught during single point parameter validation."
        )
