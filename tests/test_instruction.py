import pytest

from sonic.instruction import EditInstruction, UnsupportedInstructionError, parse_instruction


@pytest.mark.parametrize(
    "text, target",
    [
        ("Remove the dog barking while preserving everything else.", "dog barking"),
        ("Get rid of all the coughing", "coughing"),
        ("mute the siren from the recording", "siren"),
        ("Please delete the door knocking, but keep the rain", "door knocking"),
        ("Eliminate any glass breaking sounds and keep the speech", "glass breaking sounds"),
        ("Take out the clock alarm.", "clock alarm"),
        ("remove dog barking", "dog barking"),
        ("Can you erase the sneezing in this clip?", "sneezing"),
        ("Suppress the crying baby without changing the background", "crying baby"),
    ],
)
def test_parse_remove_variants(text, target):
    instr = parse_instruction(text)
    assert isinstance(instr, EditInstruction)
    assert instr.action == "remove"
    assert instr.target == target
    assert instr.raw == text


@pytest.mark.parametrize("text", ["Amplify the speech", "Make the rain louder", "Remove", "remove the", ""])
def test_unsupported_instructions_raise(text):
    with pytest.raises(UnsupportedInstructionError):
        parse_instruction(text)
