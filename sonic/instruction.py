"""Rule-based parsing of natural-language editing instructions.

SONIC supports a single action, removal. The parser extracts the target sound
description, which is used verbatim as the AudioSep / CLAP text query. No
language model is involved; unsupported instructions raise instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class UnsupportedInstructionError(ValueError):
    """The instruction does not describe a supported (removal) edit."""


@dataclass(frozen=True)
class EditInstruction:
    action: str  # currently always "remove"
    target: str  # text query describing the sound to remove
    raw: str


_REMOVE_VERBS = (
    r"get\s+rid\s+of|take\s+out|cut\s+out|remove|delete|eliminate|erase|mute|suppress|cut|filter\s+out"
)
_VERB_RE = re.compile(rf"\b(?:{_REMOVE_VERBS})\b", re.IGNORECASE)

# Anything after these markers describes what to keep, not the target.
_TRAILING_RE = re.compile(
    r"(?:,|;|\.|\?|!|\bwhile\b|\bbut\b|\band\s+(?:keep|preserve|leave)\b|\bwithout\b"
    r"|\bfrom\s+(?:the|this)\b|\bin\s+(?:the|this)\b|\bkeeping\b|\bleaving\b|\bso\s+that\b)",
    re.IGNORECASE,
)
_DETERMINER_RE = re.compile(r"^(?:(?:all|any|every|the|of|a|an|that|those|these)\s+)+", re.IGNORECASE)


def parse_instruction(text: str) -> EditInstruction:
    """Parse e.g. "Remove the dog barking while preserving everything else"."""
    match = _VERB_RE.search(text or "")
    if match is None:
        raise UnsupportedInstructionError(
            f"Unsupported instruction {text!r}: only removal is supported "
            "(e.g. 'Remove the dog barking')."
        )
    rest = text[match.end():]
    trailing = _TRAILING_RE.search(rest)
    if trailing is not None:
        rest = rest[: trailing.start()]
    target = _DETERMINER_RE.sub("", rest.strip() + " ").strip()
    target = re.sub(r"\s+", " ", target)
    if not target:
        raise UnsupportedInstructionError(f"No target sound found in instruction {text!r}.")
    return EditInstruction(action="remove", target=target, raw=text)
