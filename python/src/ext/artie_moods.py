"""Artie's moods: a shield pose when Guard blocks a call, a celebration when a migration passes.

ASCII only so legacy Windows consoles render them, and stdlib-only so ``agent-rt migrate``
and ``agent-rt demo`` can print them. Keep in step with typescript/src/ext/artie.ts.
"""

MOODS: dict[str, tuple[str, ...]] = {
    # Narrowed eyes, arm out, shield up.
    "shield": (
        "   .--------.    ______ ",
        "  /.--------.\\  |  ||  |",
        "  || >    < ||==|==||==|",
        "  |'--------'|  \\  ||  /",
        "   '-.____.-'    '.__.' ",
    ),
    # Arms up, confetti.
    "celebrate": (
        " *  .  '  *  .  '  *  . ",
        "  \\   .--------.   /    ",
        "   \\ /.--------.\\ /     ",
        "    \\|| ^    ^ ||/      ",
        "     |'--------'|       ",
        "      '-.____.-'        ",
    ),
}


def pose(mood: str, caption: str) -> str:
    """Artie in ``mood`` with ``caption`` beside the eyes."""
    lines = list(MOODS[mood])
    eyes = next(i for i, line in enumerate(lines) if "^" in line or ">" in line)
    lines[eyes] += "  " + caption
    return "\n".join(line.rstrip() for line in lines)


__all__ = ("MOODS", "pose")
