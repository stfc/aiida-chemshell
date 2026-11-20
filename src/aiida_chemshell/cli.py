"""Command-line interface for aiida-chemshell."""

import argparse
import shutil
from collections.abc import Sequence
from importlib.resources import as_file, files
from pathlib import Path

SKILL_NAME = "aiida-chemshell"
DEFAULT_SKILLS_DIR = Path.home() / ".claude" / "skills"


def install_skill(target_dir: Path) -> Path:
    """Copy the aiida-chemshell Agent Skill into a skills directory.

    Parameters
    ----------
    target_dir : Path
        The skills directory to install into, e.g. ``~/.claude/skills``. Created if it
        does not already exist.

    Returns
    -------
    Path
        The path the skill was copied to.
    """
    source = files("aiida_chemshell") / "skills" / SKILL_NAME
    destination = target_dir / SKILL_NAME
    with as_file(source) as source_path:
        shutil.copytree(source_path, destination, dirs_exist_ok=True)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``aiida-chemshell`` command-line tool.

    Parameters
    ----------
    argv : Sequence[str] or None
        Arguments to parse, or None to use ``sys.argv``.
    """
    parser = argparse.ArgumentParser(prog="aiida-chemshell")
    subparsers = parser.add_subparsers(dest="command", required=True)

    skills_parser = subparsers.add_parser(
        "skills", help="Manage the aiida-chemshell Agent Skill."
    )
    skills_subparsers = skills_parser.add_subparsers(
        dest="skills_command", required=True
    )

    install_parser = skills_subparsers.add_parser(
        "install", help="Copy the aiida-chemshell Agent Skill into a skills directory."
    )
    install_parser.add_argument(
        "dir_name",
        nargs="?",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help=f"Skills directory to install into (default: {DEFAULT_SKILLS_DIR}).",
    )

    args = parser.parse_args(argv)

    destination = install_skill(args.dir_name.expanduser())
    print(f"Installed the aiida-chemshell skill to {destination}")
