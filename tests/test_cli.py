"""Tests for the aiida-chemshell command-line interface."""

from aiida_chemshell import cli


def test_install_skill_copies_skill_md(tmp_path):
    """install_skill copies SKILL.md into a fresh target directory."""
    destination = cli.install_skill(tmp_path)

    assert destination == tmp_path / "aiida-chemshell"
    assert (destination / "SKILL.md").is_file()


def test_install_skill_can_run_twice(tmp_path):
    """install_skill can be re-run against an existing install without failing."""
    cli.install_skill(tmp_path)
    destination = cli.install_skill(tmp_path)

    assert (destination / "SKILL.md").is_file()


def test_main_install_uses_default_skills_dir(tmp_path, monkeypatch):
    """`aiida-chemshell skills install` with no argument targets the default dir."""
    default_dir = tmp_path / ".claude" / "skills"
    monkeypatch.setattr(cli, "DEFAULT_SKILLS_DIR", default_dir)

    cli.main(["skills", "install"])

    assert (default_dir / "aiida-chemshell" / "SKILL.md").is_file()


def test_main_install_with_explicit_dir(tmp_path, capsys):
    """`aiida-chemshell skills install <dir>` installs into the given directory."""
    target = tmp_path / "custom"

    cli.main(["skills", "install", str(target)])

    assert (target / "aiida-chemshell" / "SKILL.md").is_file()
    assert "Installed" in capsys.readouterr().out
