import contextlib
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from packaging.version import Version

from pyhealth.skills import _notice, skill_root
from pyhealth.skills import _installer
from pyhealth.skills._installer import BEGIN, POINTER_TARGETS, main

AGENTS = POINTER_TARGETS[0]
COPILOT = POINTER_TARGETS[1]


class SkillsTestCase(unittest.TestCase):
    """Shared temporary project directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.target = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def install(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(["--target", str(self.target), *args])

    def read(self, rel):
        return (self.target / rel).read_text()


class TestSkillRoot(SkillsTestCase):
    def test_resolves_in_checkout(self):
        root = skill_root()
        self.assertIsNotNone(root)
        self.assertTrue((root / "SKILL.md").is_file())
        self.assertTrue((root / "guides").is_dir())


class TestInstall(SkillsTestCase):
    def test_installs_skill_and_both_pointers(self):
        self.assertEqual(self.install(), 0)
        self.assertTrue((self.target / ".claude" / "skills" / "pyhealth" / "SKILL.md").is_file())
        for rel in POINTER_TARGETS:
            self.assertEqual(self.read(rel).count(BEGIN), 1)

    def test_reinstall_is_idempotent(self):
        self.install()
        before = {rel: self.read(rel) for rel in POINTER_TARGETS}
        self.install()
        for rel in POINTER_TARGETS:
            self.assertEqual(self.read(rel), before[rel])
            self.assertEqual(self.read(rel).count(BEGIN), 1)

    def test_preserves_existing_pointer_content(self):
        (self.target / AGENTS).write_text("# House rules\n\nAlways run the tests.\n")
        self.install()
        text = self.read(AGENTS)
        self.assertIn("Always run the tests.", text)
        self.assertEqual(text.count(BEGIN), 1)

        # A second run replaces the block in place rather than appending another.
        self.install()
        text = self.read(AGENTS)
        self.assertIn("Always run the tests.", text)
        self.assertEqual(text.count(BEGIN), 1)

    def test_in_tree_symlink_is_relative(self):
        """This symlink gets committed — an absolute one dangles in every clone."""
        # Stand up a checkout-shaped project: skills/ beside .claude/, which is
        # the layout `python tools/install_skills.py` produces in the repo. It
        # lives one level inside the tempdir so it can be moved wholesale.
        project = self.target / "checkout"
        project.mkdir()
        src = project / "skills"
        shutil.copytree(skill_root(), src)
        with (
            mock.patch.object(_installer, "skill_root", return_value=src),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(["--target", str(project)])

        link = project / ".claude" / "skills" / "pyhealth"
        raw = os.readlink(link)
        self.assertEqual(raw, os.path.join("..", "..", "skills"))
        self.assertTrue((link / "SKILL.md").is_file())

        # Survives the project moving, which is what a clone amounts to.
        moved = self.target / "elsewhere"
        shutil.move(str(project), str(moved))
        self.assertTrue((moved / ".claude" / "skills" / "pyhealth" / "SKILL.md").is_file())

    def test_copy_points_agents_at_the_copy(self):
        # The source may be site-packages, which is no place to send an agent.
        self.install("--copy")
        self.assertIn(".claude/skills/pyhealth/SKILL.md", self.read(AGENTS))

    def test_symlink_points_agents_at_the_source(self):
        self.install()
        self.assertIn(str(skill_root() / "SKILL.md"), self.read(AGENTS))

    def test_guide_installs_standalone_without_pointers(self):
        self.assertEqual(self.install("--guide", "define-a-task"), 0)
        dest = self.target / ".claude" / "skills" / "pyhealth-define-a-task"
        self.assertTrue((dest / "SKILL.md").is_file())
        # A standalone guide is discovered by its own frontmatter.
        self.assertFalse((self.target / AGENTS).exists())
        self.assertFalse((self.target / COPILOT).exists())

    def test_unknown_guide_errors(self):
        self.assertEqual(self.install("--guide", "no-such-guide"), 1)

    def test_uninstall_removes_only_our_block(self):
        (self.target / AGENTS).write_text("# House rules\n\nAlways run the tests.\n")
        self.install()
        self.assertEqual(self.install("--uninstall"), 0)

        self.assertFalse((self.target / ".claude" / "skills" / "pyhealth").exists())
        self.assertIn("Always run the tests.", self.read(AGENTS))
        self.assertNotIn(BEGIN, self.read(AGENTS))
        # copilot-instructions.md held nothing but the block, so it goes away.
        self.assertFalse((self.target / COPILOT).exists())

    def test_uninstall_does_not_create_a_missing_target(self):
        missing = self.target / "nope"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--target", str(missing), "--uninstall"]), 0)
        self.assertFalse(missing.exists())


class TestVersion(unittest.TestCase):
    def test_version_matches_pyproject(self):
        """__version__ drifted from pyproject for the whole 2.0.x line."""
        import tomllib

        import pyhealth

        pyproject = Path(pyhealth.__file__).resolve().parents[1] / "pyproject.toml"
        if not pyproject.is_file():
            self.skipTest("no source tree — running against an installed wheel")

        declared = tomllib.loads(pyproject.read_text())["project"]["version"]
        # Compare parsed, so 2.1a1 and 2.1.0a1 count as equal.
        self.assertEqual(Version(pyhealth.__version__), Version(declared))


class TestNotice(SkillsTestCase):
    def setUp(self):
        super().setUp()
        # maybe_notify() prints at most once per process.
        _notice._notified = False
        self.addCleanup(setattr, _notice, "_notified", False)

    def notify(self, env, cwd=None):
        self.stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(Path, "cwd", return_value=cwd or self.target),
            contextlib.redirect_stderr(self.stderr),
        ):
            return _notice.maybe_notify()

    def test_silent_without_an_agent_harness(self):
        self.assertFalse(self.notify({}))

    def test_prints_under_an_agent_harness(self):
        self.assertTrue(self.notify({"CLAUDECODE": "1"}))
        message = self.stderr.getvalue()
        self.assertIn("SKILL.md", message)
        self.assertIn("python -m pyhealth.skills", message)
        self.assertIn(_notice.OPT_OUT_ENV_VAR, message)

    def test_prints_only_once(self):
        self.assertTrue(self.notify({"CLAUDECODE": "1"}))
        self.assertFalse(self.notify({"CLAUDECODE": "1"}))

    def test_opt_out(self):
        env = {"CLAUDECODE": "1", _notice.OPT_OUT_ENV_VAR: "1"}
        self.assertFalse(self.notify(env))

    def test_silent_once_registered_via_claude_skills(self):
        self.install()
        self.assertFalse(self.notify({"CLAUDECODE": "1"}))

    def test_silent_once_registered_via_pointer_block_alone(self):
        self.install()
        # Claude Code's directory is one of two registrations; the pointer
        # block alone must also count, for a Codex-only project.
        shutil.rmtree(self.target / ".claude")
        self.assertFalse(self.notify({"CLAUDECODE": "1"}))


if __name__ == "__main__":
    unittest.main()
