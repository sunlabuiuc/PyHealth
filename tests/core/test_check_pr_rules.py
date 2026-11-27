"""Tests for tools/check_pr_rules.py: only the PR's own changes are checked."""

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_pr_rules.py"
_spec = importlib.util.spec_from_file_location("check_pr_rules", _PATH)
rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rules)

BASE_FILE = "".join(f"x{i} = {i}\n" for i in range(1, 11))


@unittest.skipIf(shutil.which("git") is None, "git is not installed")
class TestOnlyPrChangesCount(unittest.TestCase):
    """A branch behind master is not blamed for master's newer lines."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        self.addCleanup(setattr, rules, "REPO_ROOT", rules.REPO_ROOT)
        rules.REPO_ROOT = self.repo

        self.git("init", "-q", "-b", "master")
        self.write("pyhealth/a.py", BASE_FILE)
        self.write("pyhealth/b.py", "y = 1\n")
        self.fork = self.commit("base")

        # the PR changes line 2 of a.py
        self.git("checkout", "-q", "-b", "pr")
        self.write("pyhealth/a.py", BASE_FILE.replace("x2 = 2", "x2 = 20"))
        self.head = self.commit("pr change")

        # master moves on: line 9 of a.py and all of b.py
        self.git("checkout", "-q", "master")
        self.write("pyhealth/a.py", BASE_FILE.replace("x9 = 9", "x9 = 90"))
        self.write("pyhealth/b.py", "y = 2\n")
        self.base = self.commit("master change")
        self.git("checkout", "-q", "pr")

    def git(self, *args):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=self.repo, capture_output=True, text=True, check=True,
        ).stdout.strip()

    def write(self, path, text):
        (self.repo / path).parent.mkdir(parents=True, exist_ok=True)
        (self.repo / path).write_text(text)

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def test_changed_files_are_the_prs_only(self):
        self.assertEqual(rules.changed_files(self.base, self.head),
                         ["pyhealth/a.py"])

    def test_added_lines_are_the_prs_only(self):
        self.assertEqual(rules.added_lines(self.base, self.head, "pyhealth/a.py"),
                         {2})

    def test_up_to_date_branch_is_unchanged(self):
        self.assertEqual(rules.added_lines(self.fork, self.head, "pyhealth/a.py"),
                         {2})

    def test_test_merge_against_its_first_parent(self):
        # CI checks GitHub's merge of the PR into master against master. Make
        # master shift a.py's lines down so the PR's line moves in the merge.
        self.git("checkout", "-q", "master")
        self.write("pyhealth/a.py", "top1 = 1\ntop2 = 2\n"
                   + BASE_FILE.replace("x9 = 9", "x9 = 90"))
        self.commit("master adds two lines on top")
        self.git("merge", "-q", "--no-ff", "--no-edit", "pr")
        merged = (self.repo / "pyhealth/a.py").read_text().splitlines()
        self.assertEqual(merged[3], "x2 = 20")  # the PR's change, now line 4

        self.assertEqual(rules.changed_files("HEAD^1", "HEAD"), ["pyhealth/a.py"])
        self.assertEqual(rules.added_lines("HEAD^1", "HEAD", "pyhealth/a.py"), {4})


if __name__ == "__main__":
    unittest.main()
