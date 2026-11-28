"""Tests for tools/update_bounty_board.py: completion, credit and rendering."""

import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "tools" / "update_bounty_board.py"
_spec = importlib.util.spec_from_file_location("update_bounty_board", _PATH)
board = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(board)

YAML = textwrap.dedent("""\
    bounty:
      - id: 1
        priority: 1
        category: model
        title: Old model
        description: "Done long ago."
        complete: "@alice (alice@illinois.edu), @bob"
      - id: 2
        priority: 2
        category: model
        title: Merged model
        description: "Merged, credit not recorded yet."
        complete: ~
      - id: 3
        priority: 3
        category: dataset
        title: Claimed dataset
        description: "An open PR."
        complete: ~
      - id: 4
        priority: 4
        category: task
        title: Free task
        description: "Nobody yet."
        complete: ~
    """)

PRS = [
    {"number": 10, "title": "[BTY-2] Merged model", "state": "MERGED", "author": "carol"},
    {"number": 12, "title": "[BTY-2] follow-up", "state": "MERGED", "author": "carol"},
    {"number": 11, "title": "[Bounty-2] other half", "state": "MERGED", "author": "dan"},
    {"number": 13, "title": "[BTY-3] WIP dataset", "state": "OPEN", "author": "erin"},
    {"number": 14, "title": "[BTY-99] unknown", "state": "OPEN", "author": "frank"},
]


class TestBountyBoard(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(YAML)
        self.addCleanup(Path(f.name).unlink)
        self.bounties = board.load_bounties(f.name)
        self.claims, self.unmatched = board.link_prs(self.bounties, PRS)

    def test_every_bounty_is_loaded_without_emails(self):
        self.assertEqual(sorted(self.bounties), [1, 2, 3, 4])
        self.assertEqual(self.bounties[1]["complete"], "@alice, @bob")
        self.assertIsNone(self.bounties[2]["complete"])

    def test_status_and_credit(self):
        def status(bid):
            return board.status(self.bounties[bid], self.claims.get(bid, []))

        self.assertEqual(status(1), ("Completed", "@alice, @bob"))
        # a merged PR completes the bounty; credit groups PRs by author
        self.assertEqual(status(2), ("Completed", "@carol (#10, #12), @dan (#11)"))
        self.assertEqual(status(3), ("Claimed", None))
        self.assertEqual(status(4), ("Unclaimed", None))
        self.assertEqual([(bid, pr["number"]) for bid, pr in self.unmatched],
                         [(99, 14)])

    def test_uncredited_lists_merged_bounties_missing_from_yaml(self):
        self.assertEqual(board.uncredited(self.bounties, self.claims),
                         {2: "@carol (#10, #12), @dan (#11)"})

    def test_board_has_a_completed_section_with_credit(self):
        sections = dict(board.build_sections(self.bounties, self.claims))
        completed = dict(sections["Completed"])
        self.assertEqual(completed["Model"], [
            "- **BTY-2** · Merged model · @carol (#10, #12), @dan (#11)",
            "- **BTY-1** · Old model · @alice, @bob",
        ])
        self.assertEqual([c for c, _ in sections["Claimed"]], ["Dataset"])
        self.assertEqual([c for c, _ in sections["Unclaimed"]], ["Task"])
        header = board.render_header("org/repo", 1, 1, 2)
        page = board.paginate(header, board.build_sections(self.bounties,
                                                           self.claims))[0]
        self.assertIn("Completed: 2. Thank you", page)
        self.assertIn("## Completed", page)
        self.assertNotIn("illinois.edu", page)

    def test_invalid_complete_is_rejected(self):
        bad = YAML.replace("complete: ~\n  - id: 3", "complete: 5\n  - id: 3")
        self.assertNotEqual(bad, YAML)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(bad)
        self.addCleanup(Path(f.name).unlink)
        with self.assertRaises(SystemExit) as ctx:
            board.load_bounties(f.name)
        self.assertIn("`complete` must be a non-empty string", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
