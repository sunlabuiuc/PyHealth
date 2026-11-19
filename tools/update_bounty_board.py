"""
Maintain the bounty board issue from bounty.yaml.

Each run rebuilds the board from scratch:

    1. Load bounty.yaml and drop entries whose `complete` field is set.
    2. Link every open or merged PR whose title carries `[BTY-<id>]` or
       `[Bounty-<id>]` (case-insensitive) to that bounty. PRs closed without
       merging are ignored.
    3. Render the board: Unclaimed, then Claimed; within each, one subsection
       per category (alphabetical, only if non-empty); within each, id order.
    4. Find the board issue (created by the bot, hidden marker in the body),
       creating it if missing; keep it open, locked and pinned.
    5. Write the board into the issue body and, when it is longer than one
       body allows, into follow-up bot comments. A bounty entry is never split
       across two pages. Only pages whose text changed are edited.

Usage:
    python tools/update_bounty_board.py                 # needs GITHUB_TOKEN, GITHUB_REPOSITORY
    python tools/update_bounty_board.py --dry-run       # print pages, write nothing
    python tools/update_bounty_board.py --dry-run --prs-file prs.json
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

ISSUE_TITLE = "Bounty board"
BODY_MARKER = "<!-- bounty-board:body -->"
PAGE_MARKER = "<!-- bounty-board:page:{n} -->"
PAGE_MARKER_RE = re.compile(r"<!-- bounty-board:page:\d+ -->")
CONTINUED_FOOTER = "\n\n---\n_Continued in the comments below._"

# GitHub caps an issue body or comment at 65,536 characters. Pack pages to a
# lower budget so markers, footers and headings always fit.
GITHUB_MAX_CHARS = 65536
PAGE_BUDGET = 60000

TITLE_TAG_RE = re.compile(r"\[\s*(?:bounty|bty)\s*-\s*(\d+)\s*\]", re.IGNORECASE)

SECTIONS = ("Unclaimed", "Claimed")

# Subsection order; a category outside this list is rejected.
CATEGORIES = (
    "infrastructure", "dataset", "task", "model", "interpretability", "calibration",
)

# Priority 1 is the highest. U+FE0E keeps the double arrows in text style
# instead of emoji style.
PRIORITY_ICONS = {
    1: "\u23eb\ufe0e", 2: "\u25b2", 3: "\u2501", 4: "\u25bc", 5: "\u23ec\ufe0e",
}
PRIORITY_NAMES = {1: "highest", 2: "high", 3: "medium", 4: "low", 5: "lowest"}


# --------------------------------------------------------------------------- #
# bounty.yaml
# --------------------------------------------------------------------------- #
def load_bounties(path):
    """Return the open (not complete) bounties from `path`, keyed by id."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("bounty") or []
    errors = []
    bounties = {}
    seen = set()
    for i, entry in enumerate(entries):
        where = f"bounty entry #{i + 1}"
        if not isinstance(entry, dict):
            errors.append(f"{where}: expected a mapping")
            continue
        bid = entry.get("id")
        if not isinstance(bid, int) or isinstance(bid, bool) or bid < 0:
            errors.append(f"{where}: `id` must be a non-negative integer")
            continue
        where = f"bounty id {bid}"
        if bid in seen:
            errors.append(f"{where}: duplicate id")
            continue
        seen.add(bid)
        for field in ("title", "category", "description"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{where}: `{field}` must be a non-empty string")
        category = str(entry.get("category", "")).strip()
        if category and category not in CATEGORIES:
            errors.append(
                f"{where}: unknown category {category!r} "
                f"(expected one of: {', '.join(CATEGORIES)})"
            )
        priority = entry.get("priority")
        if priority not in PRIORITY_ICONS or isinstance(priority, bool):
            errors.append(
                f"{where}: `priority` must be an integer from 1 (highest) to 5"
            )
        if entry.get("complete") is not None:
            continue
        bounties[bid] = {
            "id": bid,
            "priority": priority,
            "title": str(entry.get("title", "")).strip(),
            "category": category,
            "description": str(entry.get("description", "")).strip(),
        }
    if errors:
        raise SystemExit("bounty.yaml is invalid:\n  " + "\n  ".join(errors))
    return bounties


def category_title(category):
    return category.replace("_", " ").strip().title()


# --------------------------------------------------------------------------- #
# GitHub API
# --------------------------------------------------------------------------- #
class GitHub:
    def __init__(self, token, repo, api_url, graphql_url):
        self.token = token
        self.repo = repo
        self.api_url = api_url.rstrip("/")
        self.graphql_url = graphql_url

    def _request(self, method, url, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "pyhealth-bounty-board")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as resp:
                body = resp.read()
                link = resp.headers.get("Link", "")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise GitHubError(e.code, f"{method} {url} -> {e.code}: {detail}") from None
        return (json.loads(body) if body else None), link

    def rest(self, method, path, payload=None):
        return self._request(method, f"{self.api_url}{path}", payload)[0]

    def rest_paginated(self, path):
        url = f"{self.api_url}{path}"
        items = []
        while url:
            page, link = self._request("GET", url)
            items.extend(page)
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match.group(1) if match else None
        return items

    def graphql(self, query, **variables):
        result, _ = self._request(
            "POST", self.graphql_url, {"query": query, "variables": variables}
        )
        if result.get("errors"):
            raise GitHubError(200, f"GraphQL error: {result['errors']}")
        return result["data"]

    @property
    def owner_name(self):
        return self.repo.split("/", 1)


class GitHubError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


PRS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: [OPEN, MERGED], first: 100, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes { number title state author { login } }
    }
  }
}
"""


def fetch_prs(gh):
    """All open and merged PRs as [{number, title, state, author}]."""
    owner, name = gh.owner_name
    prs, cursor = [], None
    while True:
        data = gh.graphql(PRS_QUERY, owner=owner, name=name, cursor=cursor)
        conn = data["repository"]["pullRequests"]
        for node in conn["nodes"]:
            prs.append({
                "number": node["number"],
                "title": node["title"],
                "state": node["state"],
                "author": (node["author"] or {}).get("login") or "ghost",
            })
        if not conn["pageInfo"]["hasNextPage"]:
            return prs
        cursor = conn["pageInfo"]["endCursor"]


# --------------------------------------------------------------------------- #
# Linking and rendering
# --------------------------------------------------------------------------- #
def link_prs(bounties, prs):
    """Map bounty id -> linked PRs, plus PRs that name an unknown/complete id."""
    claims = defaultdict(list)
    unmatched = []
    for pr in prs:
        if pr["state"] not in ("OPEN", "MERGED"):
            continue
        ids = {int(m) for m in TITLE_TAG_RE.findall(pr["title"])}
        for bid in sorted(ids):
            if bid in bounties:
                claims[bid].append(pr)
            else:
                unmatched.append((bid, pr))
    for bid in claims:
        claims[bid].sort(key=lambda p: p["number"])
    return claims, unmatched


def render_entry(bounty, claim_prs):
    icon = PRIORITY_ICONS[bounty["priority"]]
    heading = f"#### {icon} · BTY-{bounty['id']} · {bounty['title']}"
    if claim_prs:
        by_author = defaultdict(list)  # insertion order = first PR number
        for p in claim_prs:
            by_author[p["author"]].append(f"#{p['number']}")
        who = ", ".join(
            f"@{author} through {' and '.join(prs)}"
            for author, prs in by_author.items()
        )
        heading += f" · Claimed by {who}"
    else:
        heading += " · To be claimed"
    return f"{heading}\n\n{bounty['description']}"


def build_sections(bounties, claims):
    """[(section, [(category title, [entry text, ...]), ...]), ...]"""
    grouped = {s: defaultdict(list) for s in SECTIONS}
    for bounty in sorted(bounties.values(), key=lambda b: (b["priority"], b["id"])):
        bid = bounty["id"]
        section = "Claimed" if claims.get(bid) else "Unclaimed"
        grouped[section][bounty["category"]].append(
            render_entry(bounty, claims.get(bid, []))
        )
    return [
        (
            section,
            [
                (category_title(cat), grouped[section][cat])
                for cat in CATEGORIES
                if grouped[section][cat]
            ],
        )
        for section in SECTIONS
    ]


DISCORD_URL = "https://discord.gg/mpb835EHaX"
GUIDE_URL = "https://pyhealth.readthedocs.io/en/latest/how_to_contribute.html"

HEADER_TEMPLATE = """\
Welcome! These are the open PyHealth bounties: models, datasets, tasks and \
tools we'd love help adding. Anyone can take one on.

**Before you start**
1. Join the [PyHealth Discord]({discord}). Ask questions and request reviews there.
2. Read the [contribution guide]({guide}).

**Pick a bounty**
- Choose the highest-priority bounty you can take on. Priority: {legend}.
- Pick one marked *To be claimed*. To help with a claimed one, ask in Discord.
- Check that it isn't already in PyHealth or in an \
[open pull request](https://github.com/{repo}/pulls).
- Know a paper with public code that would fit PyHealth? Suggest it in Discord.

**Claim and build**
- Claim a bounty by opening a pull request (a draft is fine) with its ID in the \
title, like `[BTY-<id>] <short description>`. `[Bounty-<id>]` works too. This \
board usually updates within a few minutes.
- Your claim holds while the PR is open and after it merges. Closing it without \
merging releases the bounty.
- Build from the paper's official code, use datasets PyHealth already loads \
where possible, and aim to reproduce the paper's results.
- Every bounty needs an implementation that follows the \
[contribution guide]({guide}).
- When it's ready, comment `/review` on your PR for an AI review, or ask in Discord.

**Why contribute**
Students in the DL4H course can earn extra credit. Contributors who complete a \
bounty may get the chance to join a research paper and work with our industry \
partners.

*This board is generated from [`bounty.yaml`]({yaml_url}) and updated when a tagged PR changes. \
Edits made here are overwritten.*

Open bounties: {total} ({claimed} claimed, {unclaimed} unclaimed)."""


def render_header(repo, n_claimed, n_unclaimed):
    legend = ", ".join(
        f"{PRIORITY_ICONS[p]} {PRIORITY_NAMES[p]}" for p in PRIORITY_ICONS
    )
    return HEADER_TEMPLATE.format(
        discord=DISCORD_URL,
        guide=GUIDE_URL,
        legend=legend,
        repo=repo,
        yaml_url=f"https://github.com/{repo}/blob/HEAD/bounty.yaml",
        total=n_claimed + n_unclaimed,
        claimed=n_claimed,
        unclaimed=n_unclaimed,
    )


def paginate(header, sections, budget=PAGE_BUDGET):
    """Pack the board into pages of at most `budget` characters.

    An entry is the unit of packing and never splits. When a section or
    category carries over to a new page, its heading is repeated with
    "(continued)".
    """
    # Flatten to (section, category, text); an empty section gets a placeholder.
    blocks = []
    for section, categories in sections:
        if not categories:
            blocks.append((section, None, "_None._"))
        for category, entries in categories:
            blocks.extend((section, category, entry) for entry in entries)

    pages = []
    cur = [header]
    on_page = (None, None)  # (section, category) whose headings are on `cur`
    started = set()  # headings emitted on any page so far

    def headings(section, category):
        out = []
        if on_page[0] != section:
            cont = " (continued)" if section in started else ""
            out.append(f"## {section}{cont}")
        if category is not None and on_page != (section, category):
            cont = " (continued)" if (section, category) in started else ""
            out.append(f"### {category}{cont}")
        return out

    def fits(parts):
        return len("\n\n".join(cur + parts)) <= budget

    for section, category, text in blocks:
        parts = headings(section, category) + [text]
        if not fits(parts) and cur:
            pages.append("\n\n".join(cur))
            cur, on_page = [], (None, None)
            parts = headings(section, category) + [text]
        if not fits(parts):
            raise SystemExit(
                f"A single bounty entry is longer than {budget} characters; "
                "shorten its description in bounty.yaml."
            )
        cur.extend(parts)
        started.update({section, (section, category)})
        on_page = (section, category)
    pages.append("\n\n".join(cur))
    return pages


def finalize_pages(pages):
    """Add hidden markers and footer; returns (issue body, [comment bodies])."""
    body = f"{BODY_MARKER}\n{pages[0]}"
    if len(pages) > 1:
        body += CONTINUED_FOOTER
    comments = [
        f"{PAGE_MARKER.format(n=i)}\n{text}" for i, text in enumerate(pages[1:], 2)
    ]
    for text in [body, *comments]:
        assert len(text) <= GITHUB_MAX_CHARS, len(text)
    return body, comments


# --------------------------------------------------------------------------- #
# Issue sync
# --------------------------------------------------------------------------- #
def find_board_issue(gh, bot_login):
    issues = gh.rest_paginated(
        f"/repos/{gh.repo}/issues?creator={urllib.parse.quote(bot_login)}"
        "&state=all&per_page=100"
    )
    candidates = [
        i for i in issues
        if "pull_request" not in i and BODY_MARKER in (i.get("body") or "")
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"::warning::Found {len(candidates)} board issues; using the oldest.")
    return min(candidates, key=lambda i: i["number"])


def ensure_pinned(gh, number):
    owner, name = gh.owner_name
    data = gh.graphql(
        "query($owner: String!, $name: String!, $number: Int!) {"
        " repository(owner: $owner, name: $name) {"
        " issue(number: $number) { id isPinned } } }",
        owner=owner, name=name, number=number,
    )
    issue = data["repository"]["issue"]
    if issue["isPinned"]:
        return
    try:
        gh.graphql(
            "mutation($id: ID!) { pinIssue(input: {issueId: $id}) { issue { number } } }",
            id=issue["id"],
        )
        print(f"Pinned issue #{number}.")
    except GitHubError as e:
        # A repo can pin at most three issues; don't fail the board over it.
        print(f"::warning::Could not pin issue #{number}: {e}")


def sync_issue(gh, bot_login, body, comments):
    issue = find_board_issue(gh, bot_login)
    if issue is None:
        issue = gh.rest("POST", f"/repos/{gh.repo}/issues", {
            "title": ISSUE_TITLE, "body": body,
        })
        print(f"Created board issue #{issue['number']}.")
    number = issue["number"]
    path = f"/repos/{gh.repo}/issues/{number}"

    if issue["state"] != "open":
        gh.rest("PATCH", path, {"state": "open"})
        print(f"Reopened issue #{number}.")
    if (issue.get("body") or "") != body:
        gh.rest("PATCH", path, {"body": body})
        print(f"Updated body of issue #{number}.")
    if not issue.get("locked"):
        gh.rest("PUT", f"{path}/lock", {})
        print(f"Locked issue #{number}.")
    ensure_pinned(gh, number)

    # Bot page comments, in creation order, map onto pages 2..N.
    existing = [
        c for c in gh.rest_paginated(f"{path}/comments?per_page=100")
        if (c.get("user") or {}).get("login") == bot_login
        and PAGE_MARKER_RE.search(c.get("body") or "")
    ]
    existing.sort(key=lambda c: (c["created_at"], c["id"]))
    for i, text in enumerate(comments):
        if i < len(existing):
            if existing[i]["body"] != text:
                gh.rest("PATCH", f"/repos/{gh.repo}/issues/comments/{existing[i]['id']}",
                        {"body": text})
                print(f"Updated page {i + 2} (comment {existing[i]['id']}).")
        else:
            gh.rest("POST", f"{path}/comments", {"body": text})
            print(f"Posted page {i + 2}.")
    for stale in existing[len(comments):]:
        gh.rest("DELETE", f"/repos/{gh.repo}/issues/comments/{stale['id']}")
        print(f"Deleted stale page comment {stale['id']}.")
    return issue


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def write_summary(lines):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bounty-file", default=str(REPO_ROOT / "bounty.yaml"))
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--bot-login", default="github-actions[bot]",
                        help="Account that owns the board issue and its comments.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the rendered pages instead of writing them.")
    parser.add_argument("--prs-file",
                        help="Read PRs from a JSON file instead of the API "
                             "(list of {number, title, state, author}).")
    parser.add_argument("--page-budget", type=int, default=PAGE_BUDGET,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.repo:
        parser.error("--repo or GITHUB_REPOSITORY is required")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    gh = None
    if token:
        gh = GitHub(
            token, args.repo,
            os.environ.get("GITHUB_API_URL", "https://api.github.com"),
            os.environ.get("GITHUB_GRAPHQL_URL", "https://api.github.com/graphql"),
        )

    bounties = load_bounties(args.bounty_file)
    if args.prs_file:
        prs = json.loads(Path(args.prs_file).read_text(encoding="utf-8"))
    elif gh:
        prs = fetch_prs(gh)
    else:
        parser.error("GITHUB_TOKEN is required unless --prs-file is given")

    claims, unmatched = link_prs(bounties, prs)
    sections = build_sections(bounties, claims)
    n_claimed = sum(1 for bid in bounties if claims.get(bid))
    header = render_header(args.repo, n_claimed, len(bounties) - n_claimed)
    body, comments = finalize_pages(paginate(header, sections, args.page_budget))

    print(f"{len(bounties)} open bounties ({n_claimed} claimed), "
          f"{len(prs)} PRs scanned, {1 + len(comments)} page(s).")
    for bid, pr in unmatched:
        print(f"::notice::PR #{pr['number']} names BTY-{bid}, "
              "which is not an open bounty.")

    if args.dry_run:
        for i, text in enumerate([body, *comments], 1):
            print(f"\n===== page {i} ({len(text)} chars) =====\n{text}")
        return
    if gh is None:
        parser.error("GITHUB_TOKEN is required to update the issue")

    issue = sync_issue(gh, args.bot_login, body, comments)
    summary = [
        "## Bounty board",
        f"Issue: {issue['html_url']}",
        f"- Open bounties: {len(bounties)} ({n_claimed} claimed)",
        f"- Pages: {1 + len(comments)}",
    ]
    if unmatched:
        summary.append("- PRs naming a bounty that is not open: " + ", ".join(
            f"#{pr['number']} (BTY-{bid})" for bid, pr in unmatched))
    write_summary(summary)


if __name__ == "__main__":
    sys.exit(main())
