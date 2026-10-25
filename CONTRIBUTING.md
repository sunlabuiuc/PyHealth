# Contribute

We welcome all kinds of contributions to keep PyHealth growing and thriving:
bug reports, feature proposals, new datasets, tasks, and models, bug fixes,
and documentation improvements.

**The full contributor guide lives in our documentation:
[How to Contribute](https://pyhealth.readthedocs.io/en/latest/how_to_contribute.html)**
(source: [`docs/how_to_contribute.rst`](docs/how_to_contribute.rst)).
Please read it before opening a pull request. It covers:

- Examples of great contributions and the review criteria we use
- Setting up your development environment
- Getting data access (MIMIC credentialing and open demo datasets)
- Implementation requirements: file headers, code style, and docstrings
- File checklists for contributing a dataset, task, or model
- Test case requirements, including how to write fast tests with synthetic data
- How to format your pull request description
- How pull requests are reviewed, including AI review with `/review`

## Quick Start

1. [Fork the PyHealth repository](https://github.com/sunlabuiuc/PyHealth/fork)
2. Clone your fork and install it in editable mode:

   ```bash
   git clone https://github.com/your_username/PyHealth.git
   cd PyHealth
   pip install -e .
   ```

3. Implement your change with tests and documentation
4. Push to your fork and
   [open a pull request](https://github.com/sunlabuiuc/PyHealth/pulls)
   against the `master` branch, with "Allow edits by maintainers" enabled

For trivial changes you can open a pull request directly. For non-trivial
changes, please discuss them in an issue first.

## Submit an Issue

[Submit an issue](https://github.com/sunlabuiuc/PyHealth/issues) on GitHub to
report a bug or propose a new feature. For bug reports, please include:

- A quick summary
- Steps to reproduce (with sample code if possible)
- What you expected to happen
- What actually happens
- Why you think this might be happening, and anything you tried that did not work

## PR Contribution Rules (enforced in CI)

Any pull request that modifies a file under `pyhealth/` must:

- Keep newly added/modified lines free of [ruff](https://docs.astral.sh/ruff/)
  lint violations (`ruff check`, 88-char line length). Pre-existing lint
  issues elsewhere in a touched file are not blocked.
- Give every new or modified top-level public class/function a `>>>` usage
  example in its docstring, so it renders as example code in the API docs.

These rules are checked by `.github/workflows/pr_contribution_rules.yml`,
which runs `tools/check_pr_rules.py`. You can run the same check locally:

```bash
python tools/check_pr_rules.py --base <base_sha> --head HEAD
```

## AI Review with `/review`

Pull requests are labeled `status: automatic` or `status: manual`. If yours is
labeled `status: automatic`, you can request an AI review yourself:

1. Comment `/review` on your pull request. A 👀 reaction means the request is
   queued; the review starts once the tests and contribution-rule checks pass on
   your latest commit.
2. If the review approves, your pull request is approved and ready for a
   maintainer to merge. Otherwise it lists what to fix.
3. Push your fixes and comment `/review` again. Each commit is reviewed once.

Each `status: automatic` label allows up to 5 AI reviews, after which the pull
request switches to `status: manual` and a maintainer takes over. Pull requests
that change files outside `docs/`, `examples/`, `pyhealth/`, `tests/` or
`test-resources/`, are too large, or touch core parts of the library also switch
to `status: manual`. On those, only maintainers can request an AI review, and a
maintainer makes the decision.

The workflow is `.github/workflows/ai-review.yml`. See the
[Review Process](https://pyhealth.readthedocs.io/en/latest/how_to_contribute.html#review-process)
section of the guide for details.

## Community

Join our
[Slack workspace](https://join.slack.com/t/pyhealthworkspace/shared_invite/zt-1j8h4da47-ZZWENUYax7SlgCRHNyL2DQ)
to ask questions, discuss ideas, and share your work.
