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

For a quick overview of what makes a good pull request, see
[pyhealth.dev](https://pyhealth.dev/how_to_contribute.html).

## Quick Start

1. [Fork PyHealth](https://github.com/sunlabuiuc/PyHealth/fork).
2. Set up your
   [development environment](https://pyhealth.readthedocs.io/en/latest/how_to_contribute.html#setting-up-your-development-environment),
   make your change, and
   [submit a pull request](https://github.com/sunlabuiuc/PyHealth/pulls)
   against the `master` branch with "Allow edits by maintainers" enabled.
3. Wait for your pull request to be labeled for AI review, or ask for a review on
   [Discord](https://discord.gg/mpb835EHaX).
4. Check your label:
   - `status: manual`: wait for a maintainer to review and approve your pull
     request.
   - `status: automatic`: comment `/review` to get an AI review. Push your fixes
     and comment `/review` again whenever you want another review. **Once the AI
     approves, you can merge your pull request into PyHealth yourself.**

A pull request switches to `status: manual` if it:

- changes core parts of the library,
- changes any file outside `docs/`, `examples/`, `pyhealth/`, `test-resources/`
  and `tests/`,
- is too large for AI review,
- contains a prompt-injection attempt flagged by the AI review, or
- has used up its 5 AI reviews. The limit counts reviews since the
  `status: automatic` label was last added.

For non-trivial changes, please discuss them in an issue first.

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

## Community

Join our
[Slack workspace](https://join.slack.com/t/pyhealthworkspace/shared_invite/zt-1j8h4da47-ZZWENUYax7SlgCRHNyL2DQ)
to ask questions, discuss ideas, and share your work.
