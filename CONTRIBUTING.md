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

## Porting a Dataset's Official Splits

Many datasets ship with pre-defined splits, for example the train/eval partitions
of the TUH EEG corpora (TUEV, TUAB). To keep `pyhealth/datasets/splitter.py` from
growing one near-identical function per dataset, port official splits as follows.

1. **Tag every sample in the task.** In the task's `__call__`, set a `"split"`
   field on each sample to the dataset's own partition label. `EEGEventsTUEV` and
   `EEGAbnormalTUAB` in `pyhealth/tasks/temple_university_EEG_tasks.py` are the
   reference: they iterate over the `"train"` and `"eval"` partitions and write
   the partition name into `sample["split"]`. Document the field in the task's
   docstring.
2. **Do not add a new splitter to `splitter.py`.** Call
   `pyhealth.datasets.split_by_predefined`, which holds the logic for all
   pre-defined splits:
    - `ratios` has 2 values (train / val) or 3 values (train / val / cal) and is
      applied to the train pool only. Samples whose `split_key` value is not
      `train_value` form the held-out test set.
    - `by_patient=True` keeps all of a patient's samples in the same split
      (the dataset must have `patient_to_index`). Patients are sorted by ID before
