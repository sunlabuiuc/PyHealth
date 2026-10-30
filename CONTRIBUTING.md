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
      shuffling, so the result does not depend on sample order.
    - `split_key` (default `"split"`) and `train_value` (default `"train"`) adapt
      the function to a different field name or partition label.
    - `get_index=True` returns `torch.Tensor` index vectors instead of datasets.
3. **Wrap it only if the arguments are fixed.** If a dataset always needs the same
   arguments, add a thin wrapper next to that dataset (in its module under
   `pyhealth/datasets/`), not in `splitter.py`. The existing `*_tuh` functions in
   `splitter.py` are such wrappers, kept for backward compatibility.
4. **Test it.** Add a unit test that checks the held-out set is exactly the
   official eval partition, that no patient appears in more than one part when
   `by_patient=True`, and that the same seed gives the same split.

```python
>>> from pyhealth.datasets import split_by_predefined
>>> train, val, test = split_by_predefined(  # doctest: +SKIP
...     sample_dataset, [0.8, 0.2], seed=42, by_patient=True
... )
>>> train, val, cal, test = split_by_predefined(  # doctest: +SKIP
...     sample_dataset, [0.6, 0.2, 0.2], seed=42, by_patient=True
... )
```

These functions split an already-processed `SampleDataset`, so its processors
(vocabularies, learned statistics) were fitted on all samples. If preprocessing
must be fitted on the training part only, split inside `set_task` instead, with
`PatientSplit` or a custom `Split`.

## Community

We welcome you to join our community
via [Slack](https://join.slack.com/t/pyhealthworkspace/shared_invite/zt-1j8h4da47-ZZWENUYax7SlgCRHNyL2DQ)
workspace,
where you can ask questions, discuss ideas, and share your work.

## Core Development

We use [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
for development.

### Branches

We keep the following branches for parallel development and release management.

- Main branches
    - `master`: always reflects the production-ready state.
    - `develop`: reflects the latest delivered development state. When the source code
      in the `develop` branch reaches a stable point and is ready to be released, all
      of the changes will be merged back into `master`.
- Release branches
    - `*-release`: holds the code for a specific release. These branches are
      created from `develop`, and will be merged back into `master` and `develop`
      when the release is deployed. Only minor bug fixes or preparing meta-data for
      a release (version number, build dates, etc.) are allowed in this branch.
    - `archived`: holds the original version of PyHealth developed
      by [Yue Zhao](https://www.andrew.cmu.edu/user/yuezhao2/)
      et al. This branch is not maintained anymore.
- Feature branches:
    - `[feature_name]`: feature branches are used to develop new features for the
      upcoming release. These branches are created from `develop` and will be merged
      back into `develop` once the feature is completed.
- Hotfix branches:
    - `[hotfix_name]`: hotfix branches are used to quickly patch small bugs in the
      production version. These branches are created from `master` and will be merged
      back into `master` and `develop`.

See this blog post for more details
on [Git Branching Model](https://nvie.com/posts/a-successful-git-branching-model/).

### Pull Requests

All code changes happen through pull requests.

1. Create a new branch from `develop` (for feature) or `master` (for hotfix)
2. Make the changes
3. Write documentation for the changes
4. Write unit tests for the changes
5. Create a pull request to `develop` (for feature) or `master` branch (for hotfix)
6. Wait for the review and merge

### Code Review

We enforce code reviews for the main branches.

- `master`: requires two approvals from the core developers
- `develop`: requires one approval from the core developers
Join our
[Slack workspace](https://join.slack.com/t/pyhealthworkspace/shared_invite/zt-1j8h4da47-ZZWENUYax7SlgCRHNyL2DQ)
to ask questions, discuss ideas, and share your work.
