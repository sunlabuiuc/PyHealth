# Contribute

We welcome all kinds of contributions to keep PyHealth growing and thriving.

Everyone can contribute, and we value every contribution, whether it is:

- Reporting a bug;
- Proposing new features;
- Discussing the current state of the code;
- Submitting new code;
- Becoming a maintainer.

## We Use [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)

### Branches

We keep the following branches to allow for parallel development and release
management.

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

### Pull Requests

All code changes happen through pull requests. We actively
welcome your pull requests:

1. Create a new branch from `develop` (for feature) or `master` (for hotfix);
2. Make the changes;
3. Write documentation for the changes;
4. Write unit tests for the changes;
5. Create a pull request to `develop` (for feature) or `master` branch (for hotfix);
6. Wait for the review and merge.

### Code Review

We enforce code reviews for the main branches.

- `master`: requires two approvals from the core developers
- `develop`: requires one approval from the core developers

## Coding Style

We follow the [PEP8](https://peps.python.org/pep-0008/) style guide. The line length
is set to 88 characters. For docstrings, we follow
the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

## Community

We welcome you to join our community
via [Slack](https://join.slack.com/t/pyhealthworkspace/shared_invite/zt-1j8h4da47-ZZWENUYax7SlgCRHNyL2DQ),
where you can ask questions, discuss ideas, and share your work.