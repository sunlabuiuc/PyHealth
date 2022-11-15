# Contribute

We welcome all kinds of contributions to keep PyHealth growing and thriving.

Everyone can contribute, and we value every contribution, whether it is

- Reporting a bug
- Proposing new features
- Discussing the current state of the code
- Submitting new code
- Becoming a maintainer

## Submit an Issue

In the following cases, you are encouraged
to [submit an issue](https://github.com/sunlabuiuc/PyHealth/issues)
on GitHub.

- Reporting a bug
- Proposing a new feature

For bug reports, please be as detailed as possible. It is recommended to include

- A quick summary
- Steps to reproduce (with sample code if possible)
- What you expected would happen
- What actually happens
- Notes
    - Why you think this might be happening
    - Stuff you have tried that did not work

## Submit a Pull Request

You can contribute code to the project through pull requests. To submit a pull
request, you need to

- [Fork the PyHealth repository]
- Clone it to your local machine
- Implement your code and test it
- Push it to your GitHub repository
- [Create a pull request] in PyHelath repository

[Fork the PyHealth repository]: https://github.com/sunlabuiuc/PyHealth/fork

[Create a pull request]: https://github.com/sunlabuiuc/PyHealth/pulls

If your forked repository is behind the latest version, you can create a pull request in
your repository to pull the latest version from PyHealth.

You can directly make a pull request for trivial changes. For non-trivial changes, it is
better to discuss it in an issue before you implement the code.

## Coding Style

We follow the [PEP8 style](https://peps.python.org/pep-0008/) for code. The line length
is set to 88 characters. We follow
the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
for docstrings.

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