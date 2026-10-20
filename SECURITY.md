# Security Policy

## Supported versions

Security fixes go into the latest release on PyPI and the `master` branch.
Older releases are not patched; please upgrade.

## Reporting a vulnerability

**Please do not report security problems in public issues, pull requests or
discussions.**

Report them privately through GitHub:
[**Report a vulnerability**](https://github.com/sunlabuiuc/PyHealth/security/advisories/new)
(Security tab → Advisories → Report a vulnerability).

Please include:

- the affected version or commit
- steps or a minimal script to reproduce
- the impact you expect (for example code execution, data exposure, data corruption)

We aim to acknowledge reports within a week and will keep you updated until a
fix is released. If you accidentally committed a credential to this repository,
revoke it with the issuing service first, then tell us through the same channel.

## Trust model: what PyHealth treats as trusted input

PyHealth is a research library that runs with the permissions of the user
calling it. Some inputs are **loaded as code-capable Python objects** and must
come from a source you trust:

- **Processed dataset and processor directories.** `SampleDataset(path)`,
  `load_processors(path)` and the litdata sample caches written by `set_task()`
  are stored with Python `pickle`. Loading a directory someone else prepared can
  run arbitrary code. Only load caches you created, or that come from a source
  you would trust to run code on your machine.
- **Model checkpoints.** PyHealth loads checkpoints with
  `torch.load(..., weights_only=True)` where it can, but third-party model files
  (for example Hugging Face models) should still come from trusted sources.
- **Downloaded resources.** Some datasets and code mappings are downloaded on
  request (`download=True`, `medcode`). Prefer a trusted network and verify
  large downloads where checksums are published.

Reports that PyHealth executes code from one of these trusted inputs are
expected behaviour, not vulnerabilities, unless PyHealth loads them without the
user asking it to.
