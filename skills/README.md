# PyHealth agent skill

Instructions for AI coding agents — Claude Code, Codex, Copilot — working with PyHealth. Plain
markdown, no runtime dependency: an agent reads them, and so can you.

**Start here:** [SKILL.md](SKILL.md) is the router. [table-of-contents.md](table-of-contents.md)
is the full manifest of guides.

## How it works

PyHealth is large — 24 datasets, 46 tasks, 48 models, 24 registered processors, 11 calibrators
and set predictors, 12 interpreters, 10 code vocabularies. A single monolithic instruction file
either covers a fraction of that or drowns the agent in context it does not need.

So `SKILL.md` is a **router**, not a manual. It decomposes the request into the PyHealth
subsystems actually involved, then loads only the guides for those. One request pulls two
guides; another pulls four. The set is flat and composable rather than a taxonomy tree, because
"add a dataset and a task for it, then contribute upstream" cuts across any tree you would draw.

```
skills/
├── SKILL.md              ← the router: decompose, then dispatch
├── table-of-contents.md  ← full manifest
├── shared/               ← glossary, data model, repo conventions
└── guides/
    ├── set-up-the-environment/       scope-a-modeling-request/
    ├── bring-your-own-data/          use-a-dataset/
    ├── define-a-task/                choose-processors/
    ├── choose-a-model/               train-and-evaluate/
    ├── optimize-a-pipeline/          calibrate-predictions/
    ├── interpret-a-model/            map-medical-codes/
    └── add-a-component/              ship-a-contribution/
```

Only the router's `description` sits permanently in an agent's context. Guide directories are
read on demand — nested content is never auto-loaded — so fourteen guides cost no more ambient
context than one.

## Install

From a git checkout — installs into the repo root by default:

```sh
python tools/install_skills.py                        # into this repo
python tools/install_skills.py --target ../my-project # into another project
python tools/install_skills.py --guide define-a-task  # one guide, standalone
python tools/install_skills.py --copy                 # copy instead of symlink
python tools/install_skills.py --uninstall
```

After `pip install pyhealth` — same flags, installs into the current directory by default:

```sh
python -m pyhealth.skills
python -m pyhealth.skills --guide define-a-task
python -m pyhealth.skills --uninstall
```

The skill ships inside the wheel, so the pip form needs no checkout. It always *copies* rather
than symlinking, since a link into `site-packages` dies with the virtualenv.

Either form links or copies the skill into `.claude/skills/pyhealth` for Claude Code and appends
a marked pointer block to `AGENTS.md` (Codex and most harnesses) and
`.github/copilot-instructions.md` (Copilot). Re-running replaces the block rather than
duplicating it.

Because `pip install` of a prebuilt wheel runs no code of ours, PyHealth cannot announce the
skill at install time. Instead, the first `import pyhealth` under a coding-agent harness prints
a one-line pointer to stderr. It is silent for human users, silent once the skill is registered
in the project, and silenced entirely by `export PYHEALTH_NO_SKILL_NOTICE=1`. Building from
source (`pip install -e .`) prints the same pointer via a build hook, though pip only surfaces
build output under `-v`.

Those two formats carry flat prose only — they have no concept of skill discovery. That is why
the routing table lives *inside* `SKILL.md` rather than relying on the host to match
descriptions: it is the only routing mechanism non-Claude harnesses have.

Without installing, any agent can read [SKILL.md](SKILL.md) directly.

## Design notes

**Agree before building.** The expensive failure in clinical ML is not a bug — it is a
well-trained model answering the wrong question, or one whose score comes from leakage. Both are
decided before the first line of code. So modeling work gates on a confirmed
`pipeline_spec.md`, and contribution work gates on an agreed file list.

**One step at a time.** Guides deliberately do not build end to end in a single turn. Each
module boundary — dataset, task, processors, split, model, training — is a checkpoint where
output is shown and a wrong assumption is still cheap to fix.

**Written for a newcomer.** The assumed reader may be new to clinical predictive modeling and
may be starting from a folder of CSVs rather than a supported dataset. Terms get defined,
questions carry recommended defaults, and safety rules come with the reason they exist.

**Verified against source.** Every class name, kwarg, file path, and processor name was checked
against `pyhealth/`. `tools/check_skills.py` re-checks the mechanical parts in CI.

## Authoring a guide

1. `mkdir guides/<kebab-case-name>/`
2. Write `SKILL.md` with frontmatter carrying exactly `name` and `description`. The name must be
   `pyhealth-<directory-name>`. The description is what a host matches on — state what it does
   *and* when to use it.
3. Add a row to the routing table in [SKILL.md](SKILL.md) — including a "don't use it when"
   column entry, which is what keeps the router from loading everything.
4. Add a full entry to [table-of-contents.md](table-of-contents.md).
5. Run `python tools/check_skills.py`.

Keep guides focused. If one grows past ~300 lines, split the detail into its own
`references/` subdirectory (see `train-and-evaluate/` and `optimize-a-pipeline/`) rather than
letting the entry point sprawl.

## Provenance

Ported from the human-authored skills in the PyHealthAgent research repo, which targeted an
autonomous benchmark agent. The benchmark harness — experiment submission, budget allocation,
job directories — has been stripped and the interaction inverted: that agent optimized a fixed
task without asking anyone; this one asks first and builds with you.
