# run-multimodal-training skill

What it does: takes a one-sentence request like *"run cxr_mlp on a free GPU"* and turns it into a verified, GPU-pinned tmux training session that trains a MIMIC-IV multimodal mortality model and publishes its full config + metrics to a shared wandb project.

## File map (read in this order)

1. `README.md` — you are here. 60-second tour.
2. [`SKILL.md`](./SKILL.md) — the workflow Claude actually follows. Verify-on-every-step.
3. [`experiments.yaml`](./experiments.yaml) — source of truth. Per-combo GPU pinning + hyperparameters.
4. [`flags.md`](./flags.md) — flag dictionary for the underlying runner.
5. [`wandb.md`](./wandb.md) — wandb setup, config schema, prior-run lookup, config pull-back.
6. [`example_run.md`](./example_run.md) — `cxr_mlp` walked end-to-end (YAML → bash → log → wandb → metrics).

## End-to-end loop

```
👤 researcher           🤖 claude+skill              ⚙️ runner                  ☁️ wandb
"run cxr_mlp"     →
                        read experiments.yaml
                        check preconds + dedup   ←──────────────────────  api.runs(...)
                        if duplicate: ask + halt
                        else: resolve full config
                        pick free GPU (nvidia-smi)
                        tmux send-keys           →   wandb.init(config=...)   →  run URL live
                                                     trainer.train(...)            GPU util live
                                                                                   config visible
                                                     replay history           →  per-epoch curves
                                                     log_artifact(csv, json)  →  predictions saved
                                                     wandb.finish()
                        print run URL +
                          assembled command      →
open URL          →                                                              shareable view
```

## Preconditions (one-time per machine)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pyhealth2
pip install wandb
wandb login           # https://wandb.ai/authorize
```

The skill **refuses to launch** if either is missing — by design, no opt-out.

## Five things the skill will NOT do

- Invent hyperparameters not in `experiments.yaml` (asks instead).
- Launch a duplicate of a finished run without confirmation (queries wandb first).
- Bypass `wandb login` — mandatory, hardcoded project (`pyhealth-multimodal`).
- Modify `pyhealth/` source. Runs against whatever `git_sha` is at `HEAD` (and logs it).
- Use `--dev` unless you explicitly say *dev / smoke / quick test*. Defaults to full data.

## Adding a new combo

1. Append an entry to [`experiments.yaml`](./experiments.yaml) with `task`, `model`, `gpu`, and the RAM-sensitive params (`batch_size`, `embedding_dim`, `hidden_dim`, `lr`). There is no per-model defaults layer — every entry is self-contained because memory footprint varies per (task, model) combination.
2. Ask Claude to run it (`run <entry_name>`). The skill takes care of the rest.

## Reproducing someone else's run

```
"rerun the cxr_mlp from <wandb URL>"
```

The skill pulls the prior run's effective config from wandb, surfaces the diff vs. the current YAML, prints `git_sha` then-vs-now, and asks before launching. See [`wandb.md`](./wandb.md) for the pull-back snippet.

## Design principles (the short version)

The skill follows the four [Karpathy principles](https://github.com/multica-ai/andrej-karpathy-skills):

1. **Think before coding** — every workflow step has a `verify:` check. If anything is unclear, the skill halts with a specific question instead of guessing.
2. **Simplicity first** — runner-only wandb integration (no `pyhealth/` patches); YAML entries are self-contained, no inheritance gymnastics.
3. **Surgical changes** — launching a run touches the new tmux session and `logs/<name>.log`. The skill does not refactor the runner, reorganize the cache, or install packages.
4. **Goal-driven execution** — success criteria are explicit at every step: training started? log line shows it. Run logged? wandb URL captured. Metrics computed? PR-AUC printed.
