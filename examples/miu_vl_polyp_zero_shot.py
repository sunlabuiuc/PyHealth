"""Zero-shot polyp detection with GLIP-T and MIU-VL prompts.

MIU-VL (Qin et al., ICLR 2023, https://arxiv.org/abs/2209.15517) detects
medical targets with GLIP-T (Li et al., CVPR 2022) by describing them in the
text prompt. Table 3 of the paper reports zero-shot AP on the five polyp
test sets for the class name (GLIP-T "default", 4.1 AP on average), MLM
prompts (38.4), VQA prompts (27.1), hybrid prompts (35.1) and manual prompts
(41.3). This script scores, with :class:`~pyhealth.tasks.PolypDetection` and
:func:`~pyhealth.metrics.detection_metrics_fn` (COCO AP):

- ``default``: the class name, ``"polyp"``;
- ``manual``: :data:`~pyhealth.models.miu_vl.POLYP_MANUAL_PROMPT`, the polyp
  prompt of MIU-VL's configuration;
- ``mlm1`` .. ``mlm3``: :func:`~pyhealth.models.miu_vl.mlm_prompts` from
  PubMedBERT, k-th ranked words;
- ``hybrid1`` .. ``hybrid3`` (with ``--hybrid-dir``): MIU-VL's released
  image-specific hybrid prompts (VQA colour and shape, MLM location), read
  from the ``autoprompt_json`` folder of https://github.com/MembrAI/MIU-VL.
  They exist for CVC-300, CVC-ClinicDB, CVC-ColonDB and Kvasir, not ETIS.

Usage:
    python examples/miu_vl_polyp_zero_shot.py --root /data/DATA/POLYP \\
        --checkpoint /path/to/glip_tiny_model_o365_goldg.pth
    python examples/miu_vl_polyp_zero_shot.py --root /data/DATA/POLYP \\
        --prompts default manual mlm1 --hybrid-dir MIU-VL/autoprompt_json

Without ``--checkpoint``, the official GLIP-T checkpoint is downloaded from
Hugging Face (3.7 GB).
"""

import argparse
import json
import time
from pathlib import Path

import torch

from pyhealth.datasets import POLYP_TEST_SETS, PolypDataset
from pyhealth.metrics import detection_metrics_fn
from pyhealth.models import GLIP
from pyhealth.models.miu_vl import POLYP_MANUAL_PROMPT, mlm_prompts

_HYBRID_KEYS = {"CVC-300": "cvc300", "CVC-ClinicDB": "clinicdb",
                "CVC-ColonDB": "colondb", "Kvasir": "kvasir"}


def hybrid_prompts(directory, subset, k):
    """MIU-VL's per-image hybrid prompts of one test set, by file name."""
    path = Path(directory) / f"hybrid_{_HYBRID_KEYS[subset]}_path_prompt_top{k}.json"
    with open(path) as f:
        prompts = json.load(f)
    out = {}
    for image, p in prompts.items():
        names = p["name"] if isinstance(p["name"], list) else [p["name"]]
        out[Path(image).name] = [
            {"prefix": p["prefix"][i], "name": names[i], "suffix": p["suffix"][i]}
            for i in range(len(names))
        ]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="MIU-VL's DATA/POLYP folder")
    p.add_argument("--checkpoint", default=None, help="local GLIP-T checkpoint")
    p.add_argument("--prompts", nargs="+", default=["default", "manual", "mlm1", "mlm2", "mlm3"])
    p.add_argument("--hybrid-dir", default=None, help="MIU-VL's autoprompt_json folder")
    p.add_argument("--subsets", nargs="+", default=list(POLYP_TEST_SETS))
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--limit", type=int, default=None, help="images per test set")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, help="write the results as JSON")
    args = p.parse_args()

    names = list(args.prompts)
    if args.hybrid_dir:
        names += [f"hybrid{k}" for k in (1, 2, 3) if f"hybrid{k}" not in names]
    mlm = mlm_prompts("polyp", object_name="bump", topk=3) if any(
        n.startswith("mlm") for n in names) else []
    fixed = {"default": ["polyp"], "manual": [POLYP_MANUAL_PROMPT]}
    fixed.update({f"mlm{k + 1}": [prompt] for k, prompt in enumerate(mlm)})
    for name in names:
        if name in fixed:
            print(f"{name:8s} {fixed[name]}")

    dataset = PolypDataset(root=args.root, subsets=args.subsets, cache_dir=args.cache_dir)
    samples = dataset.set_task()
    model = GLIP(samples, pretrained=args.checkpoint or True).to(args.device).eval()

    results = {}
    for subset in args.subsets:
        index = [i for i in range(len(samples)) if samples[i]["subset"] == subset]
        index = index[: args.limit] if args.limit else index
        y_true = [{"boxes": samples[i]["boxes"]} for i in index]
        started = time.time()
        for name in names:
            if name.startswith("hybrid"):
                if subset not in _HYBRID_KEYS:
                    continue
                per_image = hybrid_prompts(args.hybrid_dir, subset, int(name[-1]))
                prompts = [per_image[Path(samples[i]["image"]).name] for i in index]
            else:
                prompts = [fixed[name]] * len(index)
            y_pred = []
            with torch.no_grad():
                for i, prompt in zip(index, prompts):
                    y_pred += model(image=[samples[i]["image"]], prompt=[prompt])["y_pred"]
            scores = detection_metrics_fn(y_true, y_pred, metrics=("ap", "ap50"))
            results.setdefault(name, {})[subset] = scores
            print(f"{subset:18s} {name:8s} AP {100 * scores['ap']:5.1f}  "
                  f"AP50 {100 * scores['ap50']:5.1f}  ({len(index)} images)")
        print(f"  {time.time() - started:.0f} s")

    print("\nAP (%) per test set and the average over test sets")
    print(f"{'prompt':8s} " + " ".join(f"{s[:12]:>12s}" for s in args.subsets) + "      avg")
    for name, rows in results.items():
        aps = [100 * rows[s]["ap"] for s in args.subsets if s in rows]
        cells = " ".join(f"{100 * rows[s]['ap']:12.1f}" if s in rows else f"{'-':>12s}"
                         for s in args.subsets)
        print(f"{name:8s} {cells} {sum(aps) / len(aps):8.1f}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
