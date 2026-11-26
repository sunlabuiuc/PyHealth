"""Train SafeDrug and refine its predictions on the bundled MIMIC-III demo.

Run from the repository root::

    python examples/drug_recommendation/giantmed_mimic3.py --ndc-rxcui /path/to/ndc2rxnorm_mapping.txt --ndc-atc /path/to/ndc2atc_level4.csv --model Qwen/Qwen3-8B

The demo is tiny and is intended only to exercise the pipeline, not to
reproduce the paper. The model and code-mapping resources may need downloads.
Without --model a no-op responder exercises the pipeline without an LLM.
The optional MEDI evidence filter is enabled only when its files are supplied.
"""

import argparse
import ast
import csv
import gzip
import json
import os
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import (
    MIMIC3Dataset,
    MIMIC4EHRDataset,
    PatientSplit,
    get_dataloader,
)
from pyhealth.medcode import ATC, ICD9CM, ICD9PROC
from pyhealth.metrics import ddi_rate_score, multilabel_metrics_fn
from pyhealth.models import (
    GiantMed,
    HuggingFaceGiantMedGenerator,
    MediEvidenceFilter,
    SafeDrug,
)
from pyhealth.tasks import DrugRecommendationMIMIC3, DrugRecommendationMIMIC4
from pyhealth.trainer import Trainer


class OpenRouterGenerator:
    """Example adapter for OpenRouter; the API key stays in the environment."""

    def __init__(self, model, max_tokens, max_input_tokens=None, response_trace=None):
        self.model = model
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.usage = []
        self.response_trace = response_trace
        self.api_key = os.environ["OPENROUTER_API_KEY"]

    def __call__(self, prompt):
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": self.max_tokens,
                "reasoning": {"enabled": False},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=240) as response:
            payload = json.load(response)
        self.usage.append(payload.get("usage", {}))
        actual_tokens = payload.get("usage", {}).get("prompt_tokens")
        if (
            self.max_input_tokens is not None
            and actual_tokens is not None
            and actual_tokens > self.max_input_tokens
        ):
            raise ValueError(
                f"Provider reported {actual_tokens} input tokens, exceeding "
                f"the {self.max_input_tokens} token limit"
            )
        content = payload["choices"][0]["message"]["content"]
        if self.response_trace:
            with open(self.response_trace, "a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"content": content, "usage": self.usage[-1]}) + "\n"
                )
        return content


def count_chat_tokens(tokenizer, prompt):
    """Count a single rendered chat, including its generation template."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)


def evaluate_predictions(y_true, scores, patient_ids, ddi):
    """Average visit metrics within each patient, then across patients."""
    scores = np.asarray(scores)
    if len(y_true) != len(scores) or len(scores) != len(patient_ids) or not len(scores):
        raise ValueError(
            "targets, scores and patient IDs must have equal nonzero length"
        )
    per_patient = defaultdict(list)
    counts = defaultdict(list)
    predictions = []
    for patient_id, target, probability in zip(patient_ids, y_true, scores):
        per_patient[patient_id].append(
            multilabel_metrics_fn(
                np.asarray([target]),
                np.asarray([probability]),
                metrics=["jaccard_samples", "f1_samples", "pr_auc_samples"],
                threshold=0.5,
            )
        )
        prediction = np.flatnonzero(probability >= 0.5)
        predictions.append(prediction)
        counts[patient_id].append(len(prediction))
    metrics = {
        key: float(
            np.mean(
                [
                    np.mean([visit[key] for visit in visits])
                    for visits in per_patient.values()
                ]
            )
        )
        for key in ("jaccard_samples", "f1_samples", "pr_auc_samples")
    }
    metrics["ddi_rate"] = ddi_rate_score(predictions, ddi)
    metrics["mean_medications"] = float(np.mean([np.mean(v) for v in counts.values()]))
    return metrics


def summarize_decision_changes(y_true, baseline, refined):
    """Count changed decisions against recorded medications, pooled by visit."""
    truth = np.asarray(y_true, dtype=bool)
    before = np.asarray(baseline) >= 0.5
    after = np.asarray(refined) >= 0.5
    if truth.shape != before.shape or before.shape != after.shape:
        raise ValueError("targets and scores must have matching shapes")
    added, removed = ~before & after, before & ~after
    return {
        "added_recorded": int((added & truth).sum()),
        "added_unrecorded": int((added & ~truth).sum()),
        "removed_recorded": int((removed & truth).sum()),
        "removed_unrecorded": int((removed & ~truth).sum()),
    }


def load_reference_ndc_atc3(ndc_rxcui_path, ndc_atc_path):
    """Construct the reference's NDC -> RxCUI -> first ATC3 mapping."""
    ndc_rxcui = ast.literal_eval(Path(ndc_rxcui_path).read_text(encoding="utf-8-sig"))
    rxcui_atc = {}
    with open(ndc_atc_path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"RXCUI", "ATC4"}.issubset(reader.fieldnames or []):
            raise ValueError("NDC-to-ATC file must contain RXCUI and ATC4")
        for row in reader:
            rx, atc = str(row["RXCUI"]).strip(), str(row["ATC4"]).strip()
            if rx and atc:
                rxcui_atc.setdefault(rx, atc[:4])
    return {
        str(ndc).strip(): rxcui_atc[str(rx).strip()]
        for ndc, rx in ndc_rxcui.items()
        if str(rx).strip() in rxcui_atc
    }


def ndc_coverage(prescriptions_path, ndc_map):
    """Count usable prescription NDC rows mapped by the selected vocabulary."""
    total = mapped = 0
    with gzip.open(
        prescriptions_path, "rt", encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            ndc = str(row.get("ndc", row.get("NDC", ""))).strip()
            if not ndc or ndc == "0" or ndc.lower() in {"nan", "none", "<na>"}:
                continue
            total += 1
            mapped += int(ndc in ndc_map)
    return mapped, total


def main(mimic_version=3):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=(
            str(Path(__file__).resolve().parents[2] / "test-resources/core/mimic3demo")
            if mimic_version == 3
            else None
        ),
        required=mimic_version == 4,
    )
    parser.add_argument("--model", help="Hugging Face causal chat model or local path")
    parser.add_argument(
        "--openrouter-model", help="OpenRouter model ID; sends prompts to OpenRouter"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="SafeDrug training epochs"
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument(
        "--input-token-reserve",
        type=int,
        default=256,
        help="allowance for remote provider chat formatting",
    )
    parser.add_argument(
        "--prompt-tokenizer", help="Hugging Face tokenizer name or local path"
    )
    parser.add_argument(
        "--prompt-template-file",
        help="patient-independent single-line instruction template containing {boundary}",
    )
    parser.add_argument("--boundary-low", type=float, default=0.3)
    parser.add_argument("--boundary-high", type=float, default=0.7)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        help="maximum target medications per LLM request",
    )
    parser.add_argument(
        "--save-responses", help="optional local JSON path for generated responses"
    )
    parser.add_argument(
        "--response-trace",
        help="optional local JSONL response trace; may contain patient data",
    )
    parser.add_argument(
        "--max-test-visits",
        type=int,
        default=None,
        help="limit inference visits for a smoke run",
    )
    parser.add_argument(
        "--demo-refinement",
        action="store_true",
        help="use a deterministic fake LLM to exercise score updates",
    )
    parser.add_argument("--medi", help="MEDI.csv")
    parser.add_argument("--ndc-rxcui", help="reference NDC to RxCUI mapping")
    parser.add_argument("--ndc-atc", help="reference RxCUI to ATC mapping")
    parser.add_argument("--ingredient-cache", help="reference ingredient JSON cache")
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.max_new_tokens < 1
        or (args.max_test_visits is not None and args.max_test_visits < 1)
    ):
        parser.error("epochs, token and visit limits must be positive")
    if not 0 <= args.input_token_reserve < args.max_input_tokens:
        parser.error(
            "input token reserve must be nonnegative and smaller than the input limit"
        )
    if (
        sum(bool(x) for x in (args.model, args.openrouter_model, args.demo_refinement))
        > 1
    ):
        parser.error("choose only one model or demo responder")
    if args.openrouter_model and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY must be set in the environment")
    if args.candidate_batch_size is not None:
        if args.candidate_batch_size < 1:
            parser.error("--candidate-batch-size must be positive")
        if not (
            args.prompt_tokenizer
            or args.model
            or args.openrouter_model == "qwen/qwen3-14b"
        ):
            parser.error(
                "--candidate-batch-size requires --prompt-tokenizer or --model"
            )
    if not args.ndc_rxcui or not args.ndc_atc:
        parser.error("supply --ndc-rxcui and --ndc-atc for reference medication labels")
    if bool(args.medi) != bool(args.ingredient_cache):
        parser.error("supply --medi and --ingredient-cache together")
    filter_files = [args.medi, args.ndc_rxcui, args.ndc_atc, args.ingredient_cache]
    prompt_template = None
    if args.prompt_template_file:
        prompt_template = (
            Path(args.prompt_template_file).read_text(encoding="utf-8").rstrip("\r\n")
        )
    torch.manual_seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as cache:
        dataset_class = MIMIC3Dataset if mimic_version == 3 else MIMIC4EHRDataset
        dataset = dataset_class(
            root=args.root,
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cache,
        )
        ndc_map = load_reference_ndc_atc3(args.ndc_rxcui, args.ndc_atc)
        prescriptions = (
            Path(args.root) / "PRESCRIPTIONS.csv.gz"
            if mimic_version == 3
            else Path(args.root) / "hosp/prescriptions.csv.gz"
        )
        mapped_rows, usable_rows = ndc_coverage(prescriptions, ndc_map)
        print(
            f"Reference label coverage: {mapped_rows}/{usable_rows} usable prescription rows; unmapped rows excluded."
        )
        task_class = (
            DrugRecommendationMIMIC3 if mimic_version == 3 else DrugRecommendationMIMIC4
        )
        task = task_class(ndc_to_atc3=ndc_map)
        print(
            f"Reference NDC-to-ATC3 mappings: {len(ndc_map)}; unmapped NDCs are excluded."
        )
        train, val, test = dataset.set_task(
            task, split=PatientSplit((0.8, 0.1, 0.1), seed=0)
        )
        print("Split sample counts:", len(train), len(val), len(test))
        print(
            "Split patient counts:",
            *(len(part.patient_to_index) for part in (train, val, test)),
        )
        if not all(len(part) for part in (train, val, test)):
            raise RuntimeError(
                "The subset produced an empty split; use a larger dataset."
            )
        model = SafeDrug(train)
        trainer = Trainer(model=model, metrics=["pr_auc_samples"], enable_logging=False)
        trainer.train(
            get_dataloader(train, batch_size=8, shuffle=True),
            val_dataloader=get_dataloader(val, batch_size=8),
            epochs=args.epochs,
        )

        # Raw task samples supply clinical codes for prompts and retrieval. The
        # processed dataset supplies tensors for SafeDrug, in the same visit order.
        def raw_samples(part):
            return {
                (str(s["patient_id"]), str(s["visit_id"])): s
                for pid in part.patient_to_index
                for s in task(dataset.get_patient(pid))
            }

        raw_train, raw_test = raw_samples(train), raw_samples(test)
        vocab = train.output_processors["drugs"].label_vocab
        codes = [code for code, _ in sorted(vocab.items(), key=lambda pair: pair[1])]
        if args.model:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model)
            llm = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype="auto", device_map="auto"
            )
            generate = HuggingFaceGiantMedGenerator(
                llm, tokenizer, max_new_tokens=args.max_new_tokens
            )
        elif args.openrouter_model:
            generate = OpenRouterGenerator(
                args.openrouter_model,
                args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                response_trace=args.response_trace,
            )
            print(f"Sending demo prompts to OpenRouter model {args.openrouter_model}.")
        elif args.demo_refinement:

            def generate(prompt):
                candidates_line = next(
                    line
                    for line in prompt.splitlines()
                    if line.startswith("Candidates: ")
                )
                candidates = json.loads(candidates_line.removeprefix("Candidates: "))
                if isinstance(candidates, list):
                    if "Only the first " in prompt:
                        import re

                        count = int(
                            re.search(r"Only the first (\d+) rows", prompt).group(1)
                        )
                        candidates = candidates[:count]
                    candidates = [row[0] for row in candidates]
                return json.dumps({code: 0.8 for code in candidates})

            print(
                "Using a fake LLM to exercise score updates; results have no clinical meaning."
            )
        else:
            print("No LLM selected: no-op refinement; scores will be unchanged.")

            def generate(prompt):
                return "{}"

        # Translate codes for meaningful LLM evidence. Unknown codes remain visible.
        all_raw = list(raw_train.values()) + list(raw_test.values())
        names = {}
        terminologies = [("drugs", ATC())]
        if mimic_version == 3:
            terminologies += [("conditions", ICD9CM()), ("procedures", ICD9PROC())]
        for field, terminology in terminologies:
            values = (
                set(codes)
                if field == "drugs"
                else {
                    code
                    for sample in all_raw
                    for visit in sample[field]
                    for code in visit
                }
            )
            names[field] = {
                code: terminology.lookup(code) for code in values if code in terminology
            }
        evidence_filter = None
        if all(filter_files):
            evidence_filter = MediEvidenceFilter.from_reference_files(
                list(raw_train.values()),
                prescriptions,
                args.medi,
                args.ndc_rxcui,
                args.ndc_atc,
                args.ingredient_cache,
                ndc_to_atc3=lambda ndc: task.map_drugs([ndc]),
            )
            print(
                "MEDI filter:",
                len(evidence_filter.visit_drug_ingredients),
                "training visit/drug mappings,",
                len(evidence_filter.ingredient_diagnoses),
                "ingredients with indications",
            )
        else:
            print(
                "MEDI evidence filter disabled; results differ from the released code."
            )
        tokenizer_name = args.prompt_tokenizer or args.model
        if args.openrouter_model == "qwen/qwen3-14b" and not tokenizer_name:
            tokenizer_name = "Qwen/Qwen3-14B"
        if args.openrouter_model and not tokenizer_name:
            parser.error("--prompt-tokenizer is required for this OpenRouter model")
        token_counter = None
        if tokenizer_name:
            from transformers import AutoTokenizer

            budget_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            def token_counter(prompt):
                return count_chat_tokens(budget_tokenizer, prompt)

        refiner = GiantMed(
            codes,
            model.safedrug.ddi_adj.detach().cpu().numpy(),
            generate,
            list(raw_train.values()),
            code_names=names,
            evidence_filter=evidence_filter,
            boundary_low=args.boundary_low,
            boundary_high=args.boundary_high,
            prompt_token_counter=token_counter,
            max_prompt_tokens=(args.max_input_tokens - args.input_token_reserve)
            if token_counter
            else None,
            candidate_batch_size=args.candidate_batch_size,
            prompt_template=prompt_template,
        )
        y_true, baseline, refined, patient_ids = [], [], [], []
        statuses = Counter()
        candidate_count = updated_count = changed_count = 0
        responses = []
        budgets = []
        evidence_outcomes = Counter()
        evidence_by_system = Counter()
        model.eval()
        for batch in get_dataloader(test, batch_size=1, shuffle=False):
            if args.max_test_visits is not None and len(y_true) >= args.max_test_visits:
                break
            with torch.no_grad():
                out = model(**batch)
            key = (str(batch["patient_id"][0]), str(batch["visit_id"][0]))
            scores = out["y_prob"][0].detach().cpu().numpy()
            if evidence_filter:
                candidate_codes = {
                    codes[i]
                    for i in np.flatnonzero(
                        (scores >= refiner.low)
                        & (
                            scores <= refiner.high
                            if refiner.upper_inclusive
                            else scores < refiner.high
                        )
                    )
                }
                for neighbour in refiner.retrieve(raw_test[key]):
                    for drug in set(neighbour["drugs"]) & candidate_codes:
                        status = evidence_filter.evidence_status(
                            raw_test[key], neighbour, drug
                        )
                        evidence_outcomes[status] += 1
                        systems = {
                            str(code).split("_", 1)[0] if "_" in str(code) else "9"
                            for code in raw_test[key]["conditions"][-1]
                        }
                        evidence_by_system[("+".join(sorted(systems)), status)] += 1
            try:
                result = refiner.refine_with_details(raw_test[key], scores)
            except Exception as exc:
                # The example is a long-running experiment: retain the backbone
                # for this visit and continue, while making the failure visible.
                print(
                    f"Refinement failed for visit {len(y_true) + 1}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                result = {
                    "status": "refinement_error",
                    "candidates": [
                        codes[i]
                        for i in np.flatnonzero(
                            (scores >= refiner.low)
                            & (
                                scores <= refiner.high
                                if refiner.upper_inclusive
                                else scores < refiner.high
                            )
                        )
                    ],
                    "updated": [],
                    "changed": [],
                    "probabilities": scores.copy(),
                    "response": None,
                }
            if result.get("prompt_budget"):
                budgets.append(result["prompt_budget"])
            if args.save_responses:
                responses.append(
                    {
                        "status": result["status"],
                        "candidates": len(result["candidates"]),
                        "updated": len(result["updated"]),
                        "changed": len(result["changed"]),
                        "prompt_budget": result.get("prompt_budget"),
                        "response": result["response"],
                        "responses": result.get("responses", []),
                    }
                )
            statuses[result["status"]] += 1
            candidate_count += len(result["candidates"])
            updated_count += len(result["updated"])
            changed_count += len(result["changed"])
            y_true.append(out["y_true"][0].cpu().numpy())
            baseline.append(scores)
            refined.append(result["probabilities"])
            patient_ids.append(key[0])
            print(
                f"Processed visit {len(y_true)}: {result['status']}; "
                f"accepted {len(result['updated'])}/{len(result['candidates'])} scores",
                flush=True,
            )
        print(
            "Refinement statuses:",
            dict(statuses),
            f"accepted {updated_count}/{candidate_count} boundary scores; changed {changed_count}",
        )
        if budgets:
            print("Prompt budget by visit:", json.dumps(budgets))
        differences = np.abs(np.asarray(refined) - np.asarray(baseline))
        decision_changes = np.sum(
            (np.asarray(refined) >= 0.5) != (np.asarray(baseline) >= 0.5)
        )
        print(
            f"Mean absolute score change: {differences.mean():.6f}; decision changes: {decision_changes}"
        )
        print(
            "Decision changes against recorded medications:",
            json.dumps(summarize_decision_changes(y_true, baseline, refined)),
        )
        if evidence_filter:
            print(f"MEDI evidence outcomes: {dict(evidence_outcomes)}")
            print(
                "MEDI outcomes by query ICD system:",
                {
                    f"{system}:{status}": count
                    for (system, status), count in evidence_by_system.items()
                },
            )
        if args.save_responses:
            Path(args.save_responses).write_text(
                json.dumps(responses, indent=2), encoding="utf-8"
            )
        if args.openrouter_model:
            print("OpenRouter usage:", json.dumps(generate.usage))
            if any(
                item.get("prompt_tokens", 0) > args.max_input_tokens
                for item in generate.usage
            ):
                print("WARNING: provider reported an input above the requested limit.")
        for name, scores in (("SafeDrug", baseline), ("GiantMed", refined)):
            # Average visits within patient, then patients; pool DDI pairs.
            metrics = evaluate_predictions(y_true, scores, patient_ids, refiner.ddi)
            print(name, metrics)
        print("Demo results are not a reproduction of the paper.")


if __name__ == "__main__":
    main()
