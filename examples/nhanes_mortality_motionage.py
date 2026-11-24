"""MotionAge 학습/평가 실행 스크립트 (CS 598 DLH, MotionAge reproduction).

Contributors: Inoh Pak (inohpak2), Yejin Kim (yejink6)
Paper: Song et al., MotionAge: A Deep Learning Framework for Biological Age
Prediction from Wearable Activity, MLHC 2026 (PMLR 340).
https://github.com/jackie-jiaqi-yin/MotionAge

모델(``pyhealth.models.MotionAge``)과 데이터(Task가 만든 SampleDataset)를
연결해서 학습하고, 창(window) 단위와 참가자(SEQN) 단위 성능을 출력한다.

사용법:
    # 1) 합성 데이터로 연결 확인 (NHANES 없이 1~2분)
    python examples/nhanes_mortality_motionage.py --synthetic

    # 2) 실제 데이터: 아래 build_real_dataset()의 TODO를 채운 뒤
    python examples/nhanes_mortality_motionage.py --static full --pooling masked
    python examples/nhanes_mortality_motionage.py --static none --pooling mean   # 비교 실험

Task가 내보내야 하는 샘플 형식 (한 창 = 샘플 하나):
    patient_id : SEQN (문자열)
    intensity  : 길이 1008 float 리스트, 5분 epoch 평균 활동량
    hour       : 길이 1008 int 리스트, 0-23
    day        : 길이 1008 int 리스트, 0-6
    wear_mask  : 길이 1008 0/1 리스트, 1 = 착용
    static     : [나이, 성별, 소득, BMI, 허리둘레, 수축기, 이완기] (결측 NaN)
    mortality  : 0/1 (5년 내 사망)
스키마는 아래 INPUT_SCHEMA / OUTPUT_SCHEMA와 같아야 한다.
"""

import argparse
import random
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from pyhealth.datasets import (
    SampleDataset,
    create_sample_dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import MotionAge
from pyhealth.trainer import Trainer

INPUT_SCHEMA: Dict[str, Any] = {
    "intensity": "tensor",
    "hour": "tensor",
    "day": "tensor",
    "wear_mask": "tensor",
    "static": (
        "tensor",
        {
            "normalization": "zscore",
            "missing_indicators": True,
            "impute_strategy": "zero",
        },
    ),
}
# 사망자가 약 6%라 양성 샘플에 가중치를 준다 (제안서의 class-weighted BCE).
OUTPUT_SCHEMA: Dict[str, Any] = {"mortality": ("binary", {"pos_weight": "balanced"})}


def synthetic_samples(
    num_patients: int = 80, windows_per_patient: int = 2, seq_len: int = 96
) -> List[Dict[str, Any]]:
    """NHANES와 같은 형식의 가짜 샘플을 만든다 (연결 확인용).

    사망자는 활동량이 조금 낮게 만들어서 모델이 배울 신호가 있게 한다.
    """
    rng = random.Random(0)
    samples = []
    for p in range(num_patients):
        died = int(p % 5 == 0)
        level = 0.4 if died else 0.7
        static = [
            rng.gauss(65 if died else 55, 10),
            float(p % 2),
            rng.choice([1.0, 2.0, 3.0]),
            rng.gauss(28, 4),
            float("nan") if p % 7 == 0 else rng.gauss(95, 12),
            rng.gauss(125, 15),
            rng.gauss(75, 10),
        ]
        for w in range(windows_per_patient):
            mask = [1.0 if rng.random() > 0.25 else 0.0 for _ in range(seq_len)]
            samples.append(
                {
                    "patient_id": f"seqn{p}",
                    "intensity": [(level + 0.3 * rng.random()) * m for m in mask],
                    "hour": [(t // 12) % 24 for t in range(seq_len)],
                    "day": [(w * 3 + t // 288) % 7 for t in range(seq_len)],
                    "wear_mask": mask,
                    "static": static,
                    "mortality": died,
                }
            )
    return samples


def build_real_dataset() -> SampleDataset:
    """실제 NHANES SampleDataset을 만든다.

    TODO(예진): 직접 만든 Dataset/Task로 바꿔 주세요. 예시:

        from pyhealth.datasets import NHANESActivityDataset
        from pyhealth.tasks import NHANESMortalityPrediction
        base = NHANESActivityDataset(root="/path/to/nhanes")
        return base.set_task(NHANESMortalityPrediction())

    Task 클래스의 input_schema / output_schema를 이 파일의 INPUT_SCHEMA /
    OUTPUT_SCHEMA와 같게 맞춰 주세요 (특히 pos_weight).
    """
    raise NotImplementedError("build_real_dataset()의 TODO를 채워 주세요.")


def participant_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, patient_ids: List[str]
) -> Dict[str, float]:
    """창 단위 예측을 참가자별로 평균낸 뒤 AUROC/AUPRC를 계산한다."""
    risk = MotionAge.aggregate_participant_risk(
        torch.as_tensor(y_prob), patient_ids
    )
    label = {str(pid): float(y) for pid, y in zip(patient_ids, y_true.reshape(-1))}
    pids = sorted(risk)
    truth = [label[p] for p in pids]
    probs = [risk[p] for p in pids]
    return {
        "participants": len(pids),
        "roc_auc": roc_auc_score(truth, probs),
        "pr_auc": average_precision_score(truth, probs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--static", choices=["none", "full"], default="full")
    parser.add_argument("--pooling", choices=["masked", "mean", "last"],
                        default="masked")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.synthetic:
        dataset = create_sample_dataset(
            samples=synthetic_samples(),
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
            dataset_name="nhanes_synthetic",
        )
    else:
        dataset = build_real_dataset()

    # 같은 참가자의 창이 train/test에 나뉘지 않도록 참가자 단위로 나눈다.
    train_ds, val_ds, test_ds = split_by_patient(
        dataset, [0.7, 0.15, 0.15], seed=args.seed
    )
    train_loader = get_dataloader(train_ds, args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, args.batch_size, shuffle=False)

    model = MotionAge(
        dataset,
        static_key="static" if args.static == "full" else None,
        hidden_dim=args.hidden_dim,
        pooling=args.pooling,
    )
    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc"],
                      enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": args.lr},
        monitor="pr_auc",  # 제안서: validation AUPRC로 best 모델 선택
    )

    print("\n[window-level test]", trainer.evaluate(test_loader))
    y_true, y_prob, _, pids = trainer.inference(test_loader,
                                                return_patient_ids=True)
    print("[participant-level test]",
          participant_metrics(np.asarray(y_true), np.asarray(y_prob), pids))
    print("config:", vars(args))


if __name__ == "__main__":
    main()
