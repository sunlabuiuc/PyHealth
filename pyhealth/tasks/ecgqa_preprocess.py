"""
A PyHealth task for ECG Question Answering preprocessing.

Produces one sample per QA pair, optionally attaching a resampled ECG
signal tensor via a user-provided signal_loader callable.

Dataset link:
    https://github.com/Tang-Jia-Lu/FSL_ECG_QA

Dataset paper:
    J. Tang, T. Xia, Y. Lu, C. Mascolo, and A. Saeed,
    "Electrocardiogram-language model for few-shot question answering
    with meta learning," arXiv preprint arXiv:2410.14464, 2024.

Dataset paper link:
    https://arxiv.org/abs/2410.14464

Author:
    Jovian Wang (jovianw2@illinois.edu)
    Matthew Pham (mdpham2@illinois.edu)
    Yiyun Wang (yiyunw3@illinois.edu)
"""
import torch
from typing import Any, Callable, Dict, List, Optional

from pyhealth.tasks import BaseTask


class ECGQAPreprocessing(BaseTask):
    """ECG Question Answering preprocessing task.

    For each patient (ECG recording), this task returns one sample per
    QA pair, containing the question, answer, question type, and an
    episode_class key for episodic sampling. Optionally attaches an ECG
    signal tensor via a user-provided signal_loader.

    Works with both PTB-XL and MIMIC-IV-ECG based QA data.

    Args:
        signal_loader: optional callable mapping ecg_id (int) to a signal
            tensor of shape (12, N). If None, samples are text-only.

    Each returned sample contains:
      - "question": str, the natural language question
      - "answer": str, the answer (semicolon-separated if multiple)
      - "question_type": str, one of "single-verify", "single-choose", "single-query"
      - "episode_class": str, class key for episodic sampling (template_id + attribute + answer)
      - "signal": torch.FloatTensor (only if signal_loader is provided)
    """

    task_name: str = "ECG_QA"
    input_schema: Dict[str, str] = {"question": "text"}
    output_schema: Dict[str, str] = {"answer": "text"}

    def __init__(
        self,
        signal_loader: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> None:
        self.signal_loader = signal_loader

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one patient (ECG recording). Creates one sample per QA pair.

        The ECG signal is loaded once per patient via signal_loader and
        shared across all QA pairs for the same ecg_id.
        """
        pid = patient.patient_id
        samples: List[Dict[str, Any]] = []

        events = patient.get_events("ecg_qa")
        if not events:
            return samples

        # Load signal once for this patient if loader provided.
        # If loading fails, skip the patient entirely.
        signal = None
        if self.signal_loader is not None:
            try:
                signal = self.signal_loader(int(pid))
                if not isinstance(signal, torch.Tensor):
                    signal = torch.FloatTensor(signal)
            except Exception:
                return samples

        for event in events:
            episode_class = f"{event.template_id}_{event.attribute}_{event.answer}"

            sample = {
                "patient_id": pid,
                "question": event.question,
                "answer": event.answer,
                "question_type": event.question_type,
                "episode_class": episode_class,
            }
            if signal is not None:
                sample["signal"] = signal

            samples.append(sample)

        return samples
