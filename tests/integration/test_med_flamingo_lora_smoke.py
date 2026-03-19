import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MedFlamingo
from pyhealth.tasks import GenerativeMedicalVQA


@unittest.skipUnless(
    os.getenv("PYHEALTH_RUN_HEAVY_TESTS") == "1",
    "Set PYHEALTH_RUN_HEAVY_TESTS=1 to run heavy MedFlamingo integration tests",
)
class TestMedFlamingoLoRASmoke(unittest.TestCase):
    def test_tiny_lora_smoke(self):
        if importlib.util.find_spec("open_flamingo") is None:
            self.skipTest("open_flamingo is not installed")
        if importlib.util.find_spec("bitsandbytes") is None:
            self.skipTest("bitsandbytes is not installed")

        llama_path = os.getenv("LLAMA_PATH")
        if not llama_path:
            self.skipTest("LLAMA_PATH is required")
        if not os.getenv("HF_TOKEN"):
            self.skipTest("HF_TOKEN is required for gated model access")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            img_train = root / "train.png"
            img_test = root / "test.png"
            Image.new("RGB", (16, 16), color=(120, 120, 120)).save(img_train)
            Image.new("RGB", (16, 16), color=(60, 60, 60)).save(img_test)

            task = GenerativeMedicalVQA()
            input_schema = task.input_schema
            output_schema = task.output_schema

            train_samples = [
                {
                    "patient_id": "p1",
                    "record_id": "q1",
                    "image_path": str(img_train),
                    "question": "what modality?",
                    "answer": "xray",
                    "split": "train",
                    "question_id": "q1",
                    "image_id": "img1",
                    "dataset": "vqa_rad",
                }
            ]
            val_samples = [
                {
                    "patient_id": "p2",
                    "record_id": "q2",
                    "image_path": str(img_test),
                    "question": "is lesion present?",
                    "answer": "no",
                    "split": "test",
                    "question_id": "q2",
                    "image_id": "img2",
                    "dataset": "vqa_rad",
                }
            ]

            train_dataset = create_sample_dataset(
                samples=train_samples,
                input_schema=input_schema,
                output_schema=output_schema,
                dataset_name="smoke_train",
                task_name=task.task_name,
            )
            val_dataset = create_sample_dataset(
                samples=val_samples,
                input_schema=input_schema,
                output_schema=output_schema,
                dataset_name="smoke_val",
                task_name=task.task_name,
            )

            train_loader = get_dataloader(train_dataset, batch_size=1, shuffle=False)
            val_loader = get_dataloader(val_dataset, batch_size=1, shuffle=False)

            model = MedFlamingo(
                dataset=train_dataset,
                llama_path=llama_path,
                quantization="8bit",
                enable_lora=True,
                lora_r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                max_new_tokens=16,
            )

            output_dir = root / "smoke_outputs"
            summary = model.fit_lora(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                epochs=1,
                grad_accum_steps=1,
                eval_every_n_steps=1,
                metrics=("exact_match",),
                output_dir=str(output_dir),
            )
            self.assertIn("best_metric", summary)

            model.load_lora_adapter(str(output_dir / "best_adapter"), is_trainable=False)
            eval_result = model.predict_generation(val_loader, metrics=("exact_match",))
            self.assertIn("metrics", eval_result)
            self.assertIn("predictions", eval_result)

            train_dataset.close()
            val_dataset.close()


if __name__ == "__main__":
    unittest.main()
