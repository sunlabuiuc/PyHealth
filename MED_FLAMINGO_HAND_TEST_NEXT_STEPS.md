# Med-Flamingo LoRA Hand-Test Next Steps

This guide is for running the full Med-Flamingo LoRA pipeline on Colab T4 (16GB) or equivalent single GPU.

## 1) Environment Setup

### Python packages

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft huggingface_hub bitsandbytes
pip install open-flamingo
pip install -e .
```

If `open-flamingo` installation fails, use the repository install route:

```bash
pip install git+https://github.com/mlfoundations/open_flamingo.git
```

### Hugging Face authentication and gated access

```bash
huggingface-cli login
export HF_TOKEN="<your_hf_token>"
```

You must have access to required gated LLaMA assets.

## 2) Required Inputs

- `LLAMA_PATH`: local directory for your LLaMA model/tokenizer files.
- Dataset root with image files and annotations for either VQA-RAD or Path-VQA.

Expected annotation fields per record (or equivalent aliases):
- image path
- question
- answer
- split
- question_id
- image_id

## 3) One-Command LoRA Training + Eval

```bash
python examples/med_flamingo_lora_train_eval.py \
  --dataset vqa_rad \
  --root /path/to/vqa_rad \
  --llama_path /path/to/llama \
  --quantization auto \
  --epochs 1 \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --grad_accum_steps 2 \
  --metrics exact_match \
  --output_dir ./runs/med_flamingo_vqarad
```

For Path-VQA:

```bash
python examples/med_flamingo_lora_train_eval.py \
  --dataset path_vqa \
  --root /path/to/path_vqa \
  --llama_path /path/to/llama \
  --quantization auto \
  --epochs 1 \
  --output_dir ./runs/med_flamingo_pathvqa
```

## 4) Evaluate Trained Adapter with Few-Shot Ablation

```bash
python examples/vqarad_generativevqa_medflamingo.py \
  --dataset vqa_rad \
  --root /path/to/vqa_rad \
  --llama_path /path/to/llama \
  --adapter_dir ./runs/med_flamingo_vqarad/best_adapter \
  --shots 0,1,3,5 \
  --metrics exact_match \
  --output_json ./runs/med_flamingo_vqarad/ablation.json
```

## 5) Output Artifacts

After training, inspect:

- `best_adapter/adapter_config.json`
- `best_adapter/adapter_model.safetensors` (or equivalent)
- `best_adapter/med_flamingo_config.json`
- `last_adapter/*`
- `metrics_history.json`
- `fit_summary.json`
- run summary JSON from script output

## 6) Running Tests

Fast unit tests:

```bash
./.venv/bin/python -m unittest -v \
  tests/core/test_vqa_rad_dataset.py \
  tests/core/test_path_vqa_dataset.py \
  tests/core/test_generative_vqa_task.py \
  tests/core/test_med_flamingo.py \
  tests/core/test_generative_metrics.py
```

Optional heavy smoke test (real model path + gated access required):

```bash
export PYHEALTH_RUN_HEAVY_TESTS=1
export LLAMA_PATH=/path/to/llama
export HF_TOKEN=<your_hf_token>
./.venv/bin/python -m unittest -v tests/integration/test_med_flamingo_lora_smoke.py
```

## 7) Common Failures and Fixes

### `ImportError: open_flamingo is required`
Install `open-flamingo` and verify import:

```bash
python -c "import open_flamingo; print('ok')"
```

### `ImportError: bitsandbytes` in 8-bit mode
Install `bitsandbytes` or switch to `--quantization fp16`.

### `llama_path does not exist`
Set the correct local LLaMA directory and ensure tokenizer/model files are present.

### Hugging Face download/auth failures
Confirm `HF_TOKEN` is set and account has gated model access.

### CUDA OOM
- Reduce `--train_batch_size`
- Increase `--grad_accum_steps`
- Lower `--max_new_tokens`
- Use fewer shots for generation evaluation
- Keep `--quantization auto` on T4

## 8) Recommended First Real Run

1. Run 1 epoch with `train_batch_size=1`, `grad_accum_steps=2`.
2. Confirm adapter artifacts are written.
3. Run ablation with `--shots 0,1,3,5`.
4. Inspect generated predictions JSON for clinical quality.
5. If stable, increase epochs and run with optional `bertscore_f1`.
