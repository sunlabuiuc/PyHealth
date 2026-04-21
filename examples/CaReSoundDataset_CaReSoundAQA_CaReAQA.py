from pyhealth.datasets import CaReSoundDataset
from pyhealth.tasks import CaReSoundAQA
import librosa
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from peft import get_peft_model, LoraConfig

# ==============================================================================
# MODEL SETUP INSTRUCTIONS
# ==============================================================================
# 1. DOWNLOAD: Obtain the 'CaReAQAmodel.pt' weights file from the official
#    project source (e.g., Hugging Face or the provided Google Drive link).
#
# 2. DIRECTORY STRUCTURE: Create the following folder path on your Mac:
#    /Users/rahuld/Downloads/CaReAQA/CaReAQAModel/
#
# 3. PLACEMENT: Move the downloaded file into that folder.
#    Ensure the file is named EXACTLY 'CaReAQAmodel.pt'.
#
# 4. VERIFICATION: Your final file path must match the variable below:
#    local_careqa_path = "/Users/rahuld/Downloads/CaReAQA/CaReAQAModel/CaReAQAmodel.pt"
# ==============================================================================

# ==============================================================================
# DATASET SETUP INSTRUCTIONS
# ==============================================================================
# 1. DOWNLOAD: Obtain the 5 source audio datasets: ICBHI, KAUH, CirCor,
#    SPRSound, and ZCHSound from their respective open-access repositories.
#
# 2. DIRECTORY STRUCTURE: Maintain the following folder path on your Mac:
#    /Users/rahuld/Downloads/CaReAQA/datasets/
#
# 3. PLACEMENT: Ensure audio files (.wav) are located within their respective
#    folders (e.g., 'ICBHI Respiratory Sound Dataset', 'ZCHSound', etc.).
#    The mapping logic will scan these directories to link audio to QA pairs.
#
# 4. VERIFICATION: Your current 'ls' output confirms the following layout:
#    /Users/rahuld/Downloads/CaReAQA/datasets/
#    ├── CirCor Pediatric Heart Sound Dataset/
#    ├── ICBHI Respiratory Sound Dataset/
#    ├── KAUH Respiratory Dataset/
#    ├── SPRSound Pediatric Respiratory Dataset/
#    ├── ZCHSound/
#    └── caresound_metadata.csv
# ==============================================================================

local_careqa_path = "/Users/rahuld/Downloads/CaReAQA/CaReAQAModel/CaReAQAmodel.pt"
path_to_dataset = "/Users/rahuld/Downloads/CaReAQA/datasets"
local_llama_dir = "/Users/rahuld/Downloads/meta-llama/Llama-3.2-3B"

# set variable to true to use model to generate answer
generateModelAnswer = False


# ==========================================
# 1. MODEL CODE START AI GENERATED BASED ON .pt file
# ==========================================


# ==========================================
# 1. CLIPCAP TRANSFORMER MAPPER (PREFIX PROJECTOR)
# ==========================================
class Mlp(nn.Module):
    def __init__(self, in_dim, h_dim, out_d=None, act=nn.GELU(), dropout=0.0):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim**-0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        kv = self.to_keys_values(y).reshape(
            b, m, 2, self.num_heads, c // self.num_heads
        )
        keys, values = kv[:, :, 0], kv[:, :, 1]
        attention = torch.einsum("bnhd,bmhd->bnmh", queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum("bnmh,bmhd->bnhd", attention, values).reshape(b, n, c)
        return self.project(out), attention


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim_self,
        dim_ref,
        num_heads,
        mlp_ratio=4.0,
        bias=False,
        dropout=0.0,
        act=nn.GELU(),
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(
            dim_self, dim_ref, num_heads, bias=bias, dropout=dropout
        )
        self.norm2 = norm_layer(dim_self)
        self.mlp = Mlp(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

    def forward(self, x, y=None, mask=None):
        x_, _ = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerMapper(nn.Module):
    def __init__(
        self, dim_clip, dim_embedding, prefix_length, clip_length, num_layers=8
    ):
        super().__init__()
        self.clip_length = clip_length
        self.transformer = nn.ModuleList(
            [
                TransformerLayer(
                    dim_embedding,
                    dim_embedding,
                    8,
                    2.0,
                    act=nn.GELU(),
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(
            torch.randn(prefix_length, dim_embedding), requires_grad=True
        )

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(
            x.shape[0], *self.prefix_const.shape
        )
        prefix = torch.cat((x, prefix), dim=1)
        for layer in self.transformer:
            prefix = layer(prefix)
        return prefix[:, self.clip_length :]


# ==========================================
# 2. AUDIO ENCODER (EFFICIENTNET)
# ==========================================
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        # Using timm to automatically build the EfficientNet architecture matching the state_dict
        self.efficientnet = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0
        )

    def forward(self, x):
        # Format the 3D spectrogram into a 4D batch for the CNN
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn1(x)
        x = self.efficientnet.forward_features(x)
        # Global Average Pool to turn feature map into a 1280-dim vector
        x = x.mean(dim=[2, 3])
        return x


class AudioModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AudioEncoder()

    def extract_feature(self, x, dim=1280):
        return self.encoder(x)


# ==========================================
# 3. MAIN MODEL WRAPPER
# ==========================================
class AudioQAModel(nn.Module):
    def __init__(
        self,
        llm_type,
        opera_checkpoint_path=None,
        prefix_length=8,
        clip_length=1,
        setting="lora",
        mapping_type="Transformer",
        fine_tune_opera=True,
        args=None,
    ):
        super().__init__()

        # Load the base Llama model
        print(f"Loading Base LLM: {llm_type}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_type, torch_dtype=torch.float16
        )

        # Hook up LoRA adapters matching the model.pt state_dict shapes
        if setting == "lora":
            lora_config = LoraConfig(
                r=8,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                ],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)

        # Hook up the rebuilt Audio and Projection modules
        self.audio_model = AudioModelWrapper()

        dim_embedding = (
            self.llm.config.hidden_size if hasattr(self.llm, "config") else 3072
        )
        self.prefix_project = TransformerMapper(
            dim_clip=1280,
            dim_embedding=dim_embedding,
            prefix_length=prefix_length,
            clip_length=clip_length,
            num_layers=8,
        )


# ==========================================
# 1. MODEL CODE END
# ==========================================


# ==========================================
# 1. LOAD MODEL (MODIFIED FOR LOCAL FILE & MAC)
# ==========================================
def load_careqa_model_local(
    local_model_path, llm_type="meta-llama/Llama-3.2-3B", prefix_length=8
):
    print("Initializing model architecture...")
    model = AudioQAModel(
        llm_type=llm_type,
        opera_checkpoint_path=None,
        prefix_length=prefix_length,
        clip_length=1,
        setting="lora",
        mapping_type="Transformer",
        fine_tune_opera=True,
        args=None,
    ).eval()

    # Automatically detect Apple Silicon (MPS), GPU, or CPU
    if False:
        device = torch.device("mps")
        print("Using Apple Silicon (MPS) for acceleration!")
    elif False:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    print(f"Loading weights from {local_model_path}...")
    state_dict = torch.load(local_model_path, map_location="cpu")

    # Extract nested state_dict if it exists
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully!\n")
    return model, device


# ==========================================
# 2. PREPROCESS AUDIO
# ==========================================
def preprocess_audio(audio_path, device, sr=16000):
    print(f"Processing audio file: {audio_path}")
    raw_audio, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        y=raw_audio, sr=sr, n_fft=1024, hop_length=512, n_mels=64
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Convert to tensor and move to correct device (MPS/CUDA/CPU)
    audio_tensor = (
        torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).to(device)
    )
    return audio_tensor


# ==========================================
# 3. GENERATE ANSWER
# ==========================================
def generate_answer(
    model,
    tokenizer,
    audio_tensor,
    question,
    device,
    prefix_length=8,
    audio_feature_dim=1280,
):
    # 1. Extract audio features
    with torch.no_grad():
        audio_features = model.audio_model.extract_feature(
            audio_tensor, dim=audio_feature_dim
        )
        projected_prefix = model.prefix_project(audio_features)

    # 2. Tokenize the text prompts
    q_prefix = tokenizer.encode("question: ", add_special_tokens=False)
    q_tokens = tokenizer.encode(question, add_special_tokens=False)
    a_prefix = tokenizer.encode(" answer", add_special_tokens=False)

    input_tokens = q_prefix + q_tokens + a_prefix

    # 3. Create input IDs
    input_ids = torch.tensor(
        [input_tokens + [tokenizer.eos_token_id] * prefix_length], dtype=torch.long
    ).to(device)
    attention_mask = torch.ones_like(input_ids)

    # 4. Insert the audio projection (FIXED FOR APPLE SILICON)
    # Adding .clone() prevents the Apple MPS silent deadlock!
    input_embeds = model.llm.get_input_embeddings()(input_ids).clone()
    input_embeds[
        0, len(q_prefix + q_tokens) : len(q_prefix + q_tokens) + prefix_length
    ] = projected_prefix[0]

    # 5. Generate the response
    print(">>> Firing up the LLaMA generation engine... (this should take < 2 mins)")
    with torch.no_grad():
        output_ids = model.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False,
            use_cache=False,  # FIXED: KV Cache can freeze when using custom inputs_embeds on Mac
            pad_token_id=tokenizer.eos_token_id,  # Silences the warning you got earlier
        )
    print(">>> Generation complete!")

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return answer


def get_tokenizer(model_dir: str):
    """Loads the LLaMA tokenizer from a local directory."""
    print("Loading local LLaMA Tokenizer...")
    return AutoTokenizer.from_pretrained(model_dir)


def get_base_llm(model_dir: str):
    """Loads the base LLaMA model with 4-bit quantization."""
    from transformers import AutoModelForCausalLM, QuantoConfig

    print("Loading Llama 3.2 3B in 4-bit mode...")
    quant_config = QuantoConfig(weights="int8")
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        # device_map="mps" # Uncomment to force Apple Silicon GPU allocation if supported
    )
    return llm_model


def get_careqa_model(careqa_path: str, llama_dir: str):
    """Loads the custom AudioQAModel and assigns it to the correct device."""
    # This calls your previously defined load_careqa_model_local method
    model, device = load_careqa_model_local(
        local_model_path=careqa_path, llm_type=llama_dir
    )
    return model, device


# ==========================================
# 4. GET ANSWER WRAPPER (Add this above __main__)
# ==========================================
def get_answer(question: str, audio_path: str, model, tokenizer, device) -> str:
    print("\n" + "-" * 50)
    print(f"Question: {question}")
    print("-" * 50)

    audio_tensor = preprocess_audio(audio_path, device)

    print("Generating answer...")
    answer = generate_answer(model, tokenizer, audio_tensor, question, device)

    print("-" * 50)
    print(f"Answer: {answer}")
    print("-" * 50)

    return answer


def print_stats(example_dataset: CaReSoundDataset):
    print("\n" + "=" * 50)
    print("DATASET STATS")
    print("=" * 50)
    example_dataset.stats()


def print_sample_i(sample_dataset: CaReSoundAQA, i: int):
    print("\n" + "=" * 50)
    print(f"Sample at index {i}")
    print("=" * 50)
    print(sample_dataset[i])
    print(sample_dataset[i]["audio_path"])


def get_raw_audio(sample_dataset: CaReSoundAQA, i, sr=16000):
    audio_path = sample_dataset[i]["audio_path"]

    # 2. Verify the file exists before trying to load it
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Could not find audio file at: {audio_path}")

    # 3. Load the raw audio
    # sr=16000 ensures it matches the sample rate used in medical audio tasks
    audio, sample_rate = librosa.load(audio_path, sr=sr)

    return audio, sample_rate


if __name__ == "__main__":

    example_dataset = CaReSoundDataset(root=path_to_dataset)
    sample_dataset = example_dataset.set_task(CaReSoundAQA())
    # testdataset()

    # test dataset
    print_stats(example_dataset)
    print_sample_i(sample_dataset, 10)
    get_raw_audio(sample_dataset, 10)

    if generateModelAnswer:
        # 2. Extract Data for Inference
        test_question = sample_dataset[10]["question"]
        target_audio_path = sample_dataset[10]["audio_path"]  # Renamed for clarity
        ground_truth = sample_dataset[10]["answer"]

        # 3. Load Models
        tokenizer = get_tokenizer(local_llama_dir)
        careqa_model, device = get_careqa_model(local_careqa_path, local_llama_dir)

        # 4. Run Inference (Fixed the variable name mismatch here)
        generated_answer = get_answer(
            question=test_question,
            audio_path=target_audio_path,
            model=careqa_model,
            tokenizer=tokenizer,
            device=device,
        )

        print(f"\nGround Truth was: {ground_truth}")
