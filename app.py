import os
import time
import logging
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# HARD LOCK: NO DOWNLOADS
# ============================================================
#os.environ["HF_HUB_OFFLINE"] = "1"
#os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL_PATH = "Qwen/Qwen2-0.5B-Instruct"


QLORA_CHECKPOINT = "qwen-finance-qlora/checkpoint-1200"



DEVICE = torch.device("cpu")

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "You are a financial analysis assistant. "
    "Your task is to extract key financial signals from company disclosures "
    "and present them clearly and concisely."
    "stop generating after completing the answer."
    
)

# ============================================================
# APP
# ============================================================
app = FastAPI(title="Finance LLM Inference Service")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD TOKENIZER
# ============================================================
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)

# ============================================================
# LOAD BASE MODEL
# ============================================================
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "Qwen/Qwen2-0.5B-Instruct"
LORA_PATH = "./qwen-finance-qlora/checkpoint-1200"
OUTPUT_PATH = "./qwen-finance-merged"

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="cpu",
    local_files_only=True
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(
    base,
    "guptaakash134/finance-qlora-qwen",
    local_files_only=True
)

print("Merging LoRA weights...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(OUTPUT_PATH)

model.eval()

print("✅ Done. LoRA is now permanently merged.")





# ============================================================
# WARMUP
# ============================================================
logger.info("Warmup...")
with torch.no_grad():
    dummy = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "warmup"}
        ],
        tokenize=True,
        return_tensors="pt"
    ).to(DEVICE)
    _ = model.generate(dummy, max_new_tokens=1)

logger.info("Model ready.")

# ============================================================
# REQUEST SCHEMA
# ============================================================
class InferenceRequest(BaseModel):
    prompt: str

# ============================================================
# INFERENCE
# ============================================================
@app.post("/infer")
def infer(request: InferenceRequest):
    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Empty prompt")

        # -----------------------------
        # CHAT TEMPLATE (THIS WAS MISSING)
        # -----------------------------
        input_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.prompt.strip()}
            ],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        ).to(DEVICE)

        input_tokens = input_ids.shape[-1]

        # -----------------------------
        # PREFILL
        # -----------------------------
        t_prefill = time.time()
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=True)
        prefill_time = time.time() - t_prefill

        # -----------------------------
        # DECODE
        # -----------------------------
        t_decode = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True
            )
        decode_time = time.time() - t_decode

        total_time = prefill_time + decode_time

        generated = output_ids[0][input_tokens:]
        output_text = tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

        return {
            "output": output_text,
            "latency_seconds": round(total_time, 4),
            "prefill_seconds": round(prefill_time, 4),
            "decode_seconds": round(decode_time, 4),
            "input_tokens": input_tokens,
            "total_tokens": output_ids.shape[-1]
        }

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))



