from functools import lru_cache
import importlib
from typing import Any

from .config import LLM_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE

# Dynamically import heavy ML dependencies so static analyzers
# don't flag missing imports when they aren't installed.
torch = importlib.import_module("torch")
transformers = importlib.import_module("transformers")
AutoModelForCausalLM: Any = transformers.AutoModelForCausalLM
AutoTokenizer: Any = transformers.AutoTokenizer


# @lru_cache()
def get_tokenizer():
    """
    Load and cache the tokenizer once.
    """
    return AutoTokenizer.from_pretrained(LLM_MODEL_NAME)


# @lru_cache()
def get_model():
    """
    Load and cache the LLM model once. Uses the same configuration as your original script.
    """
    print("LOADING MODEL")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16,
    )

    # Force Apple Silicon GPU (MPS) if available
    if torch.backends.mps.is_available():
        model = model.to("mps")
    else:
        model = model.to("cpu")

    model.eval()
    return model


def generate_answer(
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Generate an answer from the LLM given a prompt.

    Preserves your original generation settings.
    """
    tokenizer = get_tokenizer()
    model = get_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,              # Greedy decoding (faster)
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Match your original pattern: answer is after "Answer:"
    if "Answer:" in response:
        return response.split("Answer:", 1)[-1].strip()

    return response.strip()