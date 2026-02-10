from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LLM_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE


@lru_cache()
def get_tokenizer():
    """
    Load and cache the tokenizer once.
    """
    return AutoTokenizer.from_pretrained(LLM_MODEL_NAME)


@lru_cache()
def get_model():
    """
    Load and cache the LLM model once. Uses the same configuration as your original script.
    """
    return AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )


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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Match your original pattern: answer is after "Answer:"
    if "Answer:" in response:
        return response.split("Answer:", 1)[-1].strip()

    return response.strip()