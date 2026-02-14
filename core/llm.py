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
    # NOTE:
    # Running Phi-3 on Apple Silicon MPS can trigger shape / matmul errors.
    # To avoid these, we force the model onto CPU with float32.
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float32,
    )
    model = model.to("cpu")
    model.eval()
    return model


def generate_answer(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Generate an answer from the LLM given a prompt.

    Preserves your original generation settings.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("Input tokens:", inputs["input_ids"].shape[1])

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Match your original pattern: answer is after "Answer:"
    if "Answer:" in response:
        return response.split("Answer:", 1)[-1].strip()

    return response.strip()