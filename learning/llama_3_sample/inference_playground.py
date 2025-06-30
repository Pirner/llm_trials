import os
import urllib.request

import torch
from llms_from_scratch.llama3 import Llama3Model
from llms_from_scratch.llama3 import Llama3Tokenizer, ChatFormat, clean_text

import time

from llms_from_scratch.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)


def main():
    MODEL_FILE = "llama3.2-1B-instruct.pth"
    # MODEL_FILE = "llama3.2-1B-base.pth"
    # MODEL_FILE = "llama3.2-3B-instruct.pth"
    # MODEL_FILE = "llama3.2-3B-base.pth"

    # Text generation settings
    if "instruct" in MODEL_FILE:
        PROMPT = "What do llamas eat?"
        PROMPT = "Describe the items a dwarf would carry who is an engineer with a gun and a mallet."
    else:
        PROMPT = "Llamas eat"

    MAX_NEW_TOKENS = 500
    TEMPERATURE = 0.
    TOP_K = 1
    url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"

    if not os.path.exists(MODEL_FILE):
        urllib.request.urlretrieve(url, MODEL_FILE)
        print(f"Downloaded to {MODEL_FILE}")

    if "1B" in MODEL_FILE:
        from llms_from_scratch.llama3 import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
    elif "3B" in MODEL_FILE:
        from llms_from_scratch.llama3 import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
    else:
        raise ValueError("Incorrect model file name")

    model = Llama3Model(LLAMA32_CONFIG)
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location="cpu"))

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    model.to(device)
    TOKENIZER_FILE = "tokenizer.model"

    url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}"

    if not os.path.exists(TOKENIZER_FILE):
        urllib.request.urlretrieve(url, TOKENIZER_FILE)
        print(f"Downloaded to {TOKENIZER_FILE}")

    tokenizer = Llama3Tokenizer("tokenizer.model")

    if "instruct" in MODEL_FILE:
        tokenizer = ChatFormat(tokenizer)

    torch.manual_seed(123)

    start = time.time()

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(PROMPT, tokenizer).to(device),
        max_new_tokens=MAX_NEW_TOKENS,
        context_size=LLAMA32_CONFIG["context_length"],
        top_k=TOP_K,
        temperature=TEMPERATURE
    )

    total_time = time.time() - start
    print(f"Time: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0]) / total_time)} tokens/sec")

    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")

    output_text = token_ids_to_text(token_ids, tokenizer)

    if "instruct" in MODEL_FILE:
        output_text = clean_text(output_text)

    print("\n\nOutput text:\n\n", output_text)


if __name__ == '__main__':
    main()
