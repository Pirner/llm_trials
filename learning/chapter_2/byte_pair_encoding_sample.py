from importlib.metadata import version
import tiktoken


def main():
    print("tiktoken version:", version("tiktoken"))
    tokenizer = tiktoken.get_encoding('gpt2')
    text = (
        "Akwir ier"
    )

    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    strings = tokenizer.decode(integers)

    print(strings)


if __name__ == '__main__':
    main()
