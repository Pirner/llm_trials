from SimpleTokenizerV2 import SimpleTokenizerV2


def main():
    print('[INFO] start tokenize text')

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of character:", len(raw_text))
    tokenized_text = SimpleTokenizerV2.tokenize_text(raw_text)
    print("Total number of character after tokenization:", len(tokenized_text))
    all_words = sorted(set(tokenized_text))
    vocab_size = len(all_words)
    print('Size of vocabulary:', vocab_size)
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_words)}
    tokenizer = SimpleTokenizerV2(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."

    text = " <|endoftext|> ".join((text1, text2))

    print(text)
    tokenized = tokenizer.encode(text)
    print('tokenized:', tokenized)
    decoded = tokenizer.decode(tokenizer.encode(text))
    print('decoded:', decoded)


if __name__ == '__main__':
    main()
