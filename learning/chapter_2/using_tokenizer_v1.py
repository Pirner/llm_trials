from SimpleTokenizerV1 import SimpleTokenizerV1


def main():
    print('[INFO] start tokenize text')

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of character:", len(raw_text))
    tokenized_text = SimpleTokenizerV1.tokenize_text(raw_text)
    print("Total number of character after tokenization:", len(tokenized_text))
    all_words = sorted(set(tokenized_text))
    vocab_size = len(all_words)
    print('Size of vocabulary:', vocab_size)
    vocab = {token: integer for integer, token in enumerate(all_words)}
    tokenizer = SimpleTokenizerV1(vocab)

    text = """"It's the last he painted, you know," 
               Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))


if __name__ == '__main__':
    main()
