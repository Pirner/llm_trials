import re


def tokenize_text(text: str):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed


def main():
    print('[INFO] start tokenize text')

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of character:", len(raw_text))
    tokenized_text = tokenize_text(raw_text)
    print("Total number of character after tokenization:", len(tokenized_text))
    all_words = sorted(set(tokenized_text))
    vocab_size = len(all_words)
    print('Size of vocabulary:', vocab_size)
    vocab = {token:integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break

if __name__ == '__main__':
    main()
