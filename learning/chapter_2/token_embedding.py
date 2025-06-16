import torch


def main():
    input_ids = torch.tensor([2, 3, 5, 1, 5, 9])
    vocab_size = 10
    output_dim = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    print(embedding_layer(torch.tensor([3])))
    print(embedding_layer(input_ids))


if __name__ == '__main__':
    main()
