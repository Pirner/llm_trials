import torch
from GPTDatasetV1 import GPTDatasetV1, create_dataloader_v1


def main():
    vocab_size = 50257
    output_dim = 256

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    token_embeddings = token_embedding_layer(inputs)
    print("token embeddings:", token_embeddings.shape)
    # uncomment & execute the following line to see how the embeddings look like
    # print(token_embeddings)
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    # uncomment & execute the following line to see how the embedding layer weights look like
    print("pos layer embeddings:", pos_embedding_layer.weight.shape)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print("pos embeddings:", pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print("final input embeddings:", input_embeddings.shape)


if __name__ == '__main__':
    main()
