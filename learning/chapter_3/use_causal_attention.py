import torch
from CausalAttention import CausalAttention


def main():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )

    x_2 = inputs[1]  # second input element
    d_in = inputs.shape[1]  # the input embedding size, d=3
    d_out = 2  # the output embedding size, d=2

    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)  # 2 inputs with 6 tokens each, and each token has embedding dimension 3

    torch.manual_seed(123)

    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)

    context_vecs = ca(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)


if __name__ == '__main__':
    main()
