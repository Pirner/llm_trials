# create the masked attention weights to only process information from previous tokens
import torch
from self_attention_v2 import SelfAttention_v2


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

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)

    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T

    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    print(attn_weights)
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)
    masked_simple = attn_weights * mask_simple
    print(masked_simple)
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print(masked_simple_norm)

    # more efficient approach
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)
    attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
    print(attn_weights)

    print('[INFO] dropout')
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)  # dropout rate of 50%
    torch.manual_seed(123)
    print(dropout(attn_weights))


if __name__ == '__main__':
    main()
