import torch


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


def main():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )

    query = inputs[1]  # 2nd input token is the query

    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)  # dot product (transpose not necessary here since they are 1-dim vectors)

    print(attn_scores_2)

    res = 0.

    for idx, element in enumerate(inputs[0]):
        res += inputs[0][idx] * query[idx]

    print(res)
    print(torch.dot(inputs[0], query))

    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())

    # compute the context vector z
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        tmp_weights = attn_weights_2[i]
        tmp = attn_weights_2[i] * x_i
        context_vec_2 += attn_weights_2[i] * x_i
    print(context_vec_2)

    # now compute if for everything
    # (1) attention scores
    attention_scores = torch.zeros((6, 6))
    for i, x_i in enumerate(inputs):
        attn_scores_i = torch.empty(inputs.shape[0])
        for j, q_j in enumerate(inputs):
            attn_scores_i[j] = torch.dot(x_i, q_j)
        attention_scores[i] = attn_scores_i.clone()

    # (2) attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)
    # (3) context vector
    context_vector = attention_weights @ inputs
    print(context_vector)
    print(attention_weights.shape)
    print(inputs.shape)
    print(context_vector.shape)


if __name__ == '__main__':
    main()
