import torch
from torch import nn

from learning.chapter_4.LayerNorm import LayerNorm


def main():
    torch.manual_seed(123)

    # create 2 training examples with 5 dimensions (features) each
    batch_example = torch.randn(2, 5)

    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)

    print("Mean:\n", mean)
    print("Variance:\n", var)

    out_norm = (out - mean) / torch.sqrt(var)
    print("Normalized layer outputs:\n", out_norm)

    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    torch.set_printoptions(sci_mode=False)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

    print("Mean:\n", mean)
    print("Variance:\n", var)


if __name__ == '__main__':
    main()
