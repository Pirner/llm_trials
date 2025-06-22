import torch
from torch import nn

from GELU import GELU
from config import GPT_CONFIG_124M


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    ffn = FeedForward(GPT_CONFIG_124M)

    # input shape: [batch_size, num_token, emb_size]
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print(out.shape)


if __name__ == '__main__':
    main()
