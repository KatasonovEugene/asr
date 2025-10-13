import math
import torch
from torch import nn


class ConvolutionSubsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_sub = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
        )

    def forward(self, x): # (bs, time, n_mels)
        x = self.conv_sub(x.unsqueeze(1)) # (bs, out_chan, time, n_mels)
        bs, _, time, _ = x.shape
        x = x.contiguous().permute(0, 2, 1, 3) # (bs, time, out_chan, n_mels)
        x = x.reshape(bs, time, -1) # (bs, time, out_chan * n_mels)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor, dropout):
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.ff(x)


class ConvolutionBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size, dropout):
        super().__init__()
        self.lnorm = nn.LayerNorm(embed_dim)
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=embed_dim, 
                out_channels=embed_dim * 2,
                kernel_size=1,
            ),
            nn.GLU(dim=-2),
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                groups=embed_dim,
                kernel_size=kernel_size,
                padding='same',
            ),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
            nn.Conv1d(
                in_channels=embed_dim, 
                out_channels=embed_dim,
                kernel_size=1,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        _x = x
        x = self.lnorm(x)
        x = self.conv_block(x.transpose(1, 2)).transpose(1, 2)
        return _x + x


class RelativePositionEncoding(nn.Module):
    def __init__(self, num_heads, max_rel_pos):
        super().__init__()
        self.num_heads = num_heads
        self.max_rel_pos = max_rel_pos
        self.emb = nn.Embedding(2 * max_rel_pos + 1, num_heads)

    def forward(self, seq_len, device):
        idxs = torch.arange(seq_len, device=device)
        rel_pos = idxs[None, :] - idxs[:, None]
        rel_pos_clipped = rel_pos.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        embs = self.emb(rel_pos_clipped).permute(2, 0, 1)
        return embs


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, max_rel_pos):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.lnorm = nn.LayerNorm(embed_dim)

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.rel_pos_bias = RelativePositionEncoding(num_heads, max_rel_pos)

    def _split_heads(self, x):
        b, t, _ = x.size()
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        b, t, _, _ = x.size()
        return x.view(b, t, self.embed_dim)

    def forward(self, x, padding_mask):
        _x = x
        x = self.lnorm(x)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        bias = self.rel_pos_bias(x.size(1), device=x.device).unsqueeze(0)
        attn_logits = attn_logits + bias

        key_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        attn_logits = attn_logits.masked_fill(~key_mask, float('-inf'))

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = self._merge_heads(attn_output)

        out = self.out_proj(attn_output)
        out = self.proj_dropout(out)
        return _x + out


class ConformerBlock(nn.Module):
    def __init__(self, ff_params, msha_params, conv_params):
        super().__init__()
        embed_dim = msha_params['embed_dim']
        self.ff1 = FeedForwardBlock(**ff_params, embed_dim=embed_dim)
        self.msha = MultiHeadSelfAttentionBlock(**msha_params)
        self.convbl = ConvolutionBlock(**conv_params, embed_dim=embed_dim)
        self.ff2 = FeedForwardBlock(**ff_params, embed_dim=embed_dim)
        self.lnorm = nn.LayerNorm(embed_dim)

    def forward(self, x, padding_mask):
        x = x + self.ff1(x) * 0.5
        x = x + self.msha(x, padding_mask)
        x = x + self.convbl(x)
        x = x + self.ff2(x) * 0.5
        x = self.lnorm(x)
        return x


class Conformer(nn.Module):
    def __init__(
            self,
            num_blocks,
            conv_sub_params,
            conformer_block_params,
            dropout,
            n_mels,
            n_tokens,
        ):
        super().__init__()
        embed_dim = conformer_block_params['msha_params']['embed_dim']
        lin_input_dim = conv_sub_params['out_channels'] * ((n_mels + 3) // 4)
        self.preprocess = nn.Sequential(
            ConvolutionSubsampling(**conv_sub_params),
            nn.Linear(lin_input_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(**conformer_block_params) for _ in range(num_blocks)
        ])
        self.fc = nn.Linear(embed_dim, n_tokens)
 
    def transform_lengths(self, x_lengths):
        return (x_lengths + 3) // 4

    def forward(self, spectrogram, spectrogram_length, **batch):
        log_probs_length = self.transform_lengths(spectrogram_length)
        conf_emb = self.preprocess(spectrogram)
        padding_mask = self._create_padding_mask(log_probs_length)
        for conformer_block in self.conformer_blocks:
            conf_emb = conformer_block(conf_emb, padding_mask)
        logits = self.fc(conf_emb)
        log_probs = logits.log_softmax(dim=-1)
        return dict(log_probs=log_probs, log_probs_length=log_probs_length)

    def _create_padding_mask(self, spectrogram_length):
        max_length = spectrogram_length.max().item()
        return spectrogram_length.unsqueeze(1) > torch.arange(0, max_length, device=spectrogram_length.device).unsqueeze(0)
