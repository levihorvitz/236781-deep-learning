import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F


def invalid_locations_mask(w, device):
    diagonals_list = []
    for j in range(-w, 1):
        diagonal_mask = torch.zeros(w, dtype=torch.uint8)
        diagonal_mask[:-j] = 1
        diagonals_list.append(diagonal_mask)
    mask = torch.stack(diagonals_list, dim=-1)
    mask = mask[None, :, None, :]
    ending_mask = mask.flip(dims=(1, 3)).bool().to(device)
    return mask.bool().to(device), ending_mask


def mask_invalid_locations(input_tensor: torch.Tensor, w: int):
    beginning_mask, ending_mask = invalid_locations_mask(w, input_tensor.device)
    sequence_length = input_tensor.size(1)
    beginning_input = input_tensor[:, :w, :, : w + 1]
    beginning_mask = beginning_mask[:, :sequence_length].expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask, -9e15)
    ending_input = input_tensor[:, -w:, :, -(w + 1) :]
    ending_mask = ending_mask[:, -sequence_length:].expand(ending_input.size())
    ending_input.masked_fill_(ending_mask, -9e15)


def skew(x, direction, padding_value):
    padded_x = nn.functional.pad(x, direction, value=padding_value)
    padded_x = padded_x.view(
        *padded_x.size()[:-2], padded_x.size(-1), padded_x.size(-2)
    )
    return padded_x


def main_diagonals_indices(b, n, w):
    diag_indices = torch.arange(-w, w + 1)
    row_indices = torch.arange(0, n * n, n + 1)
    col_indices = row_indices.view(1, -1, 1) + diag_indices
    col_indices = col_indices.repeat(b, 1, 1)
    return col_indices.flatten(1)[:, w:-w]


def populate_diags(x):
    bzs, seq_len, w = x.size()
    w = (w - 1) // 2
    x = x.flatten(1)[:, w:-w].float()
    result = torch.zeros(bzs, seq_len, seq_len, device=x.device).flatten(1)
    idx = main_diagonals_indices(bzs, seq_len, w).to(x.device)
    result = result.scatter_(1, idx, x).view(bzs, seq_len, seq_len)
    return result


def qk_sliding_blocks(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    bsz, num_heads, seqlen, head_dim = q.size()

    assert q.size() == k.size()

    chunks_count = seqlen // w - 1
    q = q.reshape(bsz * num_heads, seqlen, head_dim)
    k = k.reshape(bsz * num_heads, seqlen, head_dim)
    q_block = q.unfold(-2, 2 * w, w).transpose(-1, -2)
    k_block = k.unfold(-2, 2 * w, w).transpose(-1, -2)
    attn_block = torch.einsum("bcxd,bcyd->bcxy", (q_block, k_block))
    diagonal_block_attn = skew(
        attn_block, direction=(0, 0, 0, 1), padding_value=padding_value
    )
    diagonal_attn = torch.ones(
        (bsz * num_heads, chunks_count + 1, w, w * 2 + 1), device=attn_block.device
    ) * (-9e15)

    diagonal_attn[:, :-1, :, w:] = diagonal_block_attn[:, :, :w, : w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_block_attn[:, -1, w:, : w + 1]

    diagonal_attn[:, 1:, :, :w] = diagonal_block_attn[:, :, -(w + 1) : -1, w + 1 :]
    p = w > 1
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_block_attn[:, 0, : w - 1, p - w :]

    diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(
        2, 1
    )

    mask_invalid_locations(diagonal_attn, w)
    diagonal_attn = diagonal_attn.transpose(1, 2).view(
        bsz * num_heads, seqlen, 2 * w + 1
    )
    return diagonal_attn


def skew_prob(x, padding_value):
    B, C, M, L = x.size()
    x = F.pad(x, (0, M + 1), value=padding_value)
    x = x.view(B, C, -1)
    x = x[:, :, :-M]
    x = x.view(B, C, M, M + L)
    x = x[:, :, :, :-1]
    return x


def pv_sliding_blocks(prob: torch.Tensor, v: torch.Tensor, w: int):
    bsz, seqlen, num_heads, head_dim = v.size()
    assert seqlen % (w * 2) == 0
    assert prob.size()[:3] == v.size()[:3]
    assert prob.size(3) == 2 * w + 1
    chunks_count = seqlen // w - 1
    chunk_prob = prob.transpose(1, 2).reshape(
        bsz * num_heads, seqlen // w, w, 2 * w + 1
    )
    v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    padded_v = F.pad(v, (0, 0, w, w), value=-1)

    v_block_stride = padded_v.stride()
    v_block_stride = (
        v_block_stride[0],
        w * v_block_stride[1],
        v_block_stride[1],
        v_block_stride[2],
    )
    chunk_v = padded_v.as_strided(
        size=(bsz * num_heads, chunks_count + 1, 3 * w, head_dim), stride=v_block_stride
    )

    skewed_prob = skew_prob(chunk_prob, padding_value=0)

    context = torch.einsum("bcwd,bcdh->bcwh", (skewed_prob, chunk_v))
    return context.view(bsz, num_heads, seqlen, head_dim)


def qkv_window_size_padding(
    q, k, v, one_sided_window_size, padding_mask, paddin_value=0
):
    seq_len = q.shape[-2]
    w = int(2 * one_sided_window_size)
    padding_len = (w - seq_len % w) % w
    padding_l, padding_r = (padding_len // 2, padding_len // 2) if w > 2 else (0, 1)
    q = F.pad(q, (0, 0, padding_l, padding_r), value=paddin_value)
    k = F.pad(k, (0, 0, padding_l, padding_r), value=paddin_value)
    v = F.pad(v, (0, 0, padding_l, padding_r), value=paddin_value)
    if padding_mask is not None:
        padding_mask = F.pad(padding_mask, (padding_l, padding_r), value=0)
    return q, k, v, padding_mask
