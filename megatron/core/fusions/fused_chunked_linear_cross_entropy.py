# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Output layer + cross-entropy fused over vocabulary chunks, without the unfused path's copies.

The unfused path (``ColumnParallelLinear`` then ``vocab_parallel_cross_entropy``) materialises the
logits ``[tokens, vocab]``, casts them to an fp32 copy and keeps the fp32 softmax from the forward
to the backward. This op walks the vocabulary in chunks of ``vocab_chunk_size`` columns and keeps
two fp32 numbers per token, plus the logits of the first ``saved_logit_chunks`` chunks, from the
forward to the backward; any other chunk's logits are recomputed there.

Forward, per chunk: ``logits = hidden @ weight[chunk].T`` (one GEMM), then a Triton kernel
reduces each row to the chunk's maximum and sum of ``exp(logit - maximum)`` and picks the target
logit when the label falls in the chunk. The partials combine into the row maximum ``m`` and
``s = sum(exp(logit - m))``, and the loss is ``log(s) - (target_logit - m)``, the unfused formula.

Backward, per chunk: the chunk's logits (saved, or recomputed with one GEMM) are overwritten by a
Triton kernel with ``(exp(logit - m) / s - onehot) * grad_loss``, computed in fp32 and rounded
once to the logits dtype: the same operations, in the same order, as the unfused backward followed
by its cast from fp32. ``grad_hidden += grad_logits @ weight[chunk]`` accumulates in fp32 and is
rounded once at the end, and the chunk's rows of the weight gradient come from one GEMM,
``grad_logits.T @ hidden``, written to a gradient tensor or accumulated into ``weight.main_grad``
as ``ColumnParallelLinear`` does under gradient-accumulation fusion.

A recomputed chunk costs one more GEMM; a saved one costs ``tokens x chunk`` elements of memory
from the forward to the backward. The logits are the same either way, so ``saved_logit_chunks``
trades memory for time without changing any result. The upstream gradient is applied in the
backward, so any per-token ``grad_loss`` (masked tokens, any loss scaling) is handled exactly.

Only an unsharded vocabulary is supported (tensor-parallel size 1). Labels outside
``[0, vocab_size)`` follow the unfused convention: their target logit counts as the row maximum
and no one-hot term is subtracted.
"""

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from megatron.core.tensor_parallel.layers import (
    accumulate_wgrad_into_main_grad,
    wgrad_after_main_grad_accumulation,
)

# Elements per Triton block along a row of logits. Rows are reduced in blocks of this many
# columns; the gradient kernel writes one block per program.
_MAX_BLOCK_SIZE = 4096


@triton.jit
def _logsumexp_partials_kernel(
    logits_ptr,
    labels_ptr,
    chunk_max_ptr,
    chunk_sumexp_ptr,
    target_logit_ptr,
    num_cols,
    vocab_start,
    BLOCK_SIZE: tl.constexpr,
):
    """One row of a chunk: its maximum, its sum of exp(logit - maximum), and the target logit."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * num_cols
    offsets = tl.arange(0, BLOCK_SIZE)

    block_max = tl.full([BLOCK_SIZE], float("-inf"), tl.float32)
    for start in range(0, num_cols, BLOCK_SIZE):
        cols = start + offsets
        logits = tl.load(row_ptr + cols, mask=cols < num_cols, other=float("-inf"))
        block_max = tl.maximum(block_max, logits.to(tl.float32))
    row_max = tl.max(block_max, axis=0)

    block_sum = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in range(0, num_cols, BLOCK_SIZE):
        cols = start + offsets
        logits = tl.load(row_ptr + cols, mask=cols < num_cols, other=float("-inf"))
        block_sum += libdevice.exp(logits.to(tl.float32) - row_max)
    tl.store(chunk_max_ptr + row, row_max)
    tl.store(chunk_sumexp_ptr + row, tl.sum(block_sum, axis=0))

    local_label = tl.load(labels_ptr + row) - vocab_start
    if (local_label >= 0) & (local_label < num_cols):
        tl.store(target_logit_ptr + row, tl.load(row_ptr + local_label).to(tl.float32))


@triton.jit
def _cross_entropy_grad_kernel(
    logits_ptr,
    labels_ptr,
    row_max_ptr,
    sum_exp_ptr,
    grad_loss_ptr,
    num_cols,
    vocab_start,
    BLOCK_SIZE: tl.constexpr,
):
    """Overwrite one block of a row of logits with the loss gradient with respect to them."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < num_cols
    ptrs = logits_ptr + row * num_cols + cols

    logits = tl.load(ptrs, mask=mask, other=0.0)
    softmax = tl.div_rn(
        libdevice.exp(logits.to(tl.float32) - tl.load(row_max_ptr + row)),
        tl.load(sum_exp_ptr + row),
    )
    local_label = tl.load(labels_ptr + row) - vocab_start
    softmax = tl.where(cols == local_label, softmax - 1.0, softmax)
    grad = softmax * tl.load(grad_loss_ptr + row)
    tl.store(ptrs, grad.to(logits_ptr.dtype.element_ty), mask=mask)


def _vocab_chunks(vocab_size: int, vocab_chunk_size: int) -> List[Tuple[int, int]]:
    return [
        (start, min(start + vocab_chunk_size, vocab_size))
        for start in range(0, vocab_size, vocab_chunk_size)
    ]


def _block_size(num_cols: int) -> int:
    return min(_MAX_BLOCK_SIZE, triton.next_power_of_2(num_cols))


class _ChunkLogits:
    """Where each chunk's ``[tokens, chunk]`` logits live during one forward or backward.

    The first ``num_saved`` chunks occupy consecutive slices of one ``saved`` buffer, which the
    forward fills and hands to the backward; every other chunk reuses one scratch buffer.
    """

    def __init__(
        self,
        hidden: torch.Tensor,
        chunks: List[Tuple[int, int]],
        num_saved: int,
        saved: Optional[torch.Tensor],
    ):
        self.num_tokens = hidden.shape[0]
        self.num_saved = num_saved
        self.saved = saved
        if num_saved and saved is None:
            self.saved = hidden.new_empty(self.num_tokens * chunks[num_saved - 1][1])
        self.scratch = None
        if num_saved < len(chunks):
            self.scratch = hidden.new_empty(self.num_tokens * (chunks[0][1] - chunks[0][0]))

    def view(self, index: int, start: int, end: int) -> torch.Tensor:
        """The ``[tokens, end - start]`` logits of chunk ``index``."""
        if index < self.num_saved:
            flat = self.saved[self.num_tokens * start : self.num_tokens * end]
        else:
            flat = self.scratch[: self.num_tokens * (end - start)]
        return flat.view(self.num_tokens, end - start)


class ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Per-token cross-entropy of ``hidden @ weight.T`` against ``labels``; see the module doc."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        vocab_chunk_size: int,
        saved_logit_chunks: int,
        gradient_accumulation_fusion: bool,
    ) -> torch.Tensor:
        """Loss per token, fp32, for ``hidden [tokens, hidden_size]`` and ``labels [tokens]``."""
        num_tokens = hidden.shape[0]
        vocab_size = weight.shape[0]
        chunks = _vocab_chunks(vocab_size, vocab_chunk_size)
        num_saved = min(saved_logit_chunks, len(chunks))
        chunk_logits = _ChunkLogits(hidden, chunks, num_saved, saved=None)
        chunk_max = hidden.new_empty((len(chunks), num_tokens), dtype=torch.float32)
        chunk_sumexp = torch.empty_like(chunk_max)
        target_logit = hidden.new_zeros(num_tokens, dtype=torch.float32)

        for index, (start, end) in enumerate(chunks):
            logits = chunk_logits.view(index, start, end)
            torch.mm(hidden, weight[start:end].t(), out=logits)
            _logsumexp_partials_kernel[(num_tokens,)](
                logits,
                labels,
                chunk_max[index],
                chunk_sumexp[index],
                target_logit,
                end - start,
                start,
                BLOCK_SIZE=_block_size(end - start),
            )

        row_max = chunk_max.amax(dim=0)
        sum_exp = (chunk_sumexp * torch.exp(chunk_max - row_max)).sum(dim=0)
        label_in_vocab = (labels >= 0) & (labels < vocab_size)
        predicted_logit = torch.where(label_in_vocab, target_logit - row_max, 0.0)
        loss = torch.log(sum_exp) - predicted_logit

        # main_grad is kept on ctx rather than saved: as in ColumnParallelLinear, the same weight
        # may be applied more than once before the backward.
        ctx.main_grad = weight.main_grad if gradient_accumulation_fusion else None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.num_saved = num_saved
        ctx.save_for_backward(hidden, weight, labels, row_max, sum_exp, chunk_logits.saved)
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        """Gradients for ``hidden`` and ``weight``; see the module docstring."""
        hidden, weight, labels, row_max, sum_exp, saved = ctx.saved_tensors
        need_grad_hidden, need_grad_weight = ctx.needs_input_grad[:2]
        grad_loss = grad_loss.contiguous()
        num_tokens = hidden.shape[0]
        chunks = _vocab_chunks(weight.shape[0], ctx.vocab_chunk_size)
        chunk_logits = _ChunkLogits(hidden, chunks, ctx.num_saved, saved)

        grad_hidden = (
            hidden.new_empty(hidden.shape, dtype=torch.float32) if need_grad_hidden else None
        )
        grad_weight = None
        if need_grad_weight:
            if ctx.gradient_accumulation_fusion:
                weight.main_grad = ctx.main_grad
            else:
                grad_weight = torch.empty_like(weight)

        for index, (start, end) in enumerate(chunks):
            grad_logits = chunk_logits.view(index, start, end)
            if index >= ctx.num_saved:
                torch.mm(hidden, weight[start:end].t(), out=grad_logits)
            block_size = _block_size(end - start)
            _cross_entropy_grad_kernel[(num_tokens, triton.cdiv(end - start, block_size))](
                grad_logits,
                labels,
                row_max,
                sum_exp,
                grad_loss,
                end - start,
                start,
                BLOCK_SIZE=block_size,
            )
            if need_grad_hidden:
                torch.addmm(
                    grad_hidden,
                    grad_logits,
                    weight[start:end],
                    beta=0.0 if index == 0 else 1.0,
                    out_dtype=torch.float32,
                    out=grad_hidden,
                )
            if need_grad_weight:
                if ctx.gradient_accumulation_fusion:
                    accumulate_wgrad_into_main_grad(
                        hidden, grad_logits, weight.main_grad[start:end]
                    )
                else:
                    torch.mm(grad_logits.t(), hidden, out=grad_weight[start:end])

        if need_grad_weight and ctx.gradient_accumulation_fusion:
            grad_weight = wgrad_after_main_grad_accumulation(weight, hidden.dtype)
        if need_grad_hidden:
            grad_hidden = grad_hidden.to(hidden.dtype)
        return grad_hidden, grad_weight, None, None, None, None


def chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    vocab_chunk_size: int,
    saved_logit_chunks: int,
    gradient_accumulation_fusion: bool,
) -> torch.Tensor:
    """Per-token cross-entropy loss of the logits ``hidden @ weight.T``, fp32, shaped as ``labels``.

    Args:
        hidden: ``[..., hidden_size]`` output-layer input; its leading dimensions match ``labels``.
        weight: ``[vocab_size, hidden_size]`` output-layer weight (the whole vocabulary).
        labels: target token ids.
        vocab_chunk_size: vocabulary columns per chunk.
        saved_logit_chunks: how many chunks, from the start of the vocabulary, keep their logits
            from the forward for the backward; the others are recomputed there. 0 keeps none
            (least memory), and any value at least the number of chunks keeps all of them.
        gradient_accumulation_fusion: accumulate the weight gradient into ``weight.main_grad``
            (which must exist) instead of returning it, as ``ColumnParallelLinear`` does.
    """
    if vocab_chunk_size <= 0:
        raise ValueError(f"vocab_chunk_size must be positive, got {vocab_chunk_size}")
    if saved_logit_chunks < 0:
        raise ValueError(f"saved_logit_chunks must be non-negative, got {saved_logit_chunks}")
    if weight.dim() != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"hidden {tuple(hidden.shape)} and weight {tuple(weight.shape)} do not form a "
            "[..., hidden_size] x [vocab_size, hidden_size] product"
        )
    if hidden.shape[:-1] != labels.shape:
        raise ValueError(
            f"labels {tuple(labels.shape)} do not match the leading dimensions of hidden "
            f"{tuple(hidden.shape)}"
        )
    if hasattr(weight, "__fsdp_param__"):
        raise NotImplementedError(
            "The chunked linear cross-entropy does not support Megatron-FSDP parameters."
        )
    if gradient_accumulation_fusion and weight.requires_grad and not hasattr(weight, "main_grad"):
        raise RuntimeError(
            "gradient_accumulation_fusion accumulates into weight.main_grad, which this weight "
            "does not have (it is created by Megatron's DistributedDataParallel wrapper)."
        )
    loss = ChunkedLinearCrossEntropy.apply(
        hidden.reshape(-1, hidden.shape[-1]),
        weight,
        labels.reshape(-1),
        vocab_chunk_size,
        saved_logit_chunks,
        gradient_accumulation_fusion and weight.requires_grad,
    )
    return loss.view(labels.shape)
