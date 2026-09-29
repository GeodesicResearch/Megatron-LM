# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Output layer that can return the language-model loss in place of the logits."""

from typing import Optional

import torch

from megatron.core.fusions.fused_chunked_linear_cross_entropy import chunked_linear_cross_entropy
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.utils import get_pg_size


class LinearCrossEntropyModule(ColumnParallelLinear):
    """``ColumnParallelLinear`` whose forward can fuse the cross-entropy loss.

    Built by models whose config selects ``cross_entropy_loss_fusion`` with
    ``cross_entropy_fusion_impl='linear'``. Without ``labels`` it behaves exactly as
    ``ColumnParallelLinear`` (same parameters, same logits). With ``labels`` it returns the
    per-token loss ``[batch, sequence]`` that ``LanguageModule.compute_language_model_loss``
    returns for these logits, computed by ``chunked_linear_cross_entropy`` over vocabulary chunks
    (``cross_entropy_fusion_vocab_chunk_size``, ``cross_entropy_fusion_saved_logit_chunks``).
    The loss is computed inside ``forward`` so that module hooks, such as the distributed
    optimizer's wait for this layer's parameter all-gather, run as they do for the logits.

    The loss path supports only an unsharded vocabulary (tensor-parallel size 1), without bias,
    deferred embedding weight gradients or CPU activation offloading, and raises otherwise. The
    logits path has no such limits, so a model trained with the fusion still loads anywhere, e.g.
    for checkpoint conversion at another tensor-parallel size.
    """

    def _check_loss_supported(self) -> None:
        unsupported = {
            "tensor-parallel size > 1": get_pg_size(self.tp_group) > 1,
            "a bias": self.bias is not None,
            "defer_embedding_wgrad_compute": self.config.defer_embedding_wgrad_compute,
            "cpu_offloading": self.config.cpu_offloading,
        }
        refused = [name for name, present in unsupported.items() if present]
        if refused:
            raise ValueError(
                "cross_entropy_fusion_impl='linear' does not support " + ", ".join(refused) + "."
            )

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Logits and bias as ``ColumnParallelLinear``, or the loss when ``labels`` is given.

        Args:
            input_: ``[sequence, batch, hidden]`` output-layer input.
            weight: weight to use instead of this layer's own (tied embeddings).
            runtime_gather_output: as in ``ColumnParallelLinear``; unused for the loss.
            labels: ``[batch, sequence]`` target token ids. When given, the return value is the
                fp32 per-token loss ``[batch, sequence]`` instead of ``(logits, bias)``.
        """
        if labels is None:
            return super().forward(input_, weight, runtime_gather_output)
        self._check_loss_supported()
        loss = chunked_linear_cross_entropy(
            input_,
            self._resolve_weight(weight),
            labels.transpose(0, 1),
            self.config.cross_entropy_fusion_vocab_chunk_size,
            self.config.cross_entropy_fusion_saved_logit_chunks,
            self.gradient_accumulation_fusion,
        )
        # [s b] => [b, s]
        return loss.transpose(0, 1).contiguous()
