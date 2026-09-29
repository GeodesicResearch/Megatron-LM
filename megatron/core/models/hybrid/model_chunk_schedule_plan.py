# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Schedule-plan classes for HybridStack-based decoders.

These extend the GPT-side ``TransformerLayerSchedulePlan`` /
``TransformerModelChunkSchedulePlan`` with the per-layer ``layer_type`` symbol
that HybridStack assigns to each entry of its ``layer_type_list`` (including
bracketed groups like ``[*-]``). The base classes remain GPT-only; this module
adds the hybrid-specific dispatch into ``build_hybrid_stack_callables`` and
uses ``HybridStackNode`` so the schedule node's free-input policy can diverge
from the GPT default. The pre/post-process nodes from
``core.models.common.utils`` are reused as-is — they already call
``model._preprocess`` / ``model._postprocess`` which work on a HybridModel.
"""

import os
from contextlib import nullcontext

from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerLayerSchedulePlan,
    TransformerModelChunkSchedulePlan,
)

AUTOGROUP_ENV = "MCORE_HYBRID_OVERLAP_AUTOGROUP"


def flat_pattern_autogroup_enabled():
    """Whether a flat pattern is scheduled in ``[pre-layers..., MLP/MoE]`` units.

    [Geodesic adaptation, not in upstream #4798.] On unless ``MCORE_HYBRID_OVERLAP_AUTOGROUP``
    is ``0``, which restores one plan per layer symbol for A/B comparison. ``1`` also means on;
    any other value raises instead of being read as either.
    """
    value = os.environ.get(AUTOGROUP_ENV, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{AUTOGROUP_ENV} must be '0' or '1', got {value!r}")
    return value == "1"


def check_hybrid_overlap_supported(config) -> None:
    """Raise ``ValueError`` for settings the hybrid combined-1F1B schedule gets wrong.

    [Geodesic adaptation, not in upstream #4798.] Each refused setting has a known defect on
    the hybrid path:

    - the flex dispatcher's ``ncclep`` backend: the hybrid dispatch and expert nodes hand
      detached probs and hold ``tokens_per_expert`` only for ``deepep`` / ``hybridep``;
    - ``fine_grained_activation_offloading``: the hybrid combine node does not offload the
      ``mlp_norm`` output the way the GPT node does;
    - ``delay_wgrad_compute``: the hybrid pre-dispatch node schedules the MoE layer's
      pre-dispatch weight gradients only when it has shared experts, so a latent MoE layer
      without them never computes ``fc1_latent_proj``'s weight gradient.
    """
    if (
        config.moe_token_dispatcher_type == "flex"
        and config.moe_flex_dispatcher_backend == "ncclep"
    ):
        raise ValueError(
            "The hybrid EP overlap does not support the flex dispatcher's ncclep backend; "
            "use hybridep or deepep, or turn off overlap_moe_expert_parallel_comm."
        )
    if config.fine_grained_activation_offloading:
        raise ValueError(
            "The hybrid EP overlap does not support fine_grained_activation_offloading; "
            "turn off one of the two."
        )
    if config.delay_wgrad_compute:
        raise ValueError("The hybrid EP overlap does not support delay_wgrad_compute; turn it off.")


def group_flat_pattern(layer_type_list):
    """Split a flat layer pattern into runs of layer indices that each end at an MLP/MoE layer.

    [Geodesic adaptation, not in upstream #4798.] Layers after the last MLP/MoE form a final
    run of their own.
    """
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    runs, current = [], []
    for idx, layer_type in enumerate(layer_type_list):
        current.append(idx)
        if layer_type in (Symbols.MLP, Symbols.MOE):
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


class HybridStackSchedulePlan(TransformerLayerSchedulePlan):
    """Per-layer schedule plan for HybridStack decoders.

    Adds the ``layer_type`` extra-arg propagation; routes through
    ``build_hybrid_stack_callables`` when ``layer_type`` is set (i.e. the layer
    is a HybridStack entry, possibly a bracketed group); falls back to the GPT
    path for plain TransformerLayer / MTP layers when ``layer_type`` is None.
    """

    def __init__(self, layer, event, chunk_state, comp_stream, comm_stream, extra_args=None):
        if extra_args is None:
            extra_args = {}
        self.layer_type = extra_args.get("layer_type", None)
        super().__init__(layer, event, chunk_state, comp_stream, comm_stream, extra_args)

    def _build_callable_nodes(self, event, comp_stream, comm_stream, extra_args):
        if self.layer_type is None:
            return super()._build_callable_nodes(event, comp_stream, comm_stream, extra_args)

        # Hybrid grouped path. Imports are local because hybrid pulls in TE / SSM
        # extensions that we don't want to load when only the GPT path is used.
        from megatron.core.models.hybrid.fine_grained_callables import (
            HybridStackNode,
            build_hybrid_stack_callables,
        )
        from megatron.core.pipeline_parallel.utils import NoopScheduleNode

        fwd_callables, bwd_dw_callable_map, is_moe, num_local_experts, experts_quantize_input = (
            build_hybrid_stack_callables(self.layer, layer_type=self.layer_type)
        )

        extra_args["config"] = self.layer.config
        extra_args["is_moe"] = is_moe
        extra_args["num_local_experts"] = num_local_experts
        extra_args["delay_wgrad_compute"] = self.layer.config.delay_wgrad_compute
        extra_args["is_mtp"] = False
        extra_args["experts_quantize_input"] = experts_quantize_input

        def create_node(stream, module, name):
            bwd_dw_callables = bwd_dw_callable_map.get(name, None)
            node_extra_args = dict(extra_args)
            if bwd_dw_callables is None:
                node_extra_args["delay_wgrad_compute"] = False
            return HybridStackNode(
                stream,
                event,
                self.layer_state,
                self.chunk_state,
                module,
                name=name,
                bwd_dw_callables=bwd_dw_callables,
                extra_args=node_extra_args,
            )

        (
            pre_dispatch_module,
            moe_dispatch_module,
            mlp_module,
            moe_combine_module,
            mtp_post_process_module,
        ) = fwd_callables

        self.pre_dispatch_computation = create_node(
            comp_stream, pre_dispatch_module, "pre_dispatch_computation"
        )
        self.mlp = create_node(comp_stream, mlp_module, "mlp")
        if is_moe:
            self.moe_dispatch = create_node(comm_stream, moe_dispatch_module, "moe_dispatch")
            self.moe_combine = create_node(comm_stream, moe_combine_module, "moe_combine")
        else:
            self.moe_dispatch = NoopScheduleNode()
            self.moe_combine = NoopScheduleNode()

        # HybridStack groups never carry an MTP terminal, so mtp_post_process is
        # always a no-op here.
        self.mtp_post_process = NoopScheduleNode()

    def get_fp8_context(self):
        """Return an FP8 context only for plain transformer layers."""
        # Grouped hybrid layers (and inferred-layer-type entries that point at
        # a HybridStack rather than a plain TransformerLayer) don't have a
        # ``layer_number`` we can hand to ``get_fp8_context``; the inner layers
        # manage their own per-layer fp8 context inside the hybrid callables.
        if self.layer_type is not None or not hasattr(self.layer, "layer_number"):
            return nullcontext()
        return super().get_fp8_context()


class HybridStackModelChunkSchedulePlan(TransformerModelChunkSchedulePlan):
    """Model-chunk schedule plan that builds ``HybridStackSchedulePlan`` layer plans.

    Threads HybridStack's ``layer_type_list[layer_idx]`` symbol into each
    layer plan's ``extra_args`` so the per-layer plan can dispatch grouped
    layers correctly. Ordinary GPT/MTP layers (no ``layer_type_list``)
    default to ``layer_type=None`` and follow the GPT path. The pre/post
    process nodes inherit from the GPT base class — they already dispatch
    on ``model._preprocess`` / ``model._postprocess`` which a HybridModel
    implements.
    """

    LAYER_SCHEDULE_PLAN_CLASS = HybridStackSchedulePlan

    def __init__(self, model, *args, **kwargs):
        """Initialize the hybrid chunk plan after validating the model's settings."""
        check_hybrid_overlap_supported(model.config)
        assert model.config.cuda_graph_impl == "none", (
            "EP A2A overlap with grouped HybridStack patterns (e.g. '[*E]') does not "
            "support cuda graphs yet. Set cuda_graph_impl='none' or use an ungrouped pattern."
        )
        super().__init__(model, *args, **kwargs)

    def _extra_args_for_layer(self, module, layer_idx, num_layers):
        extra_args = super()._extra_args_for_layer(module, layer_idx, num_layers)
        extra_args["layer_type"] = (
            module.layer_type_list[layer_idx] if hasattr(module, "layer_type_list") else None
        )
        return extra_args

    def _build_layer_schedule_plan(self, module, comp_stream, comm_stream):
        """Schedule a flat pattern's ``[pre-layers..., MLP/MoE]`` runs as single units.

        [Geodesic adaptation, not in upstream #4798.] With one plan per symbol, pairing a
        forward MoE layer with a backward Mamba layer (and vice versa) leaves one of the two
        all-to-alls of each pair with no compute to hide behind. Scheduling each run as one
        unit gives every dispatch/combine a Mamba/attention or expert-GEMM partner, which is
        what bracketing the pattern (``[ME][M*E]...``) does upstream, but without building
        nested stacks, so parameter names and checkpoint keys stay those of the flat model.
        Patterns that already bracket groups, and modules without a ``layer_type_list``
        (MTP), keep the upstream behaviour, as does ``MCORE_HYBRID_OVERLAP_AUTOGROUP=0`` (see
        ``flat_pattern_autogroup_enabled``).
        """
        from megatron.core.models.hybrid.fine_grained_callables import HybridLayerRun
        from megatron.core.models.hybrid.hybrid_layer_allocation import is_layer_group

        layer_type_list = getattr(module, "layer_type_list", None) if module is not None else None
        if (
            layer_type_list is None
            or not flat_pattern_autogroup_enabled()
            or any(is_layer_group(layer_type) for layer_type in layer_type_list)
        ):
            return super()._build_layer_schedule_plan(module, comp_stream, comm_stream)

        runs = group_flat_pattern(layer_type_list)
        for run_idx, run in enumerate(runs):
            layer_run = HybridLayerRun(
                [layer_type_list[i] for i in run], [module.layers[i] for i in run]
            )
            extra_args = {
                "is_first_layer": run_idx == 0,
                "is_last_layer": run_idx == len(runs) - 1,
                "layer_type": layer_run.layer_type_list,
            }
            self._transformer_layers.append(
                self.LAYER_SCHEDULE_PLAN_CLASS(
                    layer_run, self.event, self.state, comp_stream, comm_stream, extra_args
                )
            )
