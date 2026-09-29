# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""EP overlap (the combined-1F1B schedule) on hybrid models.

Covered here:

- Flat-pattern grouping. The hybrid schedule plan runs a flat pattern's
  ``[pre-layers..., MLP/MoE]`` runs as single units (``group_flat_pattern``), as a bracketed
  pattern would, without building nested stacks, so parameter names and checkpoint keys stay
  those of the flat model. ``MCORE_HYBRID_OVERLAP_AUTOGROUP=0`` restores one unit per layer.
- The settings the hybrid schedule refuses (``check_hybrid_overlap_supported``).
- The overlap schedule reproduces the eager forward and backward bit for bit, for flat, bracketed
  and MTP patterns, with and without shared experts.
"""

import gc

import pytest
import torch

from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from megatron.core.models.hybrid.fine_grained_callables import HybridLayerRun
from megatron.core.models.hybrid.model_chunk_schedule_plan import (
    AUTOGROUP_ENV,
    check_hybrid_overlap_supported,
    flat_pattern_autogroup_enabled,
    group_flat_pattern,
)
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.ssm.mamba_mixer import HAVE_MAMBA_SSM
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.utils import is_te_min_version, is_torch_min_version
from tests.unit_tests.a2a_overlap.utils import (
    build_hybrid_config,
    build_hybrid_model,
    build_input_data,
    compare_captures,
    deterministic_mode,
    get_valid_flex_dispatcher_backend,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils

try:
    import causal_conv1d  # noqa: F401

    HAVE_CAUSAL_CONV1D = True
except ImportError:
    HAVE_CAUSAL_CONV1D = False

SEQ_LEN = 32
VOCAB_SIZE = 128
# Nemotron-3 Nano 30B-A3B: 52 layers, 23 of them MoE.
NANO_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"


class TestGroupFlatPattern:
    """``group_flat_pattern`` splits a flat pattern into runs that end at an MLP/MoE layer."""

    def test_nano_pattern_gives_one_unit_per_moe_layer(self):
        runs = group_flat_pattern(list(NANO_PATTERN))
        assert len(runs) == 23
        assert [i for run in runs for i in run] == list(range(len(NANO_PATTERN)))
        symbols = ["".join(NANO_PATTERN[i] for i in run) for run in runs]
        assert set(symbols) == {"ME", "M*E"}
        assert symbols[:3] == ["ME", "ME", "M*E"]

    def test_layers_after_the_last_mlp_or_moe_form_a_final_unit(self):
        assert group_flat_pattern(list("MEM*")) == [[0, 1], [2, 3]]

    def test_dense_mlp_layers_end_a_unit_too(self):
        assert group_flat_pattern(list("M-*E")) == [[0, 1], [2, 3]]


class TestAutogroupSwitch:
    """``MCORE_HYBRID_OVERLAP_AUTOGROUP`` is on when unset or ``1``, off at ``0``, else an error."""

    def test_unset_means_on(self, monkeypatch):
        monkeypatch.delenv(AUTOGROUP_ENV, raising=False)
        assert flat_pattern_autogroup_enabled()

    @pytest.mark.parametrize("value,expected", [("1", True), ("0", False)])
    def test_zero_and_one(self, monkeypatch, value, expected):
        monkeypatch.setenv(AUTOGROUP_ENV, value)
        assert flat_pattern_autogroup_enabled() is expected

    @pytest.mark.parametrize("value", ["false", "off", "no", "", "2"])
    def test_any_other_value_raises(self, monkeypatch, value):
        monkeypatch.setenv(AUTOGROUP_ENV, value)
        with pytest.raises(ValueError, match=AUTOGROUP_ENV):
            flat_pattern_autogroup_enabled()


def _overlap_config(**overrides):
    """A MoE config with the EP overlap on, flex/HybridEP, plus ``overrides``."""
    kwargs = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        num_moe_experts=8,
        expert_model_parallel_size=4,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        overlap_moe_expert_parallel_comm=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestCheckHybridOverlapSupported:
    """The hybrid schedule refuses settings its callables get wrong, and nothing else."""

    @pytest.mark.parametrize(
        "overrides",
        [{}, {"moe_flex_dispatcher_backend": "deepep"}, {"moe_token_dispatcher_type": "alltoall"}],
        ids=["hybridep", "deepep", "alltoall"],
    )
    def test_supported_settings_pass(self, overrides):
        check_hybrid_overlap_supported(_overlap_config(**overrides))

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"moe_flex_dispatcher_backend": "ncclep"}, "ncclep"),
            (
                {"fine_grained_activation_offloading": True, "offload_modules": ["mlp_norm"]},
                "fine_grained_activation_offloading",
            ),
            ({"delay_wgrad_compute": True}, "delay_wgrad_compute"),
        ],
        ids=["ncclep", "fine-grained-offloading", "delay-wgrad-compute"],
    )
    def test_unsupported_settings_raise(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            check_hybrid_overlap_supported(_overlap_config(**overrides))


def _skip_without_mamba(pattern):
    if "M" in pattern and not (HAVE_MAMBA_SSM and HAVE_CAUSAL_CONV1D):
        pytest.skip("Mamba patterns require both mamba-ssm and causal-conv1d.")


def _flex_kwargs():
    backend = get_valid_flex_dispatcher_backend()
    if backend is None:
        pytest.skip("No flex dispatcher backend available")
    return {"moe_token_dispatcher_type": "flex", "moe_flex_dispatcher_backend": backend}


@pytest.mark.flaky_in_dev
@pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
@pytest.mark.skipif(not is_torch_min_version("2.6.0"), reason="EP overlap hangs on torch < 2.6.0")
class TestHybridOverlap:
    """A HybridModel under the overlap schedule."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("pattern", ["*E*E", "M*EME-"])
    @pytest.mark.parametrize("autogroup", ["1", "0"])
    def test_plan_units(self, monkeypatch, pattern, autogroup):
        _skip_without_mamba(pattern)
        monkeypatch.setenv(AUTOGROUP_ENV, autogroup)
        config = build_hybrid_config(pattern, extra_kwargs=_flex_kwargs())
        with deterministic_mode():
            model = build_hybrid_model(config, pattern, VOCAB_SIZE, SEQ_LEN)
            plan = model.build_schedule_plan(
                **build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
            )
        units = [plan.get_layer(i).layer for i in range(plan.num_layers())]
        if autogroup == "1":
            runs = group_flat_pattern(list(pattern))
            assert all(isinstance(unit, HybridLayerRun) for unit in units)
            assert [unit.layer_type_list for unit in units] == [
                tuple(pattern[i] for i in run) for run in runs
            ]
            assert [unit.layers for unit in units] == [
                [model.decoder.layers[i] for i in run] for run in runs
            ]
        else:
            assert units == list(model.decoder.layers)

    def test_refused_settings_stop_the_plan(self):
        """``build_schedule_plan`` applies ``check_hybrid_overlap_supported`` before anything."""
        pattern = "*E*E"
        extra_kwargs = {
            **_flex_kwargs(),
            "fine_grained_activation_offloading": True,
            "offload_modules": ["mlp_norm"],
        }
        config = build_hybrid_config(pattern, extra_kwargs=extra_kwargs)
        with deterministic_mode():
            model = build_hybrid_model(config, pattern, VOCAB_SIZE, SEQ_LEN)
            with pytest.raises(ValueError, match="fine_grained_activation_offloading"):
                model.build_schedule_plan(
                    **build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
                )

    @pytest.mark.parametrize(
        "pattern,mtp_num_layers",
        [("*E*E", None), ("M*EME", None), ("[*E][*E]", None), ("[M*E][M*E]", None), ("M*E/*E", 1)],
        ids=["flat", "flat-mamba", "bracketed", "bracketed-mamba", "mtp"],
    )
    @pytest.mark.parametrize("shared_expert_intermediate_size", [None, 512])
    def test_overlap_matches_the_eager_model(
        self, monkeypatch, pattern, mtp_num_layers, shared_expert_intermediate_size
    ):
        _skip_without_mamba(pattern)
        monkeypatch.delenv(AUTOGROUP_ENV, raising=False)
        extra_kwargs = _flex_kwargs()
        if mtp_num_layers is not None:
            extra_kwargs["mtp_num_layers"] = mtp_num_layers
        if shared_expert_intermediate_size is not None:
            extra_kwargs["moe_shared_expert_intermediate_size"] = shared_expert_intermediate_size
        config = build_hybrid_config(pattern, extra_kwargs=extra_kwargs)
        with deterministic_mode():
            data = build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
            model = build_hybrid_model(config, pattern, VOCAB_SIZE, SEQ_LEN)
            params = reset_model(model)

            loss = float16_to_fp32(model(**data))
            loss.backward(torch.ones_like(loss))
            capture_ref = {"outputs": [loss.detach()]}
            capture_ref.update({name: p.grad for name, p in model.named_parameters()})

            reset_model(model, params)
            plan = model.build_schedule_plan(**data)
            out = TransformerModelChunkSchedulePlan.run(plan, None)
            TransformerModelChunkSchedulePlan.run(None, plan, b_grad=torch.ones_like(out))
            torch.cuda.synchronize()
            capture_overlap = {"outputs": [out.detach()]}
            capture_overlap.update({name: p.grad for name, p in model.named_parameters()})

            ok, msg = compare_captures(capture_ref, capture_overlap, True)
            assert ok, f"[rank {torch.distributed.get_rank()}] {msg}"
            del plan, model
            gc.collect()
            torch.cuda.empty_cache()
