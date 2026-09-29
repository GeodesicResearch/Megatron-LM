# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The EP-overlap schedule's free-input policy and the experts property it depends on.

After its forward, the schedule's ``mlp`` node may free its input (the dispatched tokens) only
when nothing saves that tensor for backward. Under an FP8/FP4 recipe TEGroupedMLP saves a
quantized copy, so the input can go; experts that run their GEMMs outside TE stay BF16 and save
the tensor itself, and freeing it corrupts their weight gradient (``setStorage ... out of bounds
for storage of size 0`` in the mlp backward). ``should_free_input`` takes that property as
``experts_quantize_input``, and ``get_layer_moe_metadata`` reads it off the layer.
"""

import pytest
import torch

from megatron.core.models.common.fine_grained_callables import get_layer_moe_metadata
from megatron.core.models.common.utils import should_free_input
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.test_schedule_layer_1f1b import (
    run_transformer_layer_a2a_overlap_with_capture,
    run_transformer_layer_ref_with_capture,
)
from tests.unit_tests.a2a_overlap.utils import (
    build_data,
    compare_captures,
    deterministic_mode,
    get_valid_flex_dispatcher_backend,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils

NODES = ("pre_dispatch_computation", "moe_dispatch", "mlp", "moe_combine")


def _moe_config(**overrides):
    """A small MoE config on the flex dispatcher's hybridep backend, which hands its output to
    the experts unpermuted."""
    kwargs = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=64,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        num_moe_experts=8,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestMlpNode:
    """Whether the experts' input survives the ``mlp`` node's forward."""

    @pytest.mark.parametrize("experts_quantize_input", [True, False])
    def test_bf16_keeps_tokens_the_dispatcher_passes_through(self, experts_quantize_input):
        assert not should_free_input("mlp", True, _moe_config(), 8, experts_quantize_input)

    def test_fp8_frees_the_input_of_experts_that_save_a_quantized_copy(self):
        assert should_free_input("mlp", True, _moe_config(fp8="e4m3"), 8, True)

    def test_fp8_keeps_the_input_of_experts_that_save_it(self):
        assert not should_free_input("mlp", True, _moe_config(fp8="e4m3"), 8, False)

    @pytest.mark.parametrize("num_local_experts,expected", [(8, True), (1, False)])
    def test_alltoall_frees_only_when_dispatch_postprocess_makes_a_new_tensor(
        self, num_local_experts, expected
    ):
        config = _moe_config(moe_token_dispatcher_type="alltoall")
        assert should_free_input("mlp", True, config, num_local_experts, False) is expected


class TestOtherNodes:
    """The flag only decides the ``mlp`` node."""

    def test_dense_layers_keep_every_input(self):
        config = _moe_config()
        assert not any(should_free_input(name, False, config, None, None) for name in NODES)

    @pytest.mark.parametrize("experts_quantize_input", [True, False])
    def test_moe_nodes_other_than_mlp_ignore_the_flag(self, experts_quantize_input):
        config = _moe_config(fp8="e4m3")
        assert should_free_input("moe_combine", True, config, 8, experts_quantize_input)
        assert not should_free_input("moe_dispatch", True, config, 8, experts_quantize_input)
        assert not should_free_input(
            "pre_dispatch_computation", True, config, 8, experts_quantize_input
        )

    def test_moe_layers_must_state_the_flag(self):
        with pytest.raises(TypeError, match="experts_quantize_input"):
            should_free_input("mlp", True, _moe_config(), 8, None)


@pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
class TestLayerMetadata:
    """``get_layer_moe_metadata`` reports whether a real layer's experts quantize their input."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "use_transformer_engine,moe_grouped_gemm,expected",
        [(True, True, True), (False, False, False)],
        ids=["te-grouped-mlp", "local-sequential-mlp"],
    )
    def test_moe_layer(self, use_transformer_engine, moe_grouped_gemm, expected):
        config = _moe_config(expert_model_parallel_size=4, moe_grouped_gemm=moe_grouped_gemm)
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_decoder_block_spec(
                config=config, use_transformer_engine=use_transformer_engine
            ),
            vocab_size=128,
            max_sequence_length=64,
        )
        assert get_layer_moe_metadata(model.decoder.layers[0]) == (True, 2, expected)

    def test_dense_layer(self):
        config = _moe_config(num_moe_experts=None, moe_token_dispatcher_type="alltoall")
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_decoder_block_spec(
                config=config, use_transformer_engine=True
            ),
            vocab_size=128,
            max_sequence_length=64,
        )
        assert get_layer_moe_metadata(model.decoder.layers[0]) == (False, None, None)

    def test_mtp_layer_reports_the_experts_of_the_layer_it_wraps(self):
        """MTP nodes take the flag of the MoE layer inside the MTP layer, so the FP8 rule covers
        them too."""
        config = _moe_config(expert_model_parallel_size=4, moe_grouped_gemm=False, mtp_num_layers=1)
        decoder_spec = get_gpt_decoder_block_spec(config=config, use_transformer_engine=False)
        model = GPTModel(
            config=config,
            transformer_layer_spec=decoder_spec,
            mtp_block_spec=get_gpt_mtp_block_spec(
                config, decoder_spec, use_transformer_engine=False
            ),
            vocab_size=128,
            max_sequence_length=64,
        )
        assert get_layer_moe_metadata(model.mtp.layers[0]) == (True, 2, False)


@pytest.mark.flaky_in_dev
@pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
class TestExpertsThatSaveTheirInputUnderFp8:
    """Non-TE experts under an FP8 recipe train through the overlap exactly as without it.

    Local (non-TE) SequentialMLP experts ignore the FP8 recipe and save the dispatched tokens they
    are given. Freeing those tokens after the expert forward, as the rule did for every expert
    type under FP8, made the backward fail on freed storage.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        get_valid_flex_dispatcher_backend() != "hybridep", reason="Needs the hybridep backend"
    )
    def test_gpt_layer_overlap_matches_the_reference(self):
        config = _moe_config(
            attention_backend="unfused",
            hidden_size=512,
            num_attention_heads=8,
            ffn_hidden_size=512,
            pipeline_dtype=torch.bfloat16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            expert_model_parallel_size=4,
            moe_grouped_gemm=False,
            fp8="e4m3",
            deterministic_mode=True,
        )
        microbatches = 4
        with deterministic_mode():
            # The local spec's FusedLayerNorm creates fp32 parameters whatever params_dtype says;
            # training's Float16Module casts the whole model to bf16, as done here.
            model = (
                GPTModel(
                    config=config,
                    transformer_layer_spec=get_gpt_decoder_block_spec(
                        config=config, use_transformer_engine=False
                    ),
                    vocab_size=100,
                    max_sequence_length=1024,
                )
                .cuda()
                .bfloat16()
            )
            params = reset_model(model)
            input_tensors = [build_data() for _ in range(microbatches)]
            capture_ref = run_transformer_layer_ref_with_capture(model, input_tensors, microbatches)
            reset_model(model, params)
            capture_overlap = run_transformer_layer_a2a_overlap_with_capture(
                model, input_tensors, microbatches
            )
            ok, msg = compare_captures(capture_ref, capture_overlap, True)
            assert ok, f"[rank {torch.distributed.get_rank()}] {msg}"
