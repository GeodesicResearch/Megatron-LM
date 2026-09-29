# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-FSDP and the EP overlap on hybrid models.

[Geodesic adaptation] The hybrid overlap schedule's units (flat-pattern layer runs and bracketed
HybridStack groups) are not reached by Megatron-FSDP's unit discovery and reshard hooks, so
``FullyShardedDataParallel`` refuses a hybrid model with ``overlap_moe_expert_parallel_comm``.
Without the overlap, wrapping a hybrid model is unchanged.
"""

import pytest

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.utils import is_te_min_version, is_torch_min_version
from tests.unit_tests.a2a_overlap.utils import (
    build_hybrid_config,
    build_hybrid_model,
    deterministic_mode,
    get_valid_flex_dispatcher_backend,
)
from tests.unit_tests.test_utilities import Utils

SEQ_LEN = 32
VOCAB_SIZE = 128


def _make_ddp_config():
    return DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy="optim_grads_params",
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        megatron_fsdp_main_params_dtype=None,
    )


def _flex_kwargs():
    backend = get_valid_flex_dispatcher_backend()
    if backend is None:
        pytest.skip("No flex dispatcher backend available")
    return {"moe_token_dispatcher_type": "flex", "moe_flex_dispatcher_backend": backend}


@pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
@pytest.mark.skipif(not is_torch_min_version("2.6.0"), reason="EP overlap hangs on torch < 2.6.0")
class TestFSDPHybridOverlap:
    """Megatron-FSDP refuses the overlap on hybrid models and wraps them as before without it."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("hybrid_layer_pattern", ["*E*E", "[*E][*E]"])
    def test_overlap_is_refused(self, hybrid_layer_pattern):
        extra_kwargs = {**_flex_kwargs(), "overlap_moe_expert_parallel_comm": True}
        config = build_hybrid_config(hybrid_layer_pattern, extra_kwargs=extra_kwargs)
        with deterministic_mode():
            model = build_hybrid_model(config, hybrid_layer_pattern, VOCAB_SIZE, SEQ_LEN)
            with pytest.raises(NotImplementedError, match="hybrid"):
                FullyShardedDataParallel(config=config, ddp_config=_make_ddp_config(), module=model)

    def test_without_the_overlap_the_model_is_wrapped(self):
        config = build_hybrid_config("*E*E", extra_kwargs=_flex_kwargs())
        with deterministic_mode():
            model = build_hybrid_model(config, "*E*E", VOCAB_SIZE, SEQ_LEN)
            wrapped = FullyShardedDataParallel(
                config=config, ddp_config=_make_ddp_config(), module=model
            )
        assert wrapped.module.module is model
