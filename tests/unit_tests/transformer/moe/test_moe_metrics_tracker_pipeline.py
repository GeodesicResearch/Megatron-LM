# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker
from tests.unit_tests.test_utilities import Utils

# A hybrid model's MoE layers are a subset of its layers. The router records each MoE layer's
# metric at its global layer number into a tensor of the model's num_layers, while the training
# loop passes the count of MoE layers as num_layers to report(), because that count is the
# divisor of the mean. Here the model has 8 layers, 3 of them MoE (layers 2, 5 and 7), all on
# pipeline stage 0; stage 1 holds none.
NUM_LAYERS = 8
MOE_LAYERS = {2: 1.0, 5: 2.0, 7: 3.0}


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two pipeline stages")
class TestMoEMetricsTrackerOnPipelineStageWithoutMoE:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _report(self, tracker: MoEMetricsTracker) -> dict:
        totals = {}
        tracker.report(
            loss_scale=1.0,
            iteration=1,
            force_initialize=True,
            track_names=["load_balancing_loss"],
            num_layers=len(MOE_LAYERS),
            moe_layer_freq=1,
            total_loss_dict=totals,
        )
        return totals

    def _record(self, tracker: MoEMetricsTracker, value: float, layer: int) -> None:
        tracker.record(
            "load_balancing_loss",
            torch.tensor(value, device="cuda"),
            layer_number=layer,
            num_layers=NUM_LAYERS,
        )

    def test_a_stage_without_moe_layers_reports_with_the_stages_that_have_them(self):
        tracker = MoEMetricsTracker()
        if parallel_state.get_pipeline_model_parallel_rank() == 0:
            for layer, value in MOE_LAYERS.items():
                self._record(tracker, value, layer)

        totals = self._report(tracker)

        # Every rank sees the pipeline's sum over the MoE layers divided by their count, and
        # every rank's tensor has the model's length.
        expected = sum(MOE_LAYERS.values()) / len(MOE_LAYERS)
        assert totals["load_balancing_loss"].item() == pytest.approx(expected)
        assert tracker.metrics["load_balancing_loss"].values.numel() == NUM_LAYERS

    def test_a_stage_without_moe_layers_records_normally_after_its_placeholder(self):
        tracker = MoEMetricsTracker()
        if parallel_state.get_pipeline_model_parallel_rank() == 0:
            self._record(tracker, 1.0, 2)
        self._report(tracker)

        # The next step records on both stages: the placeholder that stage 1 held is replaced by
        # a recorded entry.
        layer = 2 if parallel_state.get_pipeline_model_parallel_rank() == 0 else 7
        self._record(tracker, 4.0, layer)
        totals = self._report(tracker)

        assert totals["load_balancing_loss"].item() == pytest.approx(8.0 / len(MOE_LAYERS))
