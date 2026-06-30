# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

from src.brain_utils.image_handler import ImageHandler


class LegacyVisionOutput:
    pass


def test_calculate_token_metrics_accepts_outputs_without_token_fields():
    metrics = ImageHandler.calculate_token_metrics(None, LegacyVisionOutput(), 2.0)

    assert metrics.input_tokens is None
    assert metrics.output_tokens is None
    assert metrics.total_tokens is None
    assert metrics.tokens_per_second is None
    assert metrics.total_processing_seconds == 2.0
