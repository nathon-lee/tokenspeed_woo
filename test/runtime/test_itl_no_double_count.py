# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests that inter-token latency (ITL) is observed exactly once per decode step.

Regression test for a bug where ``collect_metrics`` was calling
``observe_inter_token_latency`` three times per request lifetime:

1. Per-step wall-clock ITL   (correct — this one is kept)
2. Per-step pure/scheduler ITL via ``recv_obj.generated_time`` (removed)
3. End-to-end TPOT at finish via total elapsed / total tokens (removed)

The fix removes entries 2 and 3.  These tests assert the current (correct)
behaviour: exactly one observation per decode step and no extra observation at
finish.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.output_processor import (  # noqa: E402
    OutputProcessor,
    ReqState,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _RecordingMetrics:
    """Record every call to observe_inter_token_latency and related helpers."""

    enabled = True

    def __init__(self):
        self.itl_calls: list[tuple[float, int]] = []
        self.ttft_calls: list[float] = []
        self.finish_calls: int = 0

    def observe_inter_token_latency(self, interval: float, num_new_tokens: int):
        self.itl_calls.append((interval, num_new_tokens))

    def observe_time_to_first_token(self, value: float):
        self.ttft_calls.append(value)

    def record_request_finish(self, stats):
        self.finish_calls += 1


class _StubEngine:
    """Minimal AsyncLLM stand-in for OutputProcessor unit tests."""

    def __init__(self, metrics: _RecordingMetrics):
        self.metrics = metrics
        self.rid_to_state: dict[str, ReqState] = {}
        self.enable_metrics = True
        self.dump_requests_folder = False
        self.server_args = types.SimpleNamespace(
            speculative_algorithm=None,
            weight_version="test",
            enable_inline_detokenizer=False,
            stream_output=False,
            skip_tokenizer_init=True,
        )
        self.tokenizer = None
        self.output_processor = None  # not needed by collect_metrics

    def _attach_processor(self):
        self.output_processor = OutputProcessor(self)
        return self.output_processor


class _StubObj:
    stream = False
    log_metrics = False
    return_logprob = False
    top_logprobs_num = 0
    token_ids_logprob = None
    return_text_in_logprobs = False


def _mk_state(created_time: float = 0.0) -> ReqState:
    return ReqState(
        collector=RequestOutputCollector(),
        finished=False,
        event=asyncio.Event(),
        obj=_StubObj(),
        created_time=created_time,
    )


class _BatchOut:
    """Minimal stand-in for BatchStrOut used by collect_metrics."""

    def __init__(
        self,
        completion_tokens: int,
        finished_reason=None,
        prompt_tokens: int = 4,
        cached_tokens: int = 0,
    ):
        self.completion_tokens = [completion_tokens]
        self.finished_reasons = [finished_reason]
        self.prompt_tokens = [prompt_tokens]
        self.cached_tokens = [cached_tokens]
        self.generated_time = 1.0  # scheduler-side timestamp (was used for pure ITL)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_first_token_records_only_ttft():
    """The very first frame must record TTFT but must NOT record ITL."""
    metrics = _RecordingMetrics()
    engine = _StubEngine(metrics)
    processor = engine._attach_processor()

    state = _mk_state(created_time=0.0)
    state.finished = False

    processor.collect_metrics(state, _BatchOut(completion_tokens=1), i=0)

    assert len(metrics.ttft_calls) == 1, "TTFT should be observed once on first token"
    assert metrics.itl_calls == [], "ITL must NOT be observed on the first token frame"


def test_subsequent_tokens_record_itl_exactly_once_per_step():
    """Each decode step after the first must produce exactly one ITL observation."""
    metrics = _RecordingMetrics()
    engine = _StubEngine(metrics)
    processor = engine._attach_processor()

    state = _mk_state(created_time=0.0)

    # Frame 0: first token (completion_tokens=1)
    processor.collect_metrics(state, _BatchOut(completion_tokens=1), i=0)

    # Frame 1: second token (completion_tokens=2)
    processor.collect_metrics(state, _BatchOut(completion_tokens=2), i=0)

    # Frame 2: third token (completion_tokens=3)
    processor.collect_metrics(state, _BatchOut(completion_tokens=3), i=0)

    # Two decode steps → exactly two ITL observations (one per step, not 2×2).
    assert len(metrics.itl_calls) == 2, (
        f"Expected exactly 2 ITL observations (one per decode step), "
        f"got {len(metrics.itl_calls)}"
    )
    # Each observation accounts for 1 new token.
    for interval, n_tokens in metrics.itl_calls:
        assert n_tokens == 1, f"Each step produces 1 new token, got {n_tokens}"
        assert interval >= 0.0


def test_finish_does_not_add_extra_itl_observation():
    """Finishing a request must NOT inject an additional end-to-end TPOT observation."""
    import time

    metrics = _RecordingMetrics()
    engine = _StubEngine(metrics)
    processor = engine._attach_processor()

    state = _mk_state(created_time=time.time())

    # First token
    processor.collect_metrics(state, _BatchOut(completion_tokens=1), i=0)

    # Second token
    processor.collect_metrics(state, _BatchOut(completion_tokens=2), i=0)

    itl_after_two_steps = len(metrics.itl_calls)

    # Finish the request on the third token
    state.finished = True
    state.finished_time = time.time()
    processor.collect_metrics(
        state,
        _BatchOut(completion_tokens=3, finished_reason={"type": "length", "length": 3}),
        i=0,
    )

    # The finish frame adds one more ITL observation (for the new token in that frame).
    # It must NOT add a second, end-to-end TPOT observation on top of that.
    itl_after_finish = len(metrics.itl_calls)
    assert itl_after_finish == itl_after_two_steps + 1, (
        f"Finish frame should add exactly 1 ITL observation (for the new token), "
        f"got {itl_after_finish - itl_after_two_steps} extra observations"
    )
    # record_request_finish must be called exactly once for the end-to-end stats.
    assert metrics.finish_calls == 1


def test_speculative_decode_multi_token_step_single_itl_call():
    """A speculative-decode step that accepts N tokens is ONE ITL call with N tokens."""
    metrics = _RecordingMetrics()
    engine = _StubEngine(metrics)
    processor = engine._attach_processor()

    state = _mk_state(created_time=0.0)

    # Frame 0: first token from prefill
    processor.collect_metrics(state, _BatchOut(completion_tokens=1), i=0)

    # Frame 1: speculative step accepts 4 tokens at once (completion_tokens goes 1→5)
    processor.collect_metrics(state, _BatchOut(completion_tokens=5), i=0)

    assert len(metrics.itl_calls) == 1, (
        "One decode step regardless of accepted token count → one ITL call"
    )
    assert metrics.itl_calls[0][1] == 4, "num_new_tokens should equal tokens accepted"
