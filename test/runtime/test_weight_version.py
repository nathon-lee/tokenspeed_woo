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

"""Regression coverage for generation weight-version metadata."""

from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    BatchEmbeddingOut,
    BatchStrOut,
)
from tokenspeed.runtime.engine.output_processor import (  # noqa: E402
    OutputProcessor,
    ReqState,
)
from tokenspeed.runtime.engine.request_types import (  # noqa: E402
    FINISH_ABORT,
    FINISH_LENGTH,
)
from tokenspeed.runtime.engine.weight_transfer.manager import (  # noqa: E402
    WeightTransferManager,
)
from tokenspeed.runtime.entrypoints import control_server  # noqa: E402
from tokenspeed.runtime.entrypoints.sglang_compat_http import (  # noqa: E402
    build_sglang_compat_app,
)
from tokenspeed.runtime.entrypoints.vllm_compat_http import (  # noqa: E402
    build_vllm_compat_app,
)
from tokenspeed.runtime.utils.server_args import (  # noqa: E402
    ServerArgs,
    prepare_server_args,
)


class _FakeLLM:
    def __init__(self) -> None:
        self.server_args = SimpleNamespace(
            weight_version="default",
            model="model-x",
        )
        self.updates = []
        self.succeed = True

    async def init_weights_update_group(self, obj):
        return True, "initialized"

    async def update_weights_from_distributed(self, obj):
        self.updates.append(obj)
        return self.succeed, "distributed"

    async def update_weights_from_tensor(self, obj):
        self.updates.append(obj)
        return self.succeed, "tensor"

    async def update_weights_from_disk(self, obj):
        self.updates.append(obj)
        return self.succeed, "disk", None


class TestWeightTransferVersionLifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.llm = _FakeLLM()
        self.manager = WeightTransferManager(self.llm)
        await self.manager.init_engine(
            {
                "master_address": "127.0.0.1",
                "master_port": 1234,
                "rank_offset": 1,
                "world_size": 2,
            }
        )
        self.payload = {
            "names": ["weight"],
            "dtype_names": ["float32"],
            "shapes": [[1]],
        }

    async def test_chunk_version_is_deferred_until_finish(self):
        await self.manager.start_update()
        await self.manager.update({**self.payload, "weight_version": "v1"})

        self.assertEqual(self.manager.weight_version, "default")
        await self.manager.finish_update()
        self.assertEqual(self.manager.weight_version, "v1")

    async def test_finish_version_overrides_chunk_version(self):
        await self.manager.start_update()
        await self.manager.update({**self.payload, "weight_version": "chunk"})

        await self.manager.finish_update(weight_version="finish")
        self.assertEqual(self.manager.weight_version, "finish")

    async def test_missing_version_preserves_current_version(self):
        self.llm.server_args.weight_version = "existing"
        await self.manager.start_update()
        await self.manager.update(self.payload)

        await self.manager.finish_update()
        self.assertEqual(self.manager.weight_version, "existing")


class _FinishManager:
    def __init__(self) -> None:
        self.calls = []

    async def finish_update(self, weight_version=None):
        self.calls.append(weight_version)


class TestWeightVersionHTTP(unittest.TestCase):
    def test_vllm_finish_accepts_legacy_and_versioned_bodies(self):
        manager = _FinishManager()
        client = TestClient(build_vllm_compat_app(manager))

        self.assertEqual(client.post("/finish_weight_update").status_code, 200)
        self.assertEqual(
            client.post("/finish_weight_update", content=b"not-json").status_code,
            200,
        )
        self.assertEqual(
            client.post(
                "/finish_weight_update",
                json={"weight_version": "v3"},
            ).status_code,
            200,
        )
        self.assertEqual(manager.calls, [None, None, "v3"])

    def test_sglang_version_endpoints(self):
        llm = _FakeLLM()
        client = TestClient(build_sglang_compat_app(llm))

        self.assertEqual(
            client.get("/get_weight_version").json(),
            {"weight_version": "default"},
        )
        self.assertEqual(
            client.get("/model_info").json(),
            {"model_path": "model-x", "weight_version": "default"},
        )
        response = client.post("/update_weight_version", json={"new_version": 7})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(llm.server_args.weight_version, "7")
        self.assertEqual(
            client.post("/update_weight_version", json={}).status_code,
            400,
        )

    def test_sglang_updates_stamp_only_after_success(self):
        llm = _FakeLLM()
        client = TestClient(build_sglang_compat_app(llm))

        response = client.post(
            "/update_weights_from_distributed",
            json={
                "names": ["weight"],
                "dtypes": ["float32"],
                "shapes": [[1]],
                "weight_version": "v8",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(llm.updates[-1].weight_version, "v8")
        self.assertEqual(llm.server_args.weight_version, "v8")

        llm.succeed = False
        response = client.post(
            "/update_weights_from_disk",
            json={"model_path": "/tmp/model", "weight_version": "failed"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(llm.server_args.weight_version, "v8")

    def test_control_server_registers_version_proxy_routes(self):
        routes = {
            (route.path, frozenset(route.methods or []))
            for route in control_server.app.routes
        }
        self.assertIn(("/get_weight_version", frozenset({"GET"})), routes)
        self.assertIn(("/model_info", frozenset({"GET"})), routes)
        self.assertIn(("/update_weight_version", frozenset({"POST"})), routes)


class _Request:
    stream = False
    sampling_params = {}
    return_logprob = False
    log_metrics = False


class _RecordingMetrics:
    def __init__(self) -> None:
        self.request_finishes = []

    def observe_time_to_first_token(self, *_args, **_kwargs) -> None:
        return None

    def observe_inter_token_latency(self, *_args, **_kwargs) -> None:
        return None

    def record_request_finish(self, stats) -> None:
        self.request_finishes.append(stats)


def _batch_str_out_with_finished_reason(finished_reason) -> BatchStrOut:
    return BatchStrOut(
        rids=["rid"],
        finished_reasons=[finished_reason],
        output_strs=["hello"],
        output_ids=[[11]],
        prompt_tokens=[3],
        completion_tokens=[1],
        cached_tokens=[0],
        spec_verify_ct=[0],
        input_token_logprobs_val=[],
        input_token_logprobs_idx=[],
        output_token_logprobs_val=[],
        output_token_logprobs_idx=[],
        input_top_logprobs_val=[],
        input_top_logprobs_idx=[],
        output_top_logprobs_val=[],
        output_top_logprobs_idx=[],
        input_token_ids_logprobs_val=[],
        input_token_ids_logprobs_idx=[],
        output_token_ids_logprobs_val=[],
        output_token_ids_logprobs_idx=[],
        output_hidden_states=[[]],
        batch_accept_draft_tokens=[],
        output_extra_infos=[],
        generated_time=time.time(),
    )


class TestGenerationVersionStamp(unittest.TestCase):
    def test_output_meta_info_carries_current_version(self):
        engine = SimpleNamespace(
            server_args=SimpleNamespace(
                weight_version="v-output",
                speculative_algorithm=None,
            ),
            rid_to_state={},
            enable_metrics=False,
            dump_requests_folder=False,
        )
        state = ReqState(
            RequestOutputCollector(),
            False,
            asyncio.Event(),
            _Request(),
            created_time=0.0,
        )
        engine.rid_to_state["rid"] = state

        OutputProcessor(engine).handle_batch_output(
            BatchEmbeddingOut(
                rids=["rid"],
                finished_reasons=[None],
                embeddings=[[1.0]],
                prompt_tokens=[3],
            )
        )

        self.assertEqual(
            state.collector.take()["meta_info"]["weight_version"],
            "v-output",
        )

    def test_server_args_default_version(self):
        self.assertEqual(ServerArgs(model="model-x").weight_version, "default")

    def test_server_args_accepts_initial_version_from_cli(self):
        server_args = prepare_server_args(["model-x", "--weight-version", "policy-v1"])
        self.assertEqual(server_args.weight_version, "policy-v1")


class TestFinishedReasonNormalization(unittest.TestCase):
    def _build_engine(self, *, metrics) -> SimpleNamespace:
        return SimpleNamespace(
            server_args=SimpleNamespace(
                weight_version="v-output",
                speculative_algorithm=None,
            ),
            rid_to_state={},
            enable_metrics=True,
            metrics=metrics,
            dump_requests_folder=False,
        )

    def _build_state(self) -> ReqState:
        return ReqState(
            RequestOutputCollector(),
            False,
            asyncio.Event(),
            _Request(),
            created_time=time.time(),
        )

    def test_finish_length_object_is_normalized_to_dict(self):
        metrics = _RecordingMetrics()
        engine = self._build_engine(metrics=metrics)
        state = self._build_state()
        engine.rid_to_state["rid"] = state

        OutputProcessor(engine).handle_batch_output(
            _batch_str_out_with_finished_reason(FINISH_LENGTH(length=1))
        )

        out = state.collector.take()
        self.assertEqual(out["meta_info"]["finish_reason"], {"type": "length", "length": 1})
        self.assertEqual(len(metrics.request_finishes), 1)
        self.assertTrue(metrics.request_finishes[0].finished_ok)

    def test_finish_abort_object_marks_finished_not_ok(self):
        metrics = _RecordingMetrics()
        engine = self._build_engine(metrics=metrics)
        state = self._build_state()
        engine.rid_to_state["rid"] = state

        OutputProcessor(engine).handle_batch_output(
            _batch_str_out_with_finished_reason(FINISH_ABORT(message="boom"))
        )

        out = state.collector.take()
        self.assertEqual(out["meta_info"]["finish_reason"]["type"], "abort")
        self.assertEqual(out["meta_info"]["finish_reason"]["message"], "boom")
        self.assertEqual(len(metrics.request_finishes), 1)
        self.assertFalse(metrics.request_finishes[0].finished_ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
