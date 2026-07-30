"""Regression guards for ``AsyncLLM`` request cancellation.

Locks the contract Phase G.1 established: when the task driving a
streaming generator is cancelled (e.g. FastAPI cancelling its route
coroutine because the client disconnected), the generator's
``finally`` drops the rid from ``rid_to_state`` and fires exactly one
``AbortReq`` at the scheduler. No dangling per-request state. No
reliance on ``fastapi.Request.is_disconnected()``.

This test uses a stub that bypasses ZMQ / ModelConfig / HF tokenizer
bring-up — the same pattern as
``test/runtime/test_inline_detokenizer_receiver.py``.
"""

import os
import sys
import unittest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

import asyncio  # noqa: E402
import types  # noqa: E402
from typing import Any, Dict  # noqa: E402

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.exceptions import EngineGenerateError  # noqa: E402
from tokenspeed.runtime.engine.output_processor import ReqState  # noqa: E402


class _FakeScheduler:
    """Captures ``AbortReq``s the stub AsyncLLM pushes on cancellation."""

    def __init__(self) -> None:
        self.aborts: list = []

    def send_pyobj(self, obj: Any) -> None:
        self.aborts.append(obj)


class _StubAsyncLLM(AsyncLLM):
    """Bypass ZMQ + ModelConfig + HF bring-up.

    Populates only the attributes ``_wait_one_response`` + ``abort_request``
    read: ``rid_to_state``, ``log_requests``, ``enable_metrics``,
    ``tokenizer``, ``model_config``, and an ``engine_core_client`` whose
    ``send_to_scheduler`` is a capture fake so the test can assert the
    ``AbortReq`` was sent.
    """

    def __init__(self) -> None:
        self.rid_to_state: Dict[str, ReqState] = {}
        self.log_requests = False
        self.enable_metrics = False
        self.tokenizer = None
        self.model_config = types.SimpleNamespace(is_multimodal_gen=False)
        self.engine_core_client = types.SimpleNamespace(
            send_to_scheduler=_FakeScheduler()
        )


class _StubReqObj:
    """Minimal stand-in for ``GenerateReqInput`` used by
    ``_wait_one_response``. Only the attributes actually read by the
    generator body are provided.
    """

    def __init__(self, *, rid: str = "r1", stream: bool = True, input_ids=None) -> None:
        self.rid = rid
        self.stream = stream
        self.input_ids = input_ids
        self.text = None
        self.sampling_params = {"skip_special_tokens": False}


class _StubBatchReq:
    def __init__(self, *, stream: bool) -> None:
        self.stream = stream
        self.batch_size = 2
        self.parallel_sample_num = 1
        self.requests = [
            _StubReqObj(rid="r-failing", stream=stream),
            _StubReqObj(rid="r-sibling", stream=stream),
        ]

    def __getitem__(self, index: int) -> _StubReqObj:
        return self.requests[index]


class _StubBatchAsyncLLM(_StubAsyncLLM):
    async def _tokenize_one_request(self, obj: _StubReqObj) -> _StubReqObj:
        return obj

    def _send_one_request(
        self,
        obj: _StubReqObj,
        tokenized_obj: _StubReqObj,
        created_time: float | None = None,
    ) -> None:
        self.rid_to_state[obj.rid] = _fresh_state(obj)

    async def _wait_one_response(self, obj: _StubReqObj):
        if obj.rid == "r-failing":
            await asyncio.sleep(0)
            self.abort_request(obj.rid)
            raise EngineGenerateError("batch member failed")
        async for output in super()._wait_one_response(obj):
            yield output


def _fresh_state(obj: _StubReqObj) -> ReqState:
    return ReqState(
        RequestOutputCollector(),
        False,
        asyncio.Event(),
        obj,
        created_time=0.0,
    )


class TestWaitOneResponseCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_before_first_output_cleans_up(self) -> None:
        """Cancel while ``_wait_one_response`` is blocked on the first
        event. Expect: rid dropped from ``rid_to_state``; one ``AbortReq``
        sent to the scheduler.
        """
        mgr = _StubAsyncLLM()
        obj = _StubReqObj(rid="r-cancel-1", stream=True)
        state = _fresh_state(obj)
        mgr.rid_to_state[obj.rid] = state

        # Drive the generator via a task so we can cancel it.
        gen = mgr._wait_one_response(obj)

        async def drain() -> None:
            async for _ in gen:
                pass

        task = asyncio.create_task(drain())

        # Give the generator one event-loop tick to enter
        # ``await state.event.wait()``.
        await asyncio.sleep(0)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        # Finally block cleanup.
        self.assertNotIn(obj.rid, mgr.rid_to_state)
        aborts = mgr.engine_core_client.send_to_scheduler.aborts
        self.assertEqual(
            len(aborts),
            1,
            f"expected exactly one AbortReq on cancel, got {len(aborts)}",
        )
        # The AbortReq should carry the right rid.
        self.assertEqual(getattr(aborts[0], "rid", None), obj.rid)

    async def test_normal_finish_does_not_fire_abort(self) -> None:
        """When the generator exits on ``state.finished`` (scheduler's
        terminal frame), no ``AbortReq`` should fire — the scheduler
        already knows the request is done.
        """
        mgr = _StubAsyncLLM()
        obj = _StubReqObj(rid="r-finish-1", stream=True)
        state = _fresh_state(obj)
        mgr.rid_to_state[obj.rid] = state

        gen = mgr._wait_one_response(obj)

        async def drive() -> list:
            out = []
            async for chunk in gen:
                out.append(chunk)
            return out

        task = asyncio.create_task(drive())
        await asyncio.sleep(0)

        # Simulate a terminal frame from the scheduler: mark finished,
        # put the final out_dict into the collector, wake the generator.
        state.finished = True
        state.collector.put(
            {
                "text": "hello",
                "output_ids": [1, 2],
                "meta_info": {"id": obj.rid, "finish_reason": None},
            },
            stream=True,
        )
        state.event.set()

        results = await task

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "hello")
        self.assertNotIn(obj.rid, mgr.rid_to_state)
        # Normal finish ⇒ no abort.
        self.assertEqual(
            len(mgr.engine_core_client.send_to_scheduler.aborts),
            0,
            "normal finish should not schedule an AbortReq",
        )


class TestBatchResponseCancellation(unittest.IsolatedAsyncioTestCase):
    async def _assert_failed_batch_cleans_up_sibling(self, *, stream: bool) -> None:
        mgr = _StubBatchAsyncLLM()
        obj = _StubBatchReq(stream=stream)

        async def drain() -> None:
            async for _ in mgr._handle_batch_request(obj):
                pass

        with self.assertRaisesRegex(EngineGenerateError, "batch member failed"):
            await drain()

        self.assertEqual(mgr.rid_to_state, {})
        aborted_rids = {
            abort.rid for abort in mgr.engine_core_client.send_to_scheduler.aborts
        }
        self.assertEqual(aborted_rids, {"r-failing", "r-sibling"})

    async def test_non_streaming_error_cleans_up_sibling(self) -> None:
        await self._assert_failed_batch_cleans_up_sibling(stream=False)

    async def test_streaming_error_cleans_up_sibling(self) -> None:
        await self._assert_failed_batch_cleans_up_sibling(stream=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
