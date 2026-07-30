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

"""Unit tests for same-batch pause/generate FIFO ordering in EventLoop."""

from __future__ import annotations

from tokenspeed.runtime.engine.event_loop import EventLoop


class _Spec:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.tokens = [1, 2, 3]


class _State:
    def __init__(self):
        self.finished = False
        self.input_length = 3


class _Output:
    def __init__(self):
        self.rid_to_state = {}
        self.abort_marks = []

    def sweep_pending_aborts(self) -> None:
        return None

    def mark_abort(self, rid: str, notify_client: bool = False) -> None:
        self.abort_marks.append((rid, notify_client))

    def register(self, rid: str, state: _State) -> None:
        self.rid_to_state[rid] = state

    def publish_finished_at_admission(self, rid: str, _state: _State) -> None:
        self.rid_to_state.pop(rid, None)


class _GrammarManager:
    def __init__(self):
        self.grammar_queue = []
        self.abort_marks = []

    def process_req_with_grammar(self, _state: _State) -> bool:
        return True

    def add_to_queue(self, _spec: _Spec, _state: _State, _bootstrap) -> None:
        return None

    def get_ready_grammar_requests(self):
        return []

    def mark_abort(self, rid: str) -> None:
        self.abort_marks.append(rid)


class _Pause:
    def __init__(self, *, abort_all: bool):
        self.admit_blocked = True
        self.buffered_specs = []
        self._abort_all = abort_all

    def consume_abort_all(self) -> bool:
        out = self._abort_all
        self._abort_all = False
        return out

    def consume_cancel_grammar(self) -> bool:
        return False

    def take_buffered_specs(self):
        specs, self.buffered_specs = self.buffered_specs, []
        return specs

    def buffer_specs(self, specs):
        self.buffered_specs.extend(specs)


class _Scheduler:
    def __init__(self):
        self.submits = []

    def submit_requests(self, specs) -> None:
        self.submits.append([spec.request_id for spec in specs])


class _RequestHandler:
    def __init__(self, grammar_manager: _GrammarManager):
        self.grammar_manager = grammar_manager

    def recv_reqs(self):
        return [object()]

    def process_requests(self, _recv_reqs):
        specs = [_Spec("pre"), _Spec("post")]
        states = [_State(), _State()]
        bootstrap_infos = [None, None]
        abort_rids = []
        pre_pause_generate_rids = ["pre"]
        return specs, states, bootstrap_infos, abort_rids, pre_pause_generate_rids


def _make_loop(*, abort_all: bool) -> EventLoop:
    loop = EventLoop.__new__(EventLoop)
    grammar_manager = _GrammarManager()
    loop.request_handler = _RequestHandler(grammar_manager)
    loop.output_processor = _Output()
    loop.scheduler = _Scheduler()
    loop._pause = _Pause(abort_all=abort_all)
    loop._flatkv_pd_enabled = False
    loop.kv_transfer = None
    loop.memory_executor = None
    loop.prefetch_threshold = 1
    loop.epd_admission = None
    loop._epd_staged = {}
    loop._is_epd_request = lambda _state: False
    return loop


def test_same_batch_pause_keeps_pre_pause_generate_as_pre_pause_work() -> None:
    loop = _make_loop(abort_all=False)

    EventLoop._process_new_requests(loop)

    # The generate request that arrived before pause in the same recv batch
    # must be submitted now; only post-pause requests are buffered.
    assert loop.scheduler.submits == [["pre"]]
    assert [spec.request_id for spec in loop._pause.buffered_specs] == ["post"]


def test_same_batch_pause_abort_marks_pre_pause_generate_for_abort() -> None:
    loop = _make_loop(abort_all=True)

    EventLoop._process_new_requests(loop)

    assert loop.scheduler.submits == [["pre"]]
    assert [spec.request_id for spec in loop._pause.buffered_specs] == ["post"]
    assert loop.output_processor.abort_marks == [("pre", True)]
    assert loop.request_handler.grammar_manager.abort_marks == ["pre"]
