# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic heap updates, version ordering, API synchronization and wire metadata."""

import unittest
import uuid

import msgspec
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm import SamplingParams, dag_registry
from vllm.dag_api_router import attach_router
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, PriorityRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request


def task(rid, node=None, dag=None, priority=(110, -20), arrival=1, offset=0):
    return Request(
        rid,
        [1, 2],
        SamplingParams(max_tokens=16),
        None,
        priority=priority,
        arrival_time=arrival,
        dag_id=dag,
        node_id=node,
        dag_priority_offset=offset,
    )


def scheduler():
    instance = object.__new__(Scheduler)
    instance.requests = {}
    instance.waiting = PriorityRequestQueue()
    instance.skipped_waiting = PriorityRequestQueue()
    instance.running = []
    instance.dag_priority_snapshots = {}
    instance.dag_schedule_trace = False
    return instance


def plan():
    return {
        "nodes": [
            {"node_id": "a", "expected_tokens": 100},
            {"node_id": "b", "expected_tokens": 10},
            {"node_id": "join", "expected_tokens": 10},
        ],
        "edges": [["a", "join"], ["b", "join"]],
    }


class DynamicDAGTests(unittest.TestCase):
    def test_batch_heap_updates_preserve_identity_arrival_and_fcfs_order(self):
        for cls in (PriorityRequestQueue, FCFSRequestQueue):
            queue = cls()
            a, b, c = (
                task("a", priority=(100, -10)),
                task("b", priority=(100, -20)),
                task("c", priority=(50, -50)),
            )
            for request in (a, b, c):
                queue.add_request(request)
            self.assertEqual(
                queue.update_priorities(
                    {"a": (10, -5), "b": (10, -20), "unknown": (0, 0)}
                ),
                2,
            )
            self.assertEqual(queue.update_priorities({"a": (10, -5)}), 0)
            expected = [b, a, c] if cls is PriorityRequestQueue else [a, b, c]
            for original in expected:
                result = queue.pop_request()
                self.assertIs(result, original)
                self.assertEqual(result.arrival_time, 1)

    def test_actual_scheduler_batches_both_queues_and_rejects_stale_versions(self):
        did = uuid.uuid4().hex
        dag_registry.register(
            did, {"a": 100, "b": 10, "join": 10}, [("a", "join"), ("b", "join")]
        )
        core = scheduler()
        old = dag_registry.snapshot(did)
        core.update_dag_priorities(old)
        b = task("b", "b", did)
        a = task("a", "a", did, priority=(110, -110), offset=-1)
        active = task("running", "a", did, priority=(110, -110))
        other = task("other", "a", "different-dag", priority=(999, -999))
        for request in (a, b, active, other):
            core.requests[request.request_id] = request
        core.waiting.add_request(b)
        core.waiting.add_request(other)
        core.skipped_waiting.add_request(a)
        core.running.append(active)
        dag_registry.update_node_cost(did, "a", 1)
        updated = dag_registry.snapshot(did)
        result = core.update_dag_priorities(updated)
        self.assertEqual(
            (result["waiting_changed"], result["skipped_waiting_changed"]), (1, 1)
        )
        self.assertEqual(b.priority, (20, -20))
        self.assertEqual(a.priority, (20, -12))
        self.assertEqual(active.priority, (110, -110))
        self.assertEqual(other.priority, (999, -999))
        self.assertFalse(core.update_dag_priorities(old)["applied"])
        self.assertFalse(core.update_dag_priorities(updated)["applied"])
        # A request tokenized before the update must use the engine's newer DAG.
        late = task("late", "a", did, priority=(110, -110), arrival=2)
        core.requests["late"] = late
        core._enqueue_waiting_request(late)
        self.assertEqual(late.priority, (20, -11))
        self.assertEqual(core.waiting.pop_request().request_id, "b")
        self.assertEqual(core.waiting.pop_request().request_id, "late")
        self.assertIs(core.skipped_waiting.pop_request(), a)

    def test_previous_counterexample_now_selects_latest_snapshot_order(self):
        did = uuid.uuid4().hex
        dag_registry.register(
            did, {"a": 100, "b": 10, "join": 10}, [("a", "join"), ("b", "join")]
        )
        core = scheduler()
        core.update_dag_priorities(dag_registry.snapshot(did))
        b = task("b", "b", did)
        core.requests["b"] = b
        core._enqueue_waiting_request(b)
        dag_registry.update_node_cost(did, "a", 1)
        core.update_dag_priorities(dag_registry.snapshot(did))
        a = task("a", "a", did)
        core.requests["a"] = a
        core._enqueue_waiting_request(a)
        self.assertEqual(
            [core.waiting.pop_request().request_id for _ in range(2)], ["b", "a"]
        )

    def test_api_ack_syncs_registration_cost_completion_reopen_and_failures(self):
        core = scheduler()

        class Engine:
            fail = False
            versions = []

            async def update_dag_priorities(self, snapshot):
                if self.fail:
                    raise RuntimeError("injected IPC failure")
                self.versions.append(snapshot["version"])
                return core.update_dag_priorities(snapshot)

        engine = Engine()
        app = FastAPI()
        attach_router(app)
        app.state.engine_client = engine
        client = TestClient(app)
        did = uuid.uuid4().hex
        self.assertEqual(
            client.post("/v1/dags", json={"dag_id": did, **plan()}).status_code, 200
        )
        b = task("b", "b", did)
        core.requests["b"] = b
        core._enqueue_waiting_request(b)
        self.assertEqual(
            client.patch(
                f"/v1/dags/{did}/nodes/a", json={"expected_tokens": 1}
            ).status_code,
            200,
        )
        self.assertEqual(b.priority, (20, -20))
        self.assertEqual(
            client.post(f"/v1/dags/{did}/nodes/a/complete").status_code, 200
        )
        self.assertEqual(client.post(f"/v1/dags/{did}/nodes/a/reopen").status_code, 200)
        self.assertEqual(engine.versions, [1, 2, 3, 4])
        engine.fail = True
        self.assertEqual(
            client.patch(
                f"/v1/dags/{did}/nodes/a", json={"expected_tokens": 5}
            ).status_code,
            503,
        )
        engine.fail = False
        self.assertEqual(
            client.patch(
                f"/v1/dags/{did}/nodes/a", json={"expected_tokens": 5}
            ).status_code,
            200,
        )
        self.assertEqual(core.dag_priority_snapshots[did][0], 6)

    def test_dag_metadata_survives_msgpack_and_core_request_conversion(self):
        wire = EngineCoreRequest(
            request_id="r",
            prompt_token_ids=[1, 2],
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=16),
            pooling_params=None,
            arrival_time=123,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            dag_id="d",
            node_id="n",
            dag_priority_offset=-1,
        )
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(wire), type=EngineCoreRequest
        )
        request = Request.from_engine_core_request(decoded, None)
        self.assertEqual(
            (request.dag_id, request.node_id, request.dag_priority_offset),
            ("d", "n", -1),
        )
        self.assertEqual((request.request_id, request.arrival_time), ("r", 123))


if __name__ == "__main__":
    unittest.main()
