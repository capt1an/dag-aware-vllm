"""
DAGRegistry — server-side DAG metadata store for DAG-aware scheduling.

Clients register their task DAGs once via POST /v1/dags. Subsequent chat
completion requests can carry dag_id + node_id fields so that the serving
layer computes scheduling priority from the DAG's critical path, rather than
requiring the client to supply a raw priority integer.

Priority encoding: priority = -round(cp_remaining), so critical-path nodes
(large cp_remaining) receive more negative (higher-priority) values and are
scheduled first by vLLM's PriorityRequestQueue (min-heap).

Leaf-node correction: a node with no successors (downstream_cp == 0) gets
priority = 0. Boosting a leaf's priority delays sibling nodes without
unblocking any downstream work, which can only hurt makespan.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


_registry_lock = threading.Lock()
_dags: dict[str, "_DAGEntry"] = {}


@dataclass
class _DAGEntry:
    nodes: dict[str, int]             # node_id -> expected_tokens
    successors: dict[str, list[str]]  # node_id -> list of successor node_ids
    _cp_cache: dict[str, int] = field(default_factory=dict, init=False)

    def cp_remaining(self, node_id: str) -> int:
        """Total critical-path token cost from this node (inclusive) onward."""
        if node_id in self._cp_cache:
            return self._cp_cache[node_id]
        cost = self.nodes.get(node_id, 0)
        downstream = max(
            (self.cp_remaining(s) for s in self.successors.get(node_id, [])),
            default=0,
        )
        result = cost + downstream
        self._cp_cache[node_id] = result
        return result

    def downstream_cp(self, node_id: str) -> int:
        """Max cp_remaining among successors (excludes this node's own cost)."""
        return max(
            (self.cp_remaining(s) for s in self.successors.get(node_id, [])),
            default=0,
        )


def register(
    dag_id: str,
    nodes: dict[str, int],
    edges: list[tuple[str, str]],
) -> None:
    """
    Register a DAG.

    Parameters
    ----------
    dag_id : str
        Unique identifier for this DAG (e.g. a task UUID).
    nodes : dict[str, int]
        Mapping of node_id to expected token output cost.
    edges : list[tuple[str, str]]
        Directed edges (src_node_id, dst_node_id) representing dependencies.
    """
    successors: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        successors.setdefault(src, []).append(dst)
    with _registry_lock:
        _dags[dag_id] = _DAGEntry(nodes=dict(nodes), successors=successors)


def compute_priority(dag_id: str, node_id: str) -> int | None:
    """
    Return the scheduling priority for (dag_id, node_id).

    Returns None if the dag_id or node_id is not registered, allowing the
    caller to fall back to request.priority.
    """
    with _registry_lock:
        entry = _dags.get(dag_id)
    if entry is None or node_id not in entry.nodes:
        return None
    downstream = entry.downstream_cp(node_id)
    if downstream == 0:
        return 0  # leaf node: neutral priority avoids monopolizing slots
    return -round(entry.cp_remaining(node_id))


def list_dags() -> list[str]:
    with _registry_lock:
        return list(_dags.keys())
