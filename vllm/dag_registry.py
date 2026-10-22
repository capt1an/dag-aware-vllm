# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DAGRegistry — server-side DAG metadata store for DAG-aware scheduling.

Clients register their task DAGs once via POST /v1/dags. Subsequent chat
completion requests can carry dag_id + node_id fields so that the serving
layer computes scheduling priority from the DAG's critical path, rather than
requiring the client to supply a raw priority integer.

Priority encoding (二元组, vLLM min-heap, 值越小越优先):
    priority = (dag_remaining_cp, -node_cp_remaining)

  - 第一级 dag_remaining_cp：整个 DAG 剩余的关键路径长度越小，
    该 DAG 的所有节点整体排在前面，保证"工作量最小的 DAG 先跑"。
  - 第二级 -node_cp_remaining：同一 DAG 内，节点自身关键路径越长
    （即解锁下游工作越多）越优先，最小化 DAG makespan。

dag_remaining_cp 定义：
    从所有"尚未完成且入度（在未完成子图中）为 0"的节点出发，
    取各节点 cp_remaining 的最大值，即当前剩余子图的真实关键路径长度。
    （对并行分支求 max 而非 sum，避免重复计算。）

单例机制：
    _DAGStore 通过 sys.modules 绑定到固定 key "__dag_registry_store__"。
    无论 dag_registry 以何种路径被 import（vllm.dag_registry /
    dag_registry / 相对 import 等），所有调用方共享同一个 _DAGStore
    实例，从根本上避免重复 import 导致的多份 _dags 字典问题。
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import cast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAG entry
# ---------------------------------------------------------------------------


@dataclass
class _DAGEntry:
    nodes: dict[str, int]  # node_id -> expected_tokens
    successors: dict[str, list[str]]  # node_id -> [successor node_ids]
    predecessors: dict[str, list[str]]  # node_id -> [predecessor node_ids]
    version: int = 1

    _cp_cache: dict[str, int] = field(default_factory=dict, init=False)
    # None 表示缓存已失效，需要重新计算
    _dag_remaining_cache: int | None = field(default=None, init=False)
    completed_nodes: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # cp_remaining：节点自身 + 后继的最大关键路径（静态，不受 completed 影响）
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # dag_remaining_cp：当前剩余子图的关键路径长度
    # ------------------------------------------------------------------

    def dag_remaining_cp(self) -> int:
        """剩余子图的关键路径长度（max，非 sum）。

        在未完成节点中，找出入度为 0 的 frontier 节点
        （所有前驱均已完成，或本身就是源节点），取其 cp_remaining 的最大值。
        """
        if self._dag_remaining_cache is not None:
            return self._dag_remaining_cache
        result = self._compute_dag_remaining_cp()
        self._dag_remaining_cache = result
        logger.debug(
            "dag_remaining_cp computed: %d  (completed=%s)",
            result,
            self.completed_nodes,
        )
        return result

    def _compute_dag_remaining_cp(self) -> int:
        if not self.nodes:
            return 0
        frontier = [
            n
            for n in self.nodes
            if n not in self.completed_nodes
            and all(p in self.completed_nodes for p in self.predecessors.get(n, []))
        ]
        if not frontier:
            return 0
        return max(self.cp_remaining(n) for n in frontier)

    def invalidate_dag_remaining_cache(self) -> None:
        """节点完成或 cost 更新后调用，使 dag_remaining 缓存失效。"""
        self._dag_remaining_cache = None


# ---------------------------------------------------------------------------
# Singleton store — 绑定到 sys.modules，跨 import 路径共享同一实例
# ---------------------------------------------------------------------------

_STORE_KEY = "__dag_registry_store__"


class _DAGStore:
    """进程级单例，持有所有 DAG 的状态。

    通过 sys.modules[_STORE_KEY] 存储，确保无论 dag_registry 以哪种路径
    被 import，所有调用方都操作同一份数据。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dags: dict[str, _DAGEntry] = {}

    # ------------------------------------------------------------------
    # public API（供模块级函数委托调用）
    # ------------------------------------------------------------------

    def register(
        self,
        dag_id: str,
        nodes: dict[str, int],
        edges: list[tuple[str, str]],
    ) -> None:
        successors: dict[str, list[str]] = {n: [] for n in nodes}
        predecessors: dict[str, list[str]] = {n: [] for n in nodes}
        for src, dst in edges:
            successors.setdefault(src, []).append(dst)
            predecessors.setdefault(dst, []).append(src)
        with self._lock:
            previous = self._dags.get(dag_id)
            self._dags[dag_id] = _DAGEntry(
                nodes=dict(nodes),
                successors=successors,
                predecessors=predecessors,
                version=previous.version + 1 if previous else 1,
            )
        logger.debug("registered dag '%s' with %d nodes", dag_id, len(nodes))

    def update_node_cost(self, dag_id: str, node_id: str, new_cost: int) -> bool:
        with self._lock:
            entry = self._dags.get(dag_id)
            if entry is None or node_id not in entry.nodes:
                return False
            entry.nodes[node_id] = max(1, new_cost)
            entry.version += 1
            entry._cp_cache.clear()
            entry.invalidate_dag_remaining_cache()
            return True

    def compute_priority(self, dag_id: str, node_id: str) -> tuple[int, int] | None:
        with self._lock:
            entry = self._dags.get(dag_id)
            if entry is None or node_id not in entry.nodes:
                return None
            dag_remaining = entry.dag_remaining_cp()
            node_remaining = entry.cp_remaining(node_id)
        return (dag_remaining, -node_remaining)

    def mark_completed(self, dag_id: str, node_id: str) -> bool:
        with self._lock:
            entry = self._dags.get(dag_id)
            if entry is None:
                logger.warning(
                    "mark_completed: dag '%s' not found (store id=%d)",
                    dag_id,
                    id(self._dags),
                )
                return False
            if node_id not in entry.nodes:
                logger.warning(
                    "mark_completed: node '%s' not found in dag '%s'",
                    node_id,
                    dag_id,
                )
                return False
            entry.completed_nodes.add(node_id)
            entry.version += 1
            entry.invalidate_dag_remaining_cache()
            logger.debug(
                "mark_completed: dag='%s' node='%s' completed=%s",
                dag_id,
                node_id,
                entry.completed_nodes,
            )
            return True

    def list_dags(self) -> list[str]:
        with self._lock:
            return list(self._dags.keys())

    def reopen_node(self, dag_id: str, node_id: str) -> bool:
        """A native follow-up resumes this task and invalidates finished dependents."""
        with self._lock:
            entry = self._dags.get(dag_id)
            if entry is None or node_id not in entry.nodes:
                return False
            pending = [node_id]
            seen = set()
            while pending:
                node = pending.pop()
                if node in seen:
                    continue
                seen.add(node)
                entry.completed_nodes.discard(node)
                pending.extend(entry.successors.get(node, []))
            entry.invalidate_dag_remaining_cache()
            entry.version += 1
            return True

    def snapshot(self, dag_id: str) -> dict | None:
        """Return a consistent diagnostic view without exposing mutable state."""
        with self._lock:
            entry = self._dags.get(dag_id)
            if entry is None:
                return None
            return {
                "dag_id": dag_id,
                "version": entry.version,
                "nodes": dict(entry.nodes),
                "edges": [
                    [src, dst]
                    for src, destinations in entry.successors.items()
                    for dst in destinations
                ],
                "completed_nodes": sorted(entry.completed_nodes),
                "dag_remaining_cp": entry.dag_remaining_cp(),
                "node_cp_remaining": {
                    node: entry.cp_remaining(node) for node in entry.nodes
                },
            }


def _get_store() -> _DAGStore:
    """返回进程唯一的 _DAGStore 实例。

    首次调用时创建并写入 sys.modules[_STORE_KEY]；后续调用直接返回缓存值。
    无论本模块以 'vllm.dag_registry'、'dag_registry' 还是相对路径被 import，
    所有调用方拿到的都是同一个对象。
    """
    store = cast("_DAGStore | None", sys.modules.get(_STORE_KEY))
    if store is None:
        store = _DAGStore()
        sys.modules[_STORE_KEY] = store  # type: ignore[assignment]
        logger.debug("_DAGStore created (id=%d)", id(store))
    return store


# ---------------------------------------------------------------------------
# 模块级公开函数（保持原有调用接口不变）
# ---------------------------------------------------------------------------


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
    _get_store().register(dag_id, nodes, edges)


def update_node_cost(dag_id: str, node_id: str, new_cost: int) -> bool:
    """
    Update a node's expected token cost with an online LLM estimate.

    Clears the cp_remaining cache so subsequent compute_priority() calls
    use the new cost. Thread-safe; returns False if dag_id/node_id unknown.
    """
    return _get_store().update_node_cost(dag_id, node_id, new_cost)


def compute_priority(dag_id: str, node_id: str) -> tuple[int, int] | None:
    """
    Return the scheduling priority for (dag_id, node_id).

    Returns
    -------
    (dag_priority, node_priority) : tuple[int, int]
        The min-heap prioritizes smaller remaining DAGs, then longer node paths:
          - dag_priority  = dag_remaining_cp  (DAG 剩余关键路径越短越优先)
          - node_priority = -node_cp_remaining (节点关键路径越长越优先)

    Returns None if dag_id or node_id is not registered, allowing the
    caller to fall back to request.priority.
    """
    return _get_store().compute_priority(dag_id, node_id)


def mark_completed(dag_id: str, node_id: str) -> bool:
    """
    Mark a node as completed and invalidate the dag_remaining cache.

    节点完成后，剩余子图的关键路径可能缩短，必须使 dag_remaining_cp 缓存
    失效，确保后续 compute_priority 使用最新值。
    """
    return _get_store().mark_completed(dag_id, node_id)


def list_dags() -> list[str]:
    return _get_store().list_dags()


def snapshot(dag_id: str) -> dict | None:
    return _get_store().snapshot(dag_id)


def reopen_node(dag_id: str, node_id: str) -> bool:
    return _get_store().reopen_node(dag_id, node_id)
