# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DAG registration HTTP endpoints for vLLM.

POST  /v1/dags                          — register a DAG structure (nodes + edges).
GET   /v1/dags                          — list registered DAG IDs.
PATCH /v1/dags/{dag_id}/nodes/{node_id} — update a node's expected token cost
                                          with an online LLM self-estimate.

After registration, chat completion requests can set dag_id + node_id instead
of a raw priority value; the serving layer computes cp_remaining-based priority
automatically from the registry.

Online LLM estimation flow:
  1. Client calls _estimate_tokens() to get LLM's predicted output length.
  2. Client PATCHes /v1/dags/{dag_id}/nodes/{node_id} with the new estimate.
  3. Registry recomputes cp_remaining (cache cleared).
  4. Client submits the actual completion request with dag_id + node_id.
  5. Server computes priority from the updated cost — no offline profiling needed.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from vllm import dag_registry
from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()


async def _sync_engine(request: Request, dag_id: str) -> None:
    # The router can also be used alone as an in-memory registry in CPU tests.
    if not hasattr(request.app.state, "engine_client"):
        return
    engine = request.app.state.engine_client
    if engine is None:
        raise HTTPException(status_code=503, detail="DAG updates require an engine")
    snapshot = dag_registry.snapshot(dag_id)
    try:
        await asyncio.wait_for(engine.update_dag_priorities(snapshot), timeout=30)
    except Exception as exc:
        logger.exception("Failed to synchronize DAG %s to engine", dag_id)
        raise HTTPException(
            status_code=503,
            detail="DAG engine synchronization failed; retry the update",
        ) from exc


class DAGNode(BaseModel):
    node_id: str
    expected_tokens: int = Field(default=100, ge=1)


class DAGRegistrationRequest(BaseModel):
    dag_id: str
    nodes: list[DAGNode]
    edges: list[tuple[str, str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self):
        nodes = {node.node_id for node in self.nodes}
        if not nodes or len(nodes) != len(self.nodes):
            raise ValueError("DAG nodes must be nonempty and unique")
        successors: dict[str, list[str]] = {node: [] for node in nodes}
        indegree = dict.fromkeys(nodes, 0)
        for src, dst in self.edges:
            if src not in nodes or dst not in nodes:
                raise ValueError("DAG edge references an unknown node")
            successors[src].append(dst)
            indegree[dst] += 1
        frontier = [node for node, degree in indegree.items() if degree == 0]
        visited = 0
        while frontier:
            node = frontier.pop()
            visited += 1
            for successor in successors[node]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    frontier.append(successor)
        if visited != len(nodes):
            raise ValueError("DAG contains a cycle")
        return self


class DAGRegistrationResponse(BaseModel):
    dag_id: str
    num_nodes: int
    registered: bool = True


@router.post("/v1/dags", response_model=DAGRegistrationResponse)
async def register_dag(
    request: DAGRegistrationRequest, http_request: Request
) -> DAGRegistrationResponse:
    nodes = {n.node_id: n.expected_tokens for n in request.nodes}
    dag_registry.register(request.dag_id, nodes, request.edges)
    await _sync_engine(http_request, request.dag_id)
    logger.info("Registered DAG '%s' with %d nodes", request.dag_id, len(nodes))
    return DAGRegistrationResponse(dag_id=request.dag_id, num_nodes=len(nodes))


@router.get("/v1/dags")
async def list_dags() -> dict[str, list[str]]:
    return {"dag_ids": dag_registry.list_dags()}


@router.get("/v1/dags/{dag_id}")
async def get_dag(dag_id: str):
    result = dag_registry.snapshot(dag_id)
    if result is None:
        raise HTTPException(status_code=404, detail="DAG not found")
    return result


@router.post("/v1/dags/{dag_id}/nodes/{node_id}/complete")
async def complete_node(dag_id: str, node_id: str, request: Request):
    # An agent-task node spans multiple Responses requests. Its caller reports
    # task completion explicitly; a single response must not complete the node.
    if not dag_registry.mark_completed(dag_id, node_id):
        raise HTTPException(status_code=404, detail="DAG or node not found")
    await _sync_engine(request, dag_id)
    return {"dag_id": dag_id, "node_id": node_id, "completed": True}


@router.post("/v1/dags/{dag_id}/nodes/{node_id}/reopen")
async def reopen_node(dag_id: str, node_id: str, request: Request):
    if not dag_registry.reopen_node(dag_id, node_id):
        raise HTTPException(status_code=404, detail="DAG or node not found")
    await _sync_engine(request, dag_id)
    return {"dag_id": dag_id, "node_id": node_id, "reopened": True}


class NodeCostUpdate(BaseModel):
    expected_tokens: int = Field(
        ge=1, description="LLM-estimated output token count for this node"
    )


class NodeCostUpdateResponse(BaseModel):
    dag_id: str
    node_id: str
    expected_tokens: int
    updated: bool = True


@router.patch(
    "/v1/dags/{dag_id}/nodes/{node_id}", response_model=NodeCostUpdateResponse
)
async def update_node_cost(
    dag_id: str, node_id: str, body: NodeCostUpdate, request: Request
) -> NodeCostUpdateResponse:
    """
    Update a node's expected token cost with an online LLM self-estimate.
    Clears the cp_remaining cache so the next request for this (dag_id, node_id)
    uses the updated cost when computing scheduling priority.
    """
    updated = dag_registry.update_node_cost(dag_id, node_id, body.expected_tokens)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"DAG '{dag_id}' or node '{node_id}' not found in registry",
        )
    await _sync_engine(request, dag_id)
    logger.info(
        "Updated '%s'/'%s' expected_tokens → %d", dag_id, node_id, body.expected_tokens
    )
    return NodeCostUpdateResponse(
        dag_id=dag_id, node_id=node_id, expected_tokens=body.expected_tokens
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
