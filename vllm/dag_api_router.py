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

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from vllm import dag_registry
from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()


class DAGNode(BaseModel):
    node_id: str
    expected_tokens: int = Field(default=100, ge=1)


class DAGRegistrationRequest(BaseModel):
    dag_id: str
    nodes: list[DAGNode]
    edges: list[tuple[str, str]] = Field(default_factory=list)


class DAGRegistrationResponse(BaseModel):
    dag_id: str
    num_nodes: int
    registered: bool = True


@router.post("/v1/dags", response_model=DAGRegistrationResponse)
async def register_dag(request: DAGRegistrationRequest) -> DAGRegistrationResponse:
    nodes = {n.node_id: n.expected_tokens for n in request.nodes}
    dag_registry.register(request.dag_id, nodes, request.edges)
    logger.info("Registered DAG '%s' with %d nodes", request.dag_id, len(nodes))
    return DAGRegistrationResponse(dag_id=request.dag_id, num_nodes=len(nodes))


@router.get("/v1/dags")
async def list_dags() -> dict[str, list[str]]:
    return {"dag_ids": dag_registry.list_dags()}


class NodeCostUpdate(BaseModel):
    expected_tokens: int = Field(ge=1, description="LLM-estimated output token count for this node")


class NodeCostUpdateResponse(BaseModel):
    dag_id: str
    node_id: str
    expected_tokens: int
    updated: bool = True


@router.patch("/v1/dags/{dag_id}/nodes/{node_id}", response_model=NodeCostUpdateResponse)
async def update_node_cost(
    dag_id: str, node_id: str, body: NodeCostUpdate
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
    logger.info(
        "Updated '%s'/'%s' expected_tokens → %d", dag_id, node_id, body.expected_tokens
    )
    return NodeCostUpdateResponse(
        dag_id=dag_id, node_id=node_id, expected_tokens=body.expected_tokens
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
