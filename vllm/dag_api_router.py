"""
DAG registration HTTP endpoints for vLLM.

POST /v1/dags  — register a DAG structure (nodes + edges).
GET  /v1/dags  — list registered DAG IDs.

After registration, chat completion requests can set dag_id + node_id instead
of a raw priority value; the serving layer computes cp_remaining-based priority
automatically from the registry.
"""
from __future__ import annotations

from fastapi import APIRouter, FastAPI
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


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
