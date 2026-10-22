# Codex Responses and dynamic DAG priority integration

This fork increment connects an external Codex Gateway to the native Responses
endpoint and updates queued priorities when model-authored DAG estimates change.
The Gateway and Codex harness are separate components; neither is vendored here.

## Changes

- Responses accepts `dag_id` and `node_id` and derives a two-level priority
  from the registry. Lower `(workflow remaining critical path, negative node
  remaining critical path)` tuples are admitted first under priority scheduling.
- Request conversion and IPC retain DAG identifiers, priority offset and
  metadata through to the core request.
- Graph registration, node-cost updates, completion and reopen operations send
  versioned snapshots to the engine. The HTTP API awaits core acknowledgement;
  synchronization failure is reported rather than acknowledged as successful.
- The core scheduler applies the latest snapshot to matching requests in
  `waiting` and `skipped_waiting`, updates their keys in a batch and rebuilds
  priority heaps while retaining request identity and arrival time. Stale
  snapshots cannot overwrite newer versions. Running requests are not reordered
  or preempted by this operation.
- Optional `VLLM_DAG_SCHEDULE_TRACE=1` records actual scheduler admissions.
- Runtime environments, model files and experiment captures remain ignored.

The engine registry/queue implementation already existed in the fork baseline
`99f494e1da46c2f7541bfb646bded2cd786dccc5`; this contribution extends it.
It does not claim that upstream vLLM has these fork-specific APIs.

## Verification and limits

The five cases in `tests/v1/core/test_dynamic_dag.py` passed, covering both
waiting queues, stale versions, API synchronization failures, heap updates,
wire/request conversion and the prior stale-priority counterexample.
Twelve external Codex-side CPU/protocol tests also passed against this checkout,
including Responses-to-engine priority handoff and Gateway DAG transport.
These checks reused the retained installed development environment and did not
start model inference. The API failure-injection case deliberately logs an IPC
error and asserts the error response.

Earlier controlled GPU queue experiments checked actual admission order for
both tuple levels and dynamic updates. Their raw data are not included here.
This source publication does not constitute a new GPU benchmark or a completed
policy comparison. Token predictions are not calibrated service-time estimates.

Installation follows upstream `uv venv` and `uv pip install -e .`; see
[the environment instructions](dag-environment.md). A fresh dependency download
was stopped at the owner's request. No `uv.lock`, clean-install reproducibility
claim or automatic environment reconstruction guarantee is supplied.

## Source review and disclosure

AI assistance is recorded in `docs/ai_provenance.md`; human review remains blank.
No raw prompts, agent conversations, machine paths, credentials, weights or
virtual environments are included in this increment.

The complete branch history was fetched and scanned with Gitleaks 8.30.1,
including 15,321 non-merge diff commits before this increment. Eight rule
matches were individually reviewed: a documentation bearer-token placeholder,
ordinary parameter expressions in code/docstrings, and a registered model
class/module identifier. No real credentials were identified. Exact reviewed
finding fingerprints are retained in an external audit baseline; credential
rules are not globally disabled and no history is rewritten.

The current README omits the original fork's personal attribution, while
preserving upstream copyright/licensing. Existing public Git history remains
unchanged; this branch is not a history-free anonymous paper artifact.
