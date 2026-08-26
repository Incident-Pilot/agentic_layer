"""FakeLLMClient — deterministic, offline stand-in for a real LLM.

Used for automated tests and `--llm fake` CLI runs so the full graph --
including the replanning/verification loop -- can be exercised without
network access or an API key (per top-level spec section 9: prefer
deterministic logic where a deterministic answer is possible; this makes
that testable end-to-end offline).

NOT a general-purpose LLM simulator. It is tightly coupled to this repo's
own prompt conventions (the `Task: <slug>` marker and `json_block` CONTEXT
payloads in agents/prompts.py) and implements simple, documented,
non-fixture-specific heuristics per task:

- investigate-tools / verify-tools: a fixed query plan derived from the
  CONTEXT payload (metric names, affected services, an "uncited
  deployment" hint), issued one tool call per turn.
- synthesize-hypothesis: prefers a deployment-correlated k8s crash-loop
  explanation if one exists; otherwise ranks metric evidence by "is it
  rising" then a generic signal-priority tiebreak (cpu > latency >
  error_rate > memory) -- the same naive "loudest signal first" mistake a
  first-pass investigator might make. On a replanning round it instead
  builds a hypothesis around the evidence a rejection surfaced.
- verify-decide: REJECTED if independently-gathered evidence corroborates
  an uncited deployment; REJECTED also if there is neither resolved
  supporting_evidence nor any newly-gathered evidence to confirm against
  (silence is not grounds for confirmation); CONFIRMED otherwise.

A real LLMClient (AnthropicLLMClient) makes all of these judgment calls
via genuine reasoning instead.
"""

import json
import re
from typing import Any, Dict, List

from .base import LLMClient, LLMMessage, LLMResponse, ToolCallRequest

_TASK_RE = re.compile(r"^Task:\s*(\S+)", re.MULTILINE)
_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)

_METRIC_PRIORITY = ["cpu_usage_seconds", "latency_ms", "http_error_rate", "memory_working_set_bytes"]


def _extract_task(system: str) -> str:
    match = _TASK_RE.search(system)
    if not match:
        raise ValueError("FakeLLMClient requires a 'Task: <slug>' marker in the system prompt")
    return match.group(1)


def _extract_json(text: str) -> Dict[str, Any]:
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError("FakeLLMClient could not find a ```json fenced CONTEXT block to parse")
    return json.loads(match.group(1))


def _last_user_text(messages: List[LLMMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            for block in message.content:
                if block.get("type") == "text":
                    return block["text"]
    return ""


class FakeLLMClient(LLMClient):
    def __init__(self, refuse_tool_calls_once: bool = False, refuse_all_tool_calls: bool = False):
        """Both params exist only to test run_tool_loop's
        require_at_least_one_call retry path (see agents/tool_loop.py) --
        a real LLMClient just decides this on its own.

        refuse_tool_calls_once: the very first investigate-tools/
        verify-tools response comes back with no tool calls (simulating a
        model that stops immediately), then behaves normally (real tool
        calls, per the deterministic plan) on every call after that --
        i.e. it complies once nudged by the retry.

        refuse_all_tool_calls: every investigate-tools/verify-tools
        response comes back with no tool calls, including after a retry
        nudge -- simulates a model that never complies, so the retry is
        exhausted and the loop must give up cleanly.
        """
        self._refuse_tool_calls_once = refuse_tool_calls_once
        self._refuse_all_tool_calls = refuse_all_tool_calls
        self._has_refused = False

    async def complete(
        self, *, system: str, messages: List[LLMMessage], tools=None, max_tokens: int = 1024
    ) -> LLMResponse:
        task = _extract_task(system)

        if task in ("investigate-tools", "verify-tools"):
            should_refuse = self._refuse_all_tool_calls or (
                self._refuse_tool_calls_once and not self._has_refused and len(messages) == 1
            )
            if should_refuse:
                self._has_refused = True
                return LLMResponse(content="Nothing further needed.", tool_calls=[], stop_reason="end_turn")
            plan = self._plan_investigate(_extract_json(system)) if task == "investigate-tools" else self._plan_verify(
                _extract_json(system)
            )
            return self._next_tool_call(messages, plan)
        if task == "synthesize-hypothesis":
            return self._synthesize_hypothesis(_extract_json(_last_user_text(messages)))
        if task == "verify-decide":
            return self._verify_decide(_extract_json(_last_user_text(messages)))

        raise ValueError(f"FakeLLMClient: unrecognized task {task!r}")

    # -- tool-call planning --------------------------------------------------

    @staticmethod
    def _plan_investigate(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        namespace = ctx["namespace"]
        start, end = ctx["window_start"], ctx["window_end"]
        plan: List[Dict[str, Any]] = []
        for metric_name in ctx.get("metric_names", []):
            plan.append(
                {
                    "name": "query_prometheus",
                    "input": {"promql": f'{metric_name}{{namespace="{namespace}"}}', "start": start, "end": end, "step": "30s"},
                }
            )
        plan.append(
            {
                "name": "query_loki",
                "input": {
                    "logql": f'{{namespace="{namespace}"}} |~ "(?i)error|exception|timeout|fail"',
                    "start": start,
                    "end": end,
                    "limit": 100,
                },
            }
        )
        for service in (ctx.get("affected_services") or [])[:2]:
            plan.append(
                {
                    "name": "query_tempo",
                    "input": {"operation": "search", "service_name": service, "start": start, "end": end, "limit": 10},
                }
            )
        return plan

    @staticmethod
    def _plan_verify(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        namespace = ctx["namespace"]
        start, end = ctx["window_start"], ctx["window_end"]
        uncited = ctx.get("uncited_deployment")
        if uncited:
            keywords = ctx.get("keywords") or ["error"]
            pattern = "|".join(keywords)
            return [
                {
                    "name": "query_loki",
                    "input": {"logql": f'{{namespace="{namespace}"}} |~ "(?i){pattern}"', "start": start, "end": end, "limit": 50},
                }
            ]
        services = ctx.get("affected_services") or ["unknown"]
        return [
            {
                "name": "query_tempo",
                "input": {"operation": "search", "service_name": services[0], "start": start, "end": end, "limit": 10},
            }
        ]

    @staticmethod
    def _next_tool_call(messages: List[LLMMessage], plan: List[Dict[str, Any]]) -> LLMResponse:
        completed = (len(messages) - 1) // 2
        if completed < len(plan):
            step = plan[completed]
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id=f"fake-call-{completed}", name=step["name"], input=step["input"])],
                stop_reason="tool_use",
            )
        return LLMResponse(content="Gathering complete.", tool_calls=[], stop_reason="end_turn")

    # -- structured judgment calls -------------------------------------------

    @staticmethod
    def _synthesize_hypothesis(payload: Dict[str, Any]) -> LLMResponse:
        evidence = payload.get("evidence", [])
        rejected = set(payload.get("rejected_hypotheses", []))
        deployments = payload.get("recent_deployments", [])
        k8s_events = payload.get("k8s_events", [])

        def respond(root_cause, chain, services, supporting, confidence) -> LLMResponse:
            return LLMResponse(
                content=json.dumps(
                    {
                        "root_cause": root_cause,
                        "causal_chain": chain,
                        "affected_services": services,
                        "confidence": confidence,
                        "supporting_evidence_ids": supporting,
                    }
                )
            )

        # 1. crash-loop shortcut: a k8s crash/backoff event directly
        # attributable to a recent deployment on the same service is not a
        # subtle signal -- a competent first pass should catch it.
        for event in k8s_events:
            reason = (event.get("reason") or "").lower()
            if not any(marker in reason for marker in ("crashloop", "oomkill", "backoff")):
                continue
            object_ref = event.get("object_ref", "")
            dep = next((d for d in deployments if d["service"] in object_ref or object_ref in d["service"]), None)
            if dep is None:
                continue
            change = dep.get("change_summary") or dep.get("commit_sha") or "recent change"
            root_cause = f"{dep['service']} deployment ({change}) caused {event.get('reason')} on {object_ref}"
            if root_cause in rejected:
                continue
            chain = [
                f"{dep['service']} deployed at {dep.get('deployed_at')}: {change}",
                f"{event.get('reason')} observed on {object_ref}: {event.get('message')}",
            ]
            supporting = [e["evidence_id"] for e in evidence if e.get("service") in (object_ref, dep["service"])]
            return respond(root_cause, chain, [dep["service"]], supporting, 0.9)

        # 2. rank metric evidence: rising trend first, then a generic
        # signal-priority tiebreak among simultaneously-rising metrics.
        def metric_name_of(e: Dict[str, Any]) -> str:
            return e["summary"].split(" for ")[0].strip().lower() if " for " in e["summary"] else ""

        def priority(e: Dict[str, Any]) -> int:
            name = metric_name_of(e)
            return _METRIC_PRIORITY.index(name) if name in _METRIC_PRIORITY else len(_METRIC_PRIORITY)

        metric_evidence = [e for e in evidence if e.get("type") == "metric"]
        rising = sorted((e for e in metric_evidence if "rising" in e["summary"].lower()), key=priority)
        services_in_evidence = sorted({e["service"] for e in evidence if e.get("service")})

        # 3. replanning round: build a hypothesis around what the
        # rejection surfaced (a deployment plus whatever new log evidence
        # verification's targeted query found).
        if rejected and deployments:
            dep = deployments[0]
            log_evidence = [e for e in evidence if e.get("type") == "log"]
            change = dep.get("change_summary") or dep.get("commit_sha")
            root_cause = (
                f"{dep['service']} root cause: {change} triggered cascading failures across "
                f"{', '.join(services_in_evidence) or dep['service']}"
            )
            if root_cause not in rejected:
                chain = [f"{dep['service']} deployed at {dep.get('deployed_at')}: {change}"]
                chain += [e["summary"] for e in log_evidence[:2]]
                chain += [e["summary"] for e in rising[:2]]
                supporting = [e["evidence_id"] for e in evidence if e.get("service") == dep["service"]]
                supporting += [e["evidence_id"] for e in log_evidence[:3]]
                supporting += [e["evidence_id"] for e in rising[:2]]
                return respond(
                    root_cause, chain, services_in_evidence or [dep["service"]], list(dict.fromkeys(supporting)), 0.88
                )

        if rising:
            top = rising[0]
            root_cause = f"{top.get('service')} resource exhaustion ({top['summary']}) is the root cause"
            if root_cause not in rejected:
                return respond(
                    root_cause, [top["summary"]], [top["service"]] if top.get("service") else [], [top["evidence_id"]], 0.55
                )

        if evidence:
            ev = evidence[0]
            return respond(
                f"Unclear root cause; leading indicator: {ev['summary']}",
                [ev["summary"]],
                [ev["service"]] if ev.get("service") else [],
                [ev["evidence_id"]],
                0.3,
            )

        return respond("Insufficient evidence to determine root cause", [], [], [], 0.1)

    @staticmethod
    def _verify_decide(payload: Dict[str, Any]) -> LLMResponse:
        uncited = payload.get("uncited_deployment")
        new_evidence = payload.get("newly_gathered_evidence", [])
        supporting_evidence = payload.get("supporting_evidence", [])

        if uncited and new_evidence:
            reasoning = (
                f"Independent targeted query surfaced {len(new_evidence)} new evidence item(s) tied to an "
                f"uncited deployment on {uncited['service']} ({uncited.get('change_summary')}) deployed at "
                f"{uncited.get('deployed_at')}, which better explains the incident than the current hypothesis."
            )
            return LLMResponse(
                content=json.dumps(
                    {
                        "verdict": "REJECTED",
                        "reasoning_summary": reasoning,
                        "counter_evidence_ids": [e["evidence_id"] for e in new_evidence],
                    }
                )
            )

        if not supporting_evidence and not new_evidence:
            reasoning = (
                "No original supporting evidence resolved for this hypothesis and the independent "
                "targeted query found nothing new -- there is nothing to confirm against."
            )
            return LLMResponse(content=json.dumps({"verdict": "REJECTED", "reasoning_summary": reasoning, "counter_evidence_ids": []}))

        reasoning = (
            "Original supporting evidence substantiates the hypothesis and the independent check found no "
            "contradicting evidence."
            if not uncited
            else "An earlier deployment exists but the targeted query found no corroborating evidence; hypothesis stands."
        )
        return LLMResponse(content=json.dumps({"verdict": "CONFIRMED", "reasoning_summary": reasoning, "counter_evidence_ids": []}))
