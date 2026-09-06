"""CLI entrypoint:

    python -m incident_pilot_agent run <incident_id>
    python -m incident_pilot_agent watch

`run` prints the full trajectory and final root-cause verdict for one
incident. `watch` polls the Gateway forever for new ready-for-investigation
incidents and dispatches each through the same pipeline `run` uses -- this
is the only long-running surface in this phase; there is still no server or
API of this repo's own.
"""

import argparse
import asyncio
import logging
import signal
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import httpx
import uvicorn

from . import config
from .api.app import create_app
from .pipeline import (
    _build_context_provider,
    _build_llm,
    _build_node_llms,
    _build_tools,
    _default_llm_name,
    _load_processed_incidents,
    _node_model,
    _node_models_for,
    _save_processed_incidents,
    run_incident,
)
from .trajectory.logger import TrajectoryLogger

logger = logging.getLogger(__name__)


async def _fetch_ready_incident_ids(base_url: str, api_key: str, timeout_seconds: float = 10.0) -> List[str]:
    # The Gateway's GET /incidents only supports filtering by `status`
    # (open/resolved/closed), not by `current_phase` -- ready_for_investigation
    # is a phase, so filtering happens client-side here.
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(
            f"{base_url.rstrip('/')}/incidents", headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()
        payload = response.json()
    return [
        incident["incident_id"]
        for incident in payload.get("incidents", [])
        if incident.get("current_phase") == "ready_for_investigation"
    ]


async def watch_incidents(
    *,
    base_url: str,
    api_key: str,
    poll_interval_seconds: float,
    state_file: Path,
    llm_name: str,
    fixtures_root: Path,
    trajectory_dir: Path,
    max_iterations: int,
    agent_api_host: str,
    agent_api_port: int,
    agent_api_key: Optional[str],
) -> None:
    """Polls the Gateway forever (no webhook -- see gateway_provider.py's
    module docstring for why) for incidents in phase
    ready_for_investigation, and dispatches each new one through the same
    run_incident() pipeline `run` uses, sequentially, one at a time. A
    single incident failing (bad data, LLM error, etc.) is caught and
    logged, never allowed to kill the polling loop.

    Also serves the read-only investigation API (api/app.py) in this same
    process for as long as the polling loop runs -- one process, one thing
    to deploy. Skipped (with a warning) if agent_api_key isn't set, since
    every route but /health would just 401 anyway."""
    processed = _load_processed_incidents(state_file)
    print(f"watch: polling {base_url} every {poll_interval_seconds:.0f}s (already processed: {len(processed)})")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass  # e.g. Windows -- watch is only ever deployed on Linux containers

    async def _poll_loop() -> None:
        while not stop_event.is_set():
            try:
                incident_ids = await _fetch_ready_incident_ids(base_url, api_key)
            except Exception as exc:
                print(f"watch: failed to poll {base_url}/incidents: {exc}", file=sys.stderr)
                incident_ids = []

            for incident_id in incident_ids:
                if stop_event.is_set():
                    break
                if incident_id in processed:
                    continue

                print(f"watch: dispatching new incident {incident_id}")
                try:
                    await run_incident(
                        incident_id,
                        llm_name=llm_name,
                        fixtures_root=fixtures_root,
                        trajectory_dir=trajectory_dir,
                        max_iterations=max_iterations,
                        source="gateway",
                    )
                except Exception as exc:
                    print(
                        f"watch: investigation failed for {incident_id}: "
                        f"{type(exc).__name__}: {exc!r}",
                        file=sys.stderr,
                    )
                    logger.debug("full traceback for %s:\n%s", incident_id, traceback.format_exc())
                finally:
                    # Marked processed even on failure -- a bad incident should
                    # not be retried forever on every poll cycle.
                    processed.add(incident_id)
                    _save_processed_incidents(state_file, processed)

            if stop_event.is_set():
                break
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_seconds)
            except asyncio.TimeoutError:
                pass

        print("watch: received shutdown signal, exiting cleanly")

    if not agent_api_key:
        print(
            "watch: AGENT_API_KEY is not set -- the investigation API will not be started "
            "(polling continues normally)",
            file=sys.stderr,
        )
        await _poll_loop()
        return

    app = create_app(
        trajectory_dir,
        agent_api_key,
        llm_name=llm_name,
        fixtures_root=fixtures_root,
        max_iterations=max_iterations,
        state_file=state_file,
        # The *same* set object _poll_loop's closure above holds a
        # reference to, not a copy -- see create_app's docstring. A manual
        # trigger's `processed.add(incident_id)` is then immediately
        # visible to _poll_loop's own `if incident_id in processed` check
        # on its very next iteration, no reload-from-disk needed.
        processed_incidents=processed,
    )
    server = uvicorn.Server(uvicorn.Config(app, host=agent_api_host, port=agent_api_port, log_level="warning"))
    print(f"watch: serving investigation API on http://{agent_api_host}:{agent_api_port}")

    async def _stop_server_on_shutdown() -> None:
        await stop_event.wait()
        server.should_exit = True

    await asyncio.gather(_poll_loop(), server.serve(), _stop_server_on_shutdown())


def _print_trajectory(trajectory: TrajectoryLogger) -> None:
    print(f"\n=== Trajectory: {trajectory.path.stem} ===")
    for entry in trajectory.entries:
        tool_summary = ", ".join(f"{tc.tool_name}({'ok' if tc.ok else 'FAIL'})" for tc in entry.tool_calls)
        line = f"[{entry.sequence:02d}] round={entry.round} {entry.agent:<12} phase={entry.phase}"
        if tool_summary:
            line += f" tools=[{tool_summary}]"
        if entry.verification_verdict:
            line += f" verdict={entry.verification_verdict}"
        print(line)
        print(f"     {entry.reasoning_summary}")


def _print_verdict(result: dict) -> None:
    print("\n=== Final verdict ===")
    print(f"status: {result['final_status']}")
    print(f"phase:  {result['phase']}")
    print(f"rounds: {result['iteration']}")

    if result["final_status"] == "CONFIRMED":
        hyp = next(h for h in result["hypotheses"] if h.hypothesis_id == result["current_hypothesis_id"])
        print(f"\nRoot cause: {hyp.root_cause}")
        print(f"Confidence: {hyp.confidence:.2f}")
        print("Causal chain:")
        for step in hyp.causal_chain:
            print(f"  - {step}")
        print(f"Affected services: {', '.join(hyp.affected_services)}")
        print(f"Actionable: {hyp.actionable}")

        plan = result.get("remediation_plan")
        if plan is not None:
            print(f"\n=== Remediation plan ({plan.status}) ===")
            for action in plan.actions:
                print(f"  - [{action.risk_level}] {action.action_type}: {action.description} (target: {action.target})")
                print(f"    rationale: {action.rationale}")
            print(f"\n{plan.disclaimer}")

        report = result.get("postmortem_report")
        if report is not None:
            print("\n=== Post-mortem ===")
            print(f"Summary: {report.summary}")
            print(f"Impact:  {report.impact}")
            if report.contributing_factors:
                print("Contributing factors:")
                for factor in report.contributing_factors:
                    print(f"  - {factor}")
            if report.action_items:
                print("Follow-up action items:")
                for item in report.action_items:
                    print(f"  - [{item.priority}/{item.category}] {item.description}")
            if report.lessons_learned:
                print("Lessons learned:")
                for lesson in report.lessons_learned:
                    print(f"  - {lesson}")
    else:
        print("\nNo hypothesis survived verification within the iteration budget. Rejected hypotheses:")
        for rejected in result["rejected_hypotheses"]:
            print(f"  - {rejected.root_cause}")
            print(f"    rejection reason: {rejected.rejection_reason}")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="incident_pilot_agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the investigation graph against a fixture incident")
    run_parser.add_argument("incident_id", help="Fixture incident id, e.g. inc-001-redis-cascade")
    run_parser.add_argument(
        "--llm", choices=["fake", "anthropic", "openai", "gemini", "openrouter", "bedrock"], default=_default_llm_name()
    )
    run_parser.add_argument("--fixtures-dir", type=Path, default=config.DEFAULT_FIXTURES_DIR)
    run_parser.add_argument("--trajectory-dir", type=Path, default=config.DEFAULT_TRAJECTORY_DIR)
    run_parser.add_argument("--max-iterations", type=int, default=config.DEFAULT_MAX_ITERATIONS)
    run_parser.add_argument(
        "--source",
        choices=["auto", "fixtures", "gateway"],
        default="auto",
        help="Where to load IncidentContext from. 'auto' (default) uses the Gateway only when "
        "incident_id doesn't match a local fixture.",
    )

    watch_parser = subparsers.add_parser(
        "watch",
        help="Poll the Gateway forever for new ready-for-investigation incidents and investigate each one",
    )
    watch_parser.add_argument(
        "--llm", choices=["fake", "anthropic", "openai", "gemini", "openrouter", "bedrock"], default=_default_llm_name()
    )
    watch_parser.add_argument("--fixtures-dir", type=Path, default=config.DEFAULT_FIXTURES_DIR)
    watch_parser.add_argument("--trajectory-dir", type=Path, default=config.DEFAULT_TRAJECTORY_DIR)
    watch_parser.add_argument("--max-iterations", type=int, default=config.DEFAULT_MAX_ITERATIONS)
    watch_parser.add_argument("--poll-interval", type=float, default=config.DEFAULT_WATCH_POLL_INTERVAL_SECONDS)
    watch_parser.add_argument("--state-file", type=Path, default=config.DEFAULT_PROCESSED_INCIDENTS_FILE)
    watch_parser.add_argument("--api-host", default=config.AGENT_API_HOST)
    watch_parser.add_argument("--api-port", type=int, default=config.AGENT_API_PORT)

    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            result, trajectory = asyncio.run(
                run_incident(
                    args.incident_id,
                    llm_name=args.llm,
                    fixtures_root=args.fixtures_dir,
                    trajectory_dir=args.trajectory_dir,
                    max_iterations=args.max_iterations,
                    source=args.source,
                )
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        _print_trajectory(trajectory)
        _print_verdict(result)
        print(f"\nFull trajectory written to: {trajectory.path}")
    elif args.command == "watch":
        if not config.INCIDENT_GATEWAY_URL or not config.INCIDENT_GATEWAY_API_KEY:
            print(
                "error: watch requires INCIDENT_GATEWAY_URL and INCIDENT_GATEWAY_API_KEY to be set",
                file=sys.stderr,
            )
            sys.exit(1)
        asyncio.run(
            watch_incidents(
                base_url=config.INCIDENT_GATEWAY_URL,
                api_key=config.INCIDENT_GATEWAY_API_KEY,
                poll_interval_seconds=args.poll_interval,
                state_file=args.state_file,
                llm_name=args.llm,
                fixtures_root=args.fixtures_dir,
                trajectory_dir=args.trajectory_dir,
                max_iterations=args.max_iterations,
                agent_api_host=args.api_host,
                agent_api_port=args.api_port,
                agent_api_key=config.AGENT_API_KEY,
            )
        )


if __name__ == "__main__":
    main()
