"""CLI entrypoint: `python -m incident_pilot_agent run <fixture_incident_id>`

Prints the full trajectory and final root-cause verdict. This is the only
user-facing surface in this phase -- no server, no API, no deployment.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from . import config
from .context_provider.fixture_provider import FixtureContextProvider
from .graph.build import build_graph, finalize_status, initial_state
from .llm.base import LLMClient
from .llm.fake_client import FakeLLMClient
from .telemetry.fixture_backends import FixtureLokiBackend, FixturePrometheusBackend, FixtureTempoBackend
from .tools.loki_tool import LokiTool
from .tools.prometheus_tool import PrometheusTool
from .tools.tempo_tool import TempoTool
from .trajectory.logger import TrajectoryLogger


def _build_llm(name: str) -> LLMClient:
    if name == "fake":
        return FakeLLMClient()
    if name == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            print("error: --llm anthropic requires ANTHROPIC_API_KEY to be set", file=sys.stderr)
            sys.exit(1)
        from .llm.anthropic_client import AnthropicLLMClient

        return AnthropicLLMClient(api_key=config.ANTHROPIC_API_KEY, model=config.ANTHROPIC_MODEL)
    if name == "openai":
        if not config.OPENAI_API_KEY:
            print("error: --llm openai requires OPENAI_API_KEY to be set", file=sys.stderr)
            sys.exit(1)
        from .llm.openai_client import OpenAILLMClient

        return OpenAILLMClient(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)
    raise ValueError(f"unknown --llm {name!r}")


def _default_llm_name() -> str:
    if config.OPENAI_API_KEY:
        return "openai"
    if config.ANTHROPIC_API_KEY:
        return "anthropic"
    return "fake"


def _build_tools(fixtures_dir: Path):
    return [
        PrometheusTool(FixturePrometheusBackend(fixtures_dir)),
        LokiTool(FixtureLokiBackend(fixtures_dir)),
        TempoTool(FixtureTempoBackend(fixtures_dir)),
    ]


async def run_incident(
    incident_id: str,
    *,
    llm_name: str,
    fixtures_root: Path,
    trajectory_dir: Path,
    max_iterations: int,
) -> dict:
    provider = FixtureContextProvider(fixtures_root)
    context = await provider.get_context(incident_id)

    llm = _build_llm(llm_name)
    tools = _build_tools(provider.incident_dir(incident_id))
    trajectory = TrajectoryLogger(incident_id, trajectory_dir)

    graph = build_graph(llm, tools, trajectory)
    result = await graph.ainvoke(initial_state(context, max_iterations=max_iterations))
    result = finalize_status(result)

    _print_trajectory(trajectory)
    _print_verdict(result)
    print(f"\nFull trajectory written to: {trajectory.path}")

    return result


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
    else:
        print("\nNo hypothesis survived verification within the iteration budget. Rejected hypotheses:")
        for rejected in result["rejected_hypotheses"]:
            print(f"  - {rejected.root_cause}")
            print(f"    rejection reason: {rejected.rejection_reason}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="incident_pilot_agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the investigation graph against a fixture incident")
    run_parser.add_argument("incident_id", help="Fixture incident id, e.g. inc-001-redis-cascade")
    run_parser.add_argument("--llm", choices=["fake", "anthropic", "openai"], default=_default_llm_name())
    run_parser.add_argument("--fixtures-dir", type=Path, default=config.DEFAULT_FIXTURES_DIR)
    run_parser.add_argument("--trajectory-dir", type=Path, default=config.DEFAULT_TRAJECTORY_DIR)
    run_parser.add_argument("--max-iterations", type=int, default=config.DEFAULT_MAX_ITERATIONS)

    args = parser.parse_args(argv)

    if args.command == "run":
        asyncio.run(
            run_incident(
                args.incident_id,
                llm_name=args.llm,
                fixtures_root=args.fixtures_dir,
                trajectory_dir=args.trajectory_dir,
                max_iterations=args.max_iterations,
            )
        )


if __name__ == "__main__":
    main()
