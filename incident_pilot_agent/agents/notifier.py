"""Notifier — sends a plain-text email notification for a CONFIRMED,
actionable hypothesis, right after the remediation plan is proposed. Not
LLM-backed: this node makes exactly one HTTP call (Brevo's REST API,
notifications/brevo_client.py) and never executes anything else.

Only ever reached from graph/build.py's edge after remediation_planner --
i.e. only for a genuinely CONFIRMED, actionable hypothesis that already has
a remediation plan. Skips itself (logged, not raised) when
BREVO_API_KEY/BREVO_SENDER_EMAIL/NOTIFICATION_EMAIL_TO aren't all
configured, and never lets a send failure crash the investigation -- a
notification is an ancillary concern, never a gate on the pipeline
completing.
"""

import logging
from typing import List, Tuple

from .. import config
from ..graph.state import PHASE_REMEDIATION_PROPOSED, AgentState
from ..models.context import IncidentContext
from ..models.hypothesis import Hypothesis
from ..models.remediation import RemediationPlan
from ..notifications.brevo_client import BrevoEmailClient
from ..trajectory.logger import TrajectoryLogger

logger = logging.getLogger(__name__)


def _notification_configured() -> bool:
    return bool(config.BREVO_API_KEY and config.BREVO_SENDER_EMAIL and config.NOTIFICATION_EMAIL_TO)


def _format_email(context: IncidentContext, hypothesis: Hypothesis, plan: RemediationPlan) -> Tuple[str, str]:
    subject = f"[Incident Pilot] Actionable incident: {context.incident_id} -- {context.title}"

    lines: List[str] = [
        f"Incident: {context.incident_id} -- {context.title}",
        f"Severity: {context.severity}",
        f"Affected services: {', '.join(hypothesis.affected_services) or 'unknown'}",
        "",
        f"Root cause (confidence {hypothesis.confidence:.0%}):",
        f"  {hypothesis.root_cause}",
        "",
        "Causal chain:",
    ]
    lines += [f"  - {step}" for step in hypothesis.causal_chain]
    lines += ["", "Proposed remediation:"]
    for action in plan.actions:
        lines += [
            f"  - [{action.risk_level}] {action.action_type}: {action.description} (target: {action.target})",
            f"    rationale: {action.rationale}",
        ]
    lines += ["", plan.disclaimer]

    return subject, "\n".join(lines)


class Notifier:
    def __init__(self, trajectory: TrajectoryLogger):
        self._trajectory = trajectory

    async def __call__(self, state: AgentState) -> dict:
        hypothesis = next(h for h in state["hypotheses"] if h.hypothesis_id == state["current_hypothesis_id"])
        context = state["incident_context"]

        if not _notification_configured():
            self._trajectory.log(
                agent="notifier",
                phase=PHASE_REMEDIATION_PROPOSED,
                reasoning_summary=(
                    "Notification skipped: BREVO_API_KEY/BREVO_SENDER_EMAIL/NOTIFICATION_EMAIL_TO "
                    "not fully configured."
                ),
                hypothesis_id=hypothesis.hypothesis_id,
                round=state["iteration"],
            )
            return {}

        subject, body = _format_email(context, hypothesis, state["remediation_plan"])
        client = BrevoEmailClient(api_key=config.BREVO_API_KEY, sender_email=config.BREVO_SENDER_EMAIL)

        try:
            await client.send_text_email(to_email=config.NOTIFICATION_EMAIL_TO, subject=subject, text_content=body)
            reasoning = f"Sent incident notification email to {config.NOTIFICATION_EMAIL_TO}."
        except Exception:
            # Broad on purpose: a notification is ancillary, so any failure
            # here (BrevoAPIError, or anything else unexpected) is logged
            # and swallowed rather than crashing an otherwise-successful
            # investigation.
            logger.exception("failed to send incident notification email for %s", context.incident_id)
            reasoning = f"Notification email to {config.NOTIFICATION_EMAIL_TO} failed to send -- see logs."

        self._trajectory.log(
            agent="notifier",
            phase=PHASE_REMEDIATION_PROPOSED,
            reasoning_summary=reasoning,
            hypothesis_id=hypothesis.hypothesis_id,
            round=state["iteration"],
        )

        return {}
