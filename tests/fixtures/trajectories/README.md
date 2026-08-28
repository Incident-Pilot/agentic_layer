# Trajectory fixtures — how this was produced

`inc-001-redis-cascade.trajectory.json` is a **real, unedited trajectory
file** produced by actually running the graph end-to-end, not
hand-constructed JSON:

    python -m incident_pilot_agent run inc-001-redis-cascade --llm fake --source fixtures

`--llm fake` (llm/fake_client.py) is deterministic, so this is reproducible
-- re-running the command above regenerates byte-identical `entries`
content modulo timestamps. It exercises the full round-1-rejected /
round-2-confirmed replanning loop against the `inc-001-redis-cascade`
fixture incident (fixtures/incidents/inc-001-redis-cascade/), which is
exactly why this incident was picked for the API's tests
(tests/test_api.py): it has a genuine rejected hypothesis (round 1,
resource-exhaustion red herring) followed by a genuine confirmed one
(round 2, the actual redis-pool-size root cause), so
GET /investigations/{id}'s rejected_hypotheses_count and hypothesis
fields both have real data to assert against.

This lives under tests/fixtures/ (tracked by git) rather than being read
from trajectories/ (gitignored -- see .gitignore -- since that directory
holds developer-local run output, not committed fixtures). If the graph,
the fixture incident's data, or FakeLLMClient's heuristics change in a way
that alters this trajectory's shape, regenerate it with the command above
and copy the result back over this file.

Regenerated 2026-08-27 to add the Remediation Planner's entry (round 2,
`agent: "remediation_planner"`, `phase: "REMEDIATION_PROPOSED"`) now that
the graph routes the CONFIRMED branch through that node -- see
`incident_pilot_agent/agents/remediation_planner.py`.

## `inc-001-redis-cascade-escalated.trajectory.json`

A second **real, unedited trajectory file**, produced the same way but
with the iteration budget forced to 1:

    python -m incident_pilot_agent run inc-001-redis-cascade --llm fake --source fixtures --max-iterations 1

`inc-001-redis-cascade` genuinely rejects in round 1 (the CPU red herring
in FakeLLMClient's first-pass heuristic -- see the module docstring), so
`--max-iterations 1` exhausts the budget before a second round can run,
producing a real ESCALATED run with no `remediation_planner` entry at all
-- exactly the case tests/test_api.py's remediation-plan test needs: proof
that `GET /investigations/{id}` returns `remediation_plan: null` for an
incident that never reached REMEDIATION_PROPOSED. The file is saved under
a `-escalated` suffix (not `inc-001-redis-cascade.trajectory.json` itself)
purely so both fixtures can be read back by distinct incident_ids in the
same test file; the entries' own `incident_id` field still reads
`inc-001-redis-cascade`, which the API ignores in favor of the filename
(see `api/reader.py`).

## `inc-001-redis-cascade-actionable-fields.trajectory.json`

A third **real, unedited trajectory file**, produced the same way as the
first:

    python -m incident_pilot_agent run inc-001-redis-cascade --llm fake --source fixtures

Captured specifically for `HypothesisSummary.causal_chain`/
`affected_services`/`actionable` (api/schemas.py, api/reader.py): those
three fields were added to `TrajectoryEntry`/`TrajectoryLogger.log()`
after the two fixtures above were committed, so re-running the same
command now produces genuine non-empty `hypothesis_causal_chain`/
`hypothesis_affected_services`/`hypothesis_actionable` values on the
synthesizer/verifier entries that the two older fixtures don't carry.
Note `evidence_ids` (and therefore hypothesis/remediation content
downstream of them) are *not* byte-identical across runs despite
`--llm fake` being otherwise deterministic -- evidence ids are minted with
`uuid.uuid4()` -- so this fixture is pinned rather than regenerated
on-the-fly by tests.
