# KairoScale Swarm Brief

Goal: Implement KairoScale MVP pipeline: Profile → Analyze → Validate → Report.

Non-goals (MVP): multi-GPU comm profiling, HTML report, exhaustive loop detection.

Hard requirements:
- All optimization suggestions must cite profile evidence.
- Validation must run control + variants; detect divergence (grad cos sim + loss rules).
- Sandbox isolation + cost/time ceilings.

Source of truth for interfaces: swarm/INTERFACES.md
Decisions log: swarm/DECISIONS.md
Daily status: swarm/STATUS.md