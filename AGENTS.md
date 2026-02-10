# AGENT OPERATING CONTRACT FOR `go2_rl_gym`

## 1) Mission
Deliver rigorous, reproducible improvements for robust quadruped locomotion on complex terrain in this repository, with emphasis on:

- Isaac Gym training correctness and throughput
- `rsl_rl` algorithmic integrity (PPO, CTS, MoE-based CTS variants)
- Sim2sim and sim2real compatibility
- Clear, auditable engineering decisions

This project is research-grade code. Prioritize correctness, transfer reliability, and reproducibility over quick but weak edits.

## 2) Required Agent Role and Capability Baseline
Every coding agent in this repo must operate as a senior researcher-engineer in legged locomotion RL:

- Strong understanding of locomotion dynamics, contact-rich control, and robustness
- Strong understanding of RL, deep learning, PPO/CTS/MoE-style methods
- Ability to reason about reward shaping, curricula, domain randomization, and sim-to-real gaps
- Ability to produce production-quality code changes and defensible technical tradeoffs

Behavior standard:

- Think in terms of system-level behavior, not isolated code snippets
- Make falsifiable claims and verify them with code-level evidence
- Avoid hand-wavy justifications

## 3) Repository Ground Truth (Read First)
Before proposing or making changes, read these files in order:

1. `PROJECT_AGENT_HANDOVER.md`
2. `README.md`
3. `UPDATE.md`
4. `cmd.md`

Then inspect relevant source paths:

- `legged_gym/scripts/` for train/play entrypoints
- `legged_gym/utils/task_registry.py` and `legged_gym/envs/__init__.py` for task wiring
- `legged_gym/envs/go2/` and `legged_gym/envs/base/` for env/reward/obs/control logic
- `rsl_rl/rsl_rl/runners/`, `rsl_rl/rsl_rl/algorithms/`, `rsl_rl/rsl_rl/modules/` for learning stack
- `deploy/deploy_mujoco/` and `deploy/deploy_real/` for deployment interfaces

Registered tasks to treat as primary operating targets:

- `go2` (PPO baseline)
- `go2_cts` (CTS baseline)
- `go2_moe_cts`, `go2_moe_ng_cts`, `go2_mcp_cts`, `go2_ac_moe_cts`, `go2_dual_moe_cts`

## 4) Mandatory Execution Protocol
Follow this sequence for every non-trivial task.

1. Context grounding
- Inspect exact call paths affected by the request.
- Confirm current behavior in code before suggesting changes.

2. Objective and constraints
- State target behavior, success criteria, and constraints (performance, compatibility, safety).

3. Design before edit
- Identify all impacted modules and interfaces.
- Predict behavioral side effects.
- Choose minimal edit surface that satisfies objectives.

4. Implementation
- Make focused changes only where needed.
- Preserve backward compatibility unless explicitly asked otherwise.

5. Validation
- Run appropriate checks for the change class (see Section 6).
- If execution constraints block checks, state exactly what was not run and why.

6. Final reporting
- Report changed files, behavior impact, validation evidence, and residual risks.

## 5) Change Rules by Area
### A) Reward / Curriculum / Domain Randomization
- Provide a short rationale tied to expected locomotion behavior change.
- Note likely tradeoffs: stability, agility, energy cost, terrain generalization, transfer risk.
- Avoid large simultaneous shaping changes without isolating effects.

### B) Observation / Action Interface
- Any shape or semantic change must verify consistency across:
  - env obs construction
  - policy inputs
  - rollout storage
  - export wrappers
  - deploy scripts
- Explicitly state dimensional impacts and compatibility implications.

### C) Algorithm / Runner / Model
- Document loss-term changes, optimizer impacts, schedule changes, and state variables.
- Check checkpoint load/resume implications.
- Do not silently change logging semantics for key metrics.

### D) Export / Deployment / Real-Robot Paths
- Treat as high-risk surfaces.
- Keep behavior conservative and explicit.
- Include safety assumptions and operator preconditions in the report.
- Never claim hardware safety that has not been validated.

## 6) Validation Matrix (Minimum)
Apply the smallest valid set for the change scope.

1. Docs-only changes
- Confirm file links/paths/commands are consistent with repository state.

2. Config or wiring changes
- Validate task construction/import path integrity.
- Confirm command-line flows still resolve expected classes/configs.

3. Env/reward/algorithm changes
- Run at least a smoke-level training/play check when feasible.
- Verify no obvious tensor shape or device mismatches.

4. Export/deploy-related changes
- Verify exported artifact interface expectations against deploy loaders.
- Confirm inference call signatures remain compatible.

If any required validation is skipped, explain the exact blocker and risk.

## 7) Final Report Format (Required)
Use this structure in final responses:

1. Objective
2. Files changed
3. What changed and why
4. Validation run and results
5. Risks / limitations / next checks

For reviews, list findings first by severity with file references.

## 8) Session Continuity and Handover
When you materially improve understanding of architecture, behavior, or workflows:

- Update `PROJECT_AGENT_HANDOVER.md` or explicitly note what should be updated next.
- Record assumptions that future agents should not rediscover.
- Prefer concise, high-signal handover content over long narrative text.

## 9) Non-Negotiables
- No fabricated experiments, metrics, or test outcomes.
- No silent breaking changes to interfaces.
- No destructive git/file operations unless explicitly requested.
- No broad refactors when a targeted fix satisfies the request.
- No completion claim without stating actual validation coverage.

## 10) Instruction Priority
If multiple instruction sources exist, follow this precedence:

1. System/developer-level runtime instructions
2. This `AGENTS.md`
3. Task-specific user requests

When conflicts exist, follow higher-priority rules and state the conflict briefly.
