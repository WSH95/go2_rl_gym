# Project Agent Handover (Go2 RL Gym)

## 1. Project Mission and Research Context

This repository implements robust quadruped locomotion for Unitree Go2 across complex terrains using deep reinforcement learning.

The codebase is centered on:

- Isaac Gym simulation for large-scale parallel training.
- A rewritten `rsl_rl` stack that includes PPO, CTS, and multiple MoE-based CTS variants.
- A sim-to-sim and sim-to-real workflow aligned with the paper:
  `Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion` (arXiv:2602.00678, Jan 31, 2026).

Paper-to-code mapping in this repo:

- MoE policy variants: `go2_moe_cts`, `go2_moe_ng_cts`, `go2_ac_moe_cts`, `go2_dual_moe_cts`, `go2_mcp_cts`.
- CTS baseline: `go2_cts`.
- PPO baseline: `go2`.
- RoboGauge training-loop integration is implemented in both runner families (`OnPolicyRunner`, `OnPolicyRunnerCTS`) via async client hooks.

## 2. Repository Architecture Map

| Path | Responsibility | Typical edits |
|---|---|---|
| `legged_gym/scripts/train.py` | Main training entrypoint | Task selection, run command flow |
| `legged_gym/scripts/play.py` | In-sim evaluation + policy export | Export behavior, evaluation overrides |
| `legged_gym/utils/task_registry.py` | Task registration, env/runner construction, resume loading | New tasks, runner wiring |
| `legged_gym/envs/__init__.py` | Task name to config mapping | Register/remove tasks |
| `legged_gym/envs/go2/go2_config.py` | Go2 env + train config specializations | Rewards, DR, curricula, experiment names |
| `legged_gym/envs/go2/go2_env.py` | Go2 observation/reward custom terms | Obs layout, privileged obs, custom rewards |
| `legged_gym/envs/base/legged_robot.py` | Base env loop, reward/curriculum mechanics | Shared env behavior |
| `legged_gym/envs/base/legged_robot_config.py` | Base config definitions for PPO/CTS/MoE variants | Default algorithm/env knobs |
| `legged_gym/utils/terrain.py` | Terrain generator and curriculum grid | Terrain type, difficulty schedule |
| `rsl_rl/rsl_rl/runners/on_policy_runner.py` | PPO runner + logging/saving/RoboGauge | PPO train loop behavior |
| `rsl_rl/rsl_rl/runners/on_policy_runner_cts.py` | CTS/MoE runner + history handling | CTS train loop behavior |
| `rsl_rl/rsl_rl/algorithms/*.py` | PPO/CTS/MoE algorithm updates | Losses, optimization schedule |
| `rsl_rl/rsl_rl/modules/*.py` | Actor-critic and encoder architectures | Model design |
| `legged_gym/utils/exporter.py` | JIT/ONNX/PKL export wrappers | Deployment export compatibility |
| `deploy/deploy_mujoco/*` | Sim2Sim inference scripts/config | Deployment observation/control sync |
| `deploy/deploy_real/*` | Real robot deployment scripts/config | SDK interface, control loop |
| `resources/robots/go2/*` | URDF/XML/assets | Robot and terrain assets |
| `tools/*` | Log utilities | Post-processing experiments |
| `UPDATE.md` | Experimental timeline + critical change history | Regression context |

## 3. End-to-End Execution Flows

### 3.1 Training flow

1. `python legged_gym/scripts/train.py --task=<task>`
2. `task_registry.make_env(...)` builds `Go2Robot` with `GO2Cfg`-derived env cfg.
3. `task_registry.make_alg_runner(...)` instantiates runner class by string:
   `OnPolicyRunner` or `OnPolicyRunnerCTS`.
4. Runner instantiates policy and algorithm class from config strings.
5. Training loop:
   rollout -> return computation -> optimization update -> logging -> checkpoint save.
6. `train.py` additionally:
   sets `env.common_step_counter` on resume and forces reward curriculum update once at startup.

### 3.2 Play/evaluation and export flow

1. `python legged_gym/scripts/play.py --task=<task>`
2. Script overrides env for testing (reduced env count, disables most randomization/noise).
3. Loads latest/specified checkpoint by resume path logic.
4. Runs policy in Isaac Gym.
5. Exports policy to:
   `logs/<experiment>/exported/policies/policy.pt`, `policy.onnx`, `policy.pkl`.

### 3.3 Sim2Sim (Mujoco) flow

1. `python deploy/deploy_mujoco/deploy_go2.py`
2. Loads config from `deploy/deploy_mujoco/configs/go2.yaml`.
3. Builds observation layout consistent with training obs (45-dim).
4. Runs JIT policy with PD control on Mujoco model.
5. Optional gamepad command input and optional video recording.

### 3.4 Sim2Real flow

1. `python deploy/deploy_real/deploy_real_go2.py <network_interface>`
2. Loads config from `deploy/deploy_real/configs/go2.yaml`.
3. Uses `unitree_sdk2py` low-level channels.
4. Control state machine:
   zero torque -> move to default pose -> default pose hold -> running loop.
5. Running loop mirrors training observation semantics and applies PD targets from policy outputs.

## 4. Task and Algorithm Matrix (go2, go2_cts, go2_moe_cts, ...)

Registered in `legged_gym/envs/__init__.py`.

| `--task` | Train cfg class | Runner | Policy class | Algorithm class | Default experiment name |
|---|---|---|---|---|---|
| `go2` | `GO2CfgPPO` | `OnPolicyRunner` | `ActorCritic` | `PPO` | `go2_ppo` |
| `go2_cts` | `GO2CfgCTS` | `OnPolicyRunnerCTS` | `ActorCriticCTS` | `CTS` | `go2_cts` |
| `go2_moe_cts` | `GO2CfgMoECTS` | `OnPolicyRunnerCTS` | `ActorCriticMoECTS` | `MoECTS` | `go2_moe_cts` |
| `go2_moe_ng_cts` | `GO2CfgMoENGCTS` | `OnPolicyRunnerCTS` | `ActorCriticMoENGCTS` | `MoENGCTS` | `go2_moe_no_goal_cts` |
| `go2_mcp_cts` | `GO2CfgMCPCTS` | `OnPolicyRunnerCTS` | `ActorCriticMCPCTS` | `MCPCTS` | `go2_mcp_cts` |
| `go2_ac_moe_cts` | `GO2CfgACMoECTS` | `OnPolicyRunnerCTS` | `ActorCriticACMoECTS` | `ACMoECTS` | `go2_ac_moe_cts` |
| `go2_dual_moe_cts` | `GO2CfgDualMoECTS` | `OnPolicyRunnerCTS` | `ActorCriticDualMoECTS` | `DualMoECTS` | `go2_dual_moe_cts` |

Notes:

- All registered tasks currently use `GO2Cfg()` for env settings and differ mainly in train algorithm/policy config.
- Several extra config files exist (`go2_config_vanilla*.py`, `go2_config_fast_flat_move.py`) but are not registered by default tasks.

## 5. Environment, Observation, Action, Reward, Curriculum

### 5.1 Observation and privileged observation

`go2_env.py` observation (45):

- Base angular velocity (3)
- Projected gravity (3)
- Commands `[vx, vy, yaw_rate]` (3)
- Joint position error to default (12)
- Joint velocity (12)
- Previous action (12)

Privileged observation (263 in `GO2Cfg`):

- Includes all actor observation content plus:
- Base linear velocity (3)
- Foot contact forces (4)
- Motor torques normalized by limits (12)
- Motor acceleration estimate (12)
- Terrain height measurements (187)

### 5.2 Control and timing

- PD position control (`control_type = 'P'`)
- `action_scale = 0.25`
- `decimation = 4`
- Physics `dt = 0.005` (`legged_robot_config.py`) -> policy/control step 50 Hz.

### 5.3 Terrain and curriculum

- Terrain system in `legged_gym/utils/terrain.py`.
- 9 terrain categories: wave, slope, rough_slope, stairs_up, stairs_down, obstacles, stepping_stones, gap, flat.
- `IS_HARD = True` in terrain generation currently hard-codes harder difficulty profile.
- Go2 config uses curriculum + custom terrain proportions (bias toward slope/stairs/obstacles plus flat portion).

### 5.4 Command curriculum and sampling

Go2 config includes:

- Dynamic resampling (`dynamic_resample_commands = True`)
- Zero-command curriculum ramp (`zero_command_curriculum`)
- Velocity limit sampling probability (`limit_vel_prob`)
- Command-range curriculum at specific iterations (`iter = 20000`, `50000`)
- Terrain-specific max command ranges.

### 5.5 Domain randomization

Go2 defaults enable broad DR:

- Friction, base mass, link mass, base COM, restitution
- PD gains, motor zero offset, motor strength
- External pushes
- Random action delay.

### 5.6 Reward shaping highlights

In `GO2Cfg.rewards`:

- Core tracking (`tracking_lin_vel`, `tracking_ang_vel`)
- Stability/effort penalties (`lin_vel_z`, `ang_vel_xy`, `torques`, `dof_acc`, `dof_power`)
- Contact/collision and joint-limit terms
- `correct_base_height` curriculum scaling
- `hip_to_default` regularizer added for sim-to-real gait quality.

## 6. MoE/CTS Implementation Details in This Codebase

### 6.1 CTS base mechanism

- `history_length = 5` stacked actor observations.
- Teacher uses privileged observations; student uses observation history.
- Parallel teacher/student env partitions controlled by `teacher_env_ratio` (default `0.75`).
- Two optimizers:
  actor/critic/teacher branch and student encoder branch.
- Student latent distillation loss is MSE to teacher latent.

### 6.2 MoE variants

- `MoECTS`: MoE student encoder + load-balance regularization.
- `MoENGCTS` (MoE-NG): MoE student encoder with goal-masked branch (`obs_no_goal_mask`) and load-balance term.
- `MCPCTS`: student encoder + MCP actor path (`actor_mcp`) with latent distillation.
- `ACMoECTS`: actor-critic MoE path with actor load-balance + CTS-style student distillation.
- `DualMoECTS`: both student and actor use MoE; tracks both student and actor load-balance losses.

### 6.3 Export and deployment behavior for CTS/MoE

`exporter.py` handles variant-specific forward signatures. For many CTS/MoE exports, JIT forward returns tuples like:

- CTS: `action, (None, latent)`
- MoE-style variants: `action, (weights, latent)` or variant-specific tuple variants.

Deploy scripts already check `isinstance(result, tuple)` and use first output as action.

## 7. Training, Resume, Evaluation, Export, Deployment Commands

### 7.1 Training

```bash
python legged_gym/scripts/train.py --task=go2_moe_cts --headless --num_envs=8192
python legged_gym/scripts/train.py --task=go2_cts --headless --num_envs=8192
python legged_gym/scripts/train.py --task=go2 --headless --num_envs=8192
```

### 7.2 Resume

```bash
python legged_gym/scripts/train.py --task=go2_moe_cts --resume --experiment_name=go2_moe_cts --load_run=<run_dir> --checkpoint=<iter>
```

### 7.3 Play + export

```bash
python legged_gym/scripts/play.py --task=go2_moe_cts --num_envs=64
```

### 7.4 RoboGauge-enabled training

```bash
python legged_gym/scripts/train.py --task=go2_moe_cts --headless --robogauge --robogauge_port=9973
```

### 7.5 Sim2Sim and Sim2Real

```bash
python deploy/deploy_mujoco/deploy_go2.py
python deploy/deploy_mujoco/deploy_go2_moe.py
python deploy/deploy_real/deploy_real_go2.py eth0
```

## 8. Config and Logging Conventions

- Logs root:
  `logs/<experiment_name>/<timestamp>_<run_name>/`
- Checkpoints:
  `model_<iteration>.pt`
- Per-run frozen config:
  `config.yaml` with both `train_cfg` and `env_cfg`.
- Optional RoboGauge outputs:
  `robogauge_results/results_<step>.yaml`
- JIT exports (for RoboGauge eval loop):
  `jit_models/policy_jit_<step>.pt`

Resume path behavior:

- Implemented in `get_load_path(...)` (`helpers.py`).
- `load_run = -1` and `checkpoint = -1` choose latest run/checkpoint automatically.

## 9. Known Pitfalls and High-Risk Areas

1. `task_registry` uses `eval(...)` on class-name strings for runner/policy/algorithm classes.
   Typos in config class strings fail at runtime.
2. `task_registry.get_cfgs(...)` returns stored config objects (not deep copies).
   In-process mutations can persist across calls in the same Python process.
3. Terrain generator has `IS_HARD = True` hard-coded in `terrain.py`.
   Results may differ strongly from earlier “easy” assumptions.
4. `play.py` force-sets `FIX_COMMAND = True` and disables many randomization terms.
   Do not treat play behavior as representative training DR behavior.
5. `UPDATE.md` history shows frequent naming changes (`rem_cts` -> `moe_cts`, old `moe_cts` -> `moe_no_goal_cts`).
   Always verify current task names in `legged_gym/envs/__init__.py`.
6. Last-model RoboGauge loop can block waiting for results by design.
   Treat this as expected if RoboGauge server is enabled.
7. There is historical mismatch between claimed max-iteration guidance and current config defaults.
   Current active config classes should be treated as source of truth.

## 10. Fast Start for Future Agents (First 30 Minutes)

1. Read `PROJECT_AGENT_HANDOVER.md` fully once.
2. Confirm task wiring:
   `legged_gym/envs/__init__.py`, `legged_gym/utils/task_registry.py`.
3. Inspect active Go2 config:
   `legged_gym/envs/go2/go2_config.py`.
4. Inspect env observation/reward implementation:
   `legged_gym/envs/go2/go2_env.py`, `legged_gym/envs/base/legged_robot.py`.
5. Inspect algorithm selected by target task:
   `rsl_rl/rsl_rl/algorithms/<variant>.py`, plus matching module in `rsl_rl/rsl_rl/modules/`.
6. Run quick local sanity:
   `python legged_gym/scripts/train.py --task=<target> --num_envs=8`
7. Verify logs and exported config:
   check generated `logs/.../config.yaml`.

## 11. Change-Safe Editing Playbook

Use this mapping to avoid scattering edits:

| Goal | Primary files |
|---|---|
| Add a new algorithm variant | `rsl_rl/rsl_rl/algorithms/`, `rsl_rl/rsl_rl/modules/`, `rsl_rl/rsl_rl/algorithms/__init__.py`, `rsl_rl/rsl_rl/modules/__init__.py`, `legged_gym/envs/base/legged_robot_config.py`, `legged_gym/envs/go2/go2_config.py`, `legged_gym/envs/__init__.py` |
| Change rewards/curriculum | `legged_gym/envs/go2/go2_config.py`, possibly reward functions in `legged_gym/envs/base/legged_robot.py` or `legged_gym/envs/go2/go2_env.py` |
| Change terrain difficulty/type mix | `legged_gym/envs/go2/go2_config.py`, `legged_gym/utils/terrain.py` |
| Change domain randomization | `legged_gym/envs/go2/go2_config.py`, domain rand handling in base env |
| Change export/deploy compatibility | `legged_gym/utils/exporter.py`, `deploy/deploy_mujoco/*.py`, `deploy/deploy_real/*.py`, deploy YAML files |
| Change resume/log behavior | `legged_gym/utils/helpers.py`, `legged_gym/utils/task_registry.py`, runner files |

Safety checks after any training-path edit:

1. Launch `--num_envs=8` debug train run.
2. Verify one checkpoint is saved.
3. Run `play.py` load path.
4. Export JIT and run Mujoco deploy script.

## 12. Living Update Protocol

Update this file whenever any of the following changes:

- New registered task name.
- New/removed algorithm or policy module.
- Observation/reward/curriculum semantic changes.
- Deployment I/O contract changes.
- Logging/checkpoint/export path conventions.

Update procedure (mandatory):

1. Add/refresh a short “Last verified” line below.
2. Update task matrix and edit-playbook tables first.
3. Update pitfalls if behavior or naming changed.
4. Keep command examples runnable against current CLI flags.
5. Keep file paths exact and workspace-relative.

Last verified against repository state: 2026-02-10.

## 13. Open Questions and Suggested Next Investigations

1. Unify the task naming/story across README and `UPDATE.md` to remove historical alias confusion.
2. Decide whether to expose currently unregistered config variants as official tasks.
3. Evaluate making terrain hardness configurable instead of hard-coded `IS_HARD`.
4. Consider adding a lightweight automated smoke test script for:
   train (8 env) -> save checkpoint -> play load -> export -> Mujoco inference.
5. Consider documenting expected RoboGauge server setup directly in repo docs to reduce integration friction.
