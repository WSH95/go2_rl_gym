import os
import re
from pathlib import Path

from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import Logger, get_args, get_load_path, task_registry
from legged_gym.utils.exporter import export_policy_as_jit, export_policy_as_onnx, export_policy_as_pkl

import numpy as np
import torch


def _jit_iteration_key(path):
    match = re.search(r"policy_jit_(\d+)\.pt$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _resolve_jit_path(args, train_cfg):
    attempted_paths = []

    if args.jit_path:
        jit_path = Path(args.jit_path).expanduser()
        if jit_path.is_file():
            return str(jit_path)
        attempted_paths.append(jit_path)
        raise FileNotFoundError(
            "JIT policy path does not exist: {}\nTried:\n  - {}".format(
                args.jit_path, "\n  - ".join(str(p) for p in attempted_paths)
            )
        )

    log_root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
    exported_path = log_root / "exported" / "policies" / "policy.pt"
    attempted_paths.append(exported_path)
    if exported_path.is_file():
        return str(exported_path)

    load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run
    checkpoint = args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint

    run_dir = None
    try:
        checkpoint_path = Path(get_load_path(str(log_root), load_run=load_run, checkpoint=checkpoint))
        run_dir = checkpoint_path.parent
    except Exception:
        pass

    if run_dir is not None:
        run_exported_path = run_dir / "exported" / "policies" / "policy.pt"
        attempted_paths.append(run_exported_path)
        if run_exported_path.is_file():
            return str(run_exported_path)

        jit_dir = run_dir / "jit_models"

        if checkpoint is not None and checkpoint != -1:
            checkpoint_jit_path = jit_dir / f"policy_jit_{checkpoint}.pt"
            attempted_paths.append(checkpoint_jit_path)
            if checkpoint_jit_path.is_file():
                return str(checkpoint_jit_path)

        jit_candidates = sorted(jit_dir.glob("policy_jit_*.pt"), key=_jit_iteration_key)
        if len(jit_candidates) > 0:
            return str(jit_candidates[-1])
        attempted_paths.append(jit_dir / "policy_jit_*.pt")

    raise FileNotFoundError(
        "Could not resolve JIT policy automatically. Provide --jit_path or export policy first.\nTried:\n  - "
        + "\n  - ".join(str(p) for p in attempted_paths)
    )


def _extract_jit_actions(policy_output):
    if isinstance(policy_output, torch.Tensor):
        return policy_output
    if isinstance(policy_output, (tuple, list)) and len(policy_output) > 0 and isinstance(policy_output[0], torch.Tensor):
        return policy_output[0]
    raise TypeError(
        "Unsupported JIT policy output type: {}. Expected Tensor or tuple/list with Tensor as first item.".format(
            type(policy_output)
        )
    )


def _build_jit_policy(jit_policy):
    def _policy(obs):
        return _extract_jit_actions(jit_policy(obs))

    return _policy


def play(args):
    if args.use_jit and args.use_teacher:
        raise ValueError("--use_jit does not support --use_teacher. Please disable one of them.")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 7
    env_cfg.terrain.num_cols = 7
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_zero_offset = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    privileged_obs = env.get_privileged_observations()

    if args.use_jit:
        jit_path = _resolve_jit_path(args, train_cfg)
        print(f"Loading JIT policy from: {jit_path}")
        jit_policy = torch.jit.load(jit_path, map_location=env.device)
        jit_policy.eval()
        policy = _build_jit_policy(jit_policy)
    else:
        # load policy from checkpoint runner
        train_cfg.runner.resume = True
        runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = runner.get_inference_policy(device=env.device, use_teacher=args.use_teacher)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY and not args.use_jit:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        if hasattr(runner.alg, 'actor_critic'):
            model = runner.alg.actor_critic
        else:
            model = runner.alg.model
        export_policy_as_jit(model, path)
        export_policy_as_onnx(model, path)
        export_policy_as_pkl(model, path)
        print('Exported policy as jit script / onnx to: ', path)

    for i in range(10*int(env.max_episode_length)):
        if args.use_teacher:
            if privileged_obs is None:
                raise RuntimeError("Teacher inference requires privileged observations, but got None.")
            actions = policy(obs.detach(), privileged_obs.detach())
        else:
            actions = policy(obs.detach())

        if FIX_COMMAND:
            env.commands[:, 0] = 1.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0

        obs, privileged_obs, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIX_COMMAND = True
    args = get_args()
    play(args)
