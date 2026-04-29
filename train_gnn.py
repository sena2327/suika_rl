"""
PPO training with GNN input from graph observations.

Flow:
  fruit objects
    -> node encoder
    -> GNN layers (3)
    -> global pooling
    -> board_feature
  current_type, next_type
    -> one-hot
    -> hand_feature
  concat(board_feature, hand_feature)
    -> policy / value
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

import suika_env  # noqa: F401
import suika_env_node  # noqa: F401
from policy_gif_callback import PolicyGifCallback
from train import (
    ActionStatsLoggingCallback,
    BrowserRestartCallback,
    EpisodeLengthMaxLoggingCallback,
    FinalScoreLoggingCallback,
    PolicyStdLoggingCallback,
    resolve_device,
    restore_terminal_cursor,
)


class SuikaGnnObsWrapper(gym.ObservationWrapper):
    """Keep graph obs + current/next hand type for GNN policy."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "node": spaces.Box(low=-1.0, high=1.0, shape=(50, 14), dtype=np.float32),
                "node_mask": spaces.Box(low=0.0, high=1.0, shape=(50,), dtype=np.float32),
                "edge": spaces.Box(low=-2.0, high=2.0, shape=(1225, 6), dtype=np.float32),
                "edge_index": spaces.Box(low=0, high=49, shape=(1225, 2), dtype=np.int32),
                "edge_mask": spaces.Box(low=0.0, high=1.0, shape=(1225,), dtype=np.float32),
                "global_feature": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                "largest_type_onehot": spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32),
                "current_fruit_type": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                "next_fruit_type": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            }
        )

    def observation(self, observation):
        return {
            "node": observation["node"].astype(np.float32, copy=False),
            "node_mask": observation["node_mask"].astype(np.float32, copy=False),
            "edge": observation["edge"].astype(np.float32, copy=False),
            "edge_index": observation["edge_index"].astype(np.int32, copy=False),
            "edge_mask": observation["edge_mask"].astype(np.float32, copy=False),
            "global_feature": observation["global_feature"].astype(np.float32, copy=False),
            "largest_type_onehot": observation["largest_type_onehot"].astype(np.float32, copy=False),
            "current_fruit_type": observation["current_fruit_type"].astype(np.float32, copy=False),
            "next_fruit_type": observation["next_fruit_type"].astype(np.float32, copy=False),
        }


class SuikaGnnExtractor(BaseFeaturesExtractor):
    """3-layer residual message-passing GNN + hand feature concat."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)
        self.n_hand = 11
        self.hidden_dim = 64
        self.node_encoder = nn.Sequential(
            nn.Linear(14, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear((self.hidden_dim * 2) + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.gnn_layers = nn.ModuleList()
        for _ in range(3):
            self.gnn_layers.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
            )
        self.gnn_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(3)])
        self.hand_head = nn.Sequential(nn.Linear(self.n_hand * 2, 32), nn.ReLU())
        self.global_head = nn.Sequential(nn.Linear(4 + 11, 32), nn.ReLU())
        self.out = nn.Sequential(nn.Linear((self.hidden_dim * 2) + 32 + 32, 256), nn.ReLU())
        self.edge_chunk_size = 128
        self._features_dim = 256

    def _message_pass(self, h, edge_index, edge_feat, edge_mask):
        bsz, n_nodes, dim = h.shape
        e = edge_index.shape[1]
        src = edge_index[:, :, 0].long()
        dst = edge_index[:, :, 1].long()
        base = (th.arange(bsz, device=h.device) * n_nodes).unsqueeze(1)
        src_flat = (src + base).reshape(-1)
        dst_flat = (dst + base).reshape(-1)

        h_flat = h.reshape(bsz * n_nodes, dim)
        agg_flat = th.zeros_like(h_flat)
        # Chunked edge processing to avoid giant (B,E,*) tensors on MPS.
        for s in range(0, e, self.edge_chunk_size):
            t = min(e, s + self.edge_chunk_size)
            src_c = src[:, s:t]
            dst_c = dst[:, s:t]
            src_flat_c = (src_c + base).reshape(-1)
            dst_flat_c = (dst_c + base).reshape(-1)

            h_src_c = h_flat[src_flat_c].reshape(bsz, t - s, dim)
            h_dst_c = h_flat[dst_flat_c].reshape(bsz, t - s, dim)
            edge_c = edge_feat[:, s:t, :]
            edge_mask_c = edge_mask[:, s:t].unsqueeze(-1)

            gate_in = th.cat([h_src_c, h_dst_c, edge_c], dim=-1)
            gate_in = th.nan_to_num(gate_in, nan=0.0, posinf=1e3, neginf=-1e3)
            gate = self.edge_gate(gate_in) * edge_mask_c
            gate = th.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0)

            msg_src_to_dst = h_src_c * gate
            msg_dst_to_src = h_dst_c * gate

            agg_flat.index_add_(0, dst_flat_c, msg_src_to_dst.reshape(-1, dim))
            agg_flat.index_add_(0, src_flat_c, msg_dst_to_src.reshape(-1, dim))
        out = agg_flat.reshape(bsz, n_nodes, dim)
        return th.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def forward(self, observations):
        node = th.nan_to_num(observations["node"].float(), nan=0.0, posinf=1.0, neginf=-1.0)
        node_mask = th.nan_to_num(observations["node_mask"].float(), nan=0.0, posinf=1.0, neginf=0.0)
        edge = th.nan_to_num(observations["edge"].float(), nan=0.0, posinf=2.0, neginf=-2.0)
        edge_index = observations["edge_index"].long()
        edge_mask = th.nan_to_num(observations["edge_mask"].float(), nan=0.0, posinf=1.0, neginf=0.0)

        h = self.node_encoder(node)
        h = th.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        h = h * node_mask.unsqueeze(-1)
        for layer, norm in zip(self.gnn_layers, self.gnn_norms):
            agg = self._message_pass(h, edge_index, edge, edge_mask)
            h_res = layer(th.cat([h, agg], dim=-1))
            h_res = th.nan_to_num(h_res, nan=0.0, posinf=1e6, neginf=-1e6)
            h = F.relu(norm(h + h_res))
            h = th.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
            h = h * node_mask.unsqueeze(-1)

        denom = th.clamp(node_mask.sum(dim=1, keepdim=True), min=1.0)
        board_mean = (h * node_mask.unsqueeze(-1)).sum(dim=1) / denom
        very_neg = th.full_like(h, -1e4)
        h_masked_for_max = th.where(node_mask.unsqueeze(-1) > 0.5, h, very_neg)
        board_max = h_masked_for_max.max(dim=1).values
        has_node = (node_mask.sum(dim=1, keepdim=True) > 0.5)
        board_max = th.where(has_node, board_max, th.zeros_like(board_max))
        board_max = th.where(th.isfinite(board_max), board_max, th.zeros_like(board_max))
        board_feature = th.cat([board_mean, board_max], dim=1)

        cur_idx = observations["current_fruit_type"].long().squeeze(1).clamp(0, self.n_hand - 1)
        nxt_idx = observations["next_fruit_type"].long().squeeze(1).clamp(0, self.n_hand - 1)
        cur_onehot = F.one_hot(cur_idx, num_classes=self.n_hand).float()
        nxt_onehot = F.one_hot(nxt_idx, num_classes=self.n_hand).float()
        hand_feature = self.hand_head(th.cat([cur_onehot, nxt_onehot], dim=1))
        global_in = th.cat(
            [
                th.nan_to_num(observations["global_feature"].float(), nan=0.0, posinf=1.0, neginf=0.0),
                th.nan_to_num(observations["largest_type_onehot"].float(), nan=0.0, posinf=1.0, neginf=0.0),
            ],
            dim=1,
        )
        global_feature = self.global_head(global_in)
        out = self.out(th.cat([board_feature, hand_feature, global_feature], dim=1))
        return th.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


def make_env_gnn(
    rank: int,
    seed: int,
    headless: bool,
    port_base: int,
    env_id: str,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env_kwargs = dict(
            headless=headless,
            delay_before_img_capture=0.0,
            mute_sound=True,
            wait_for_ready_on_step=True,
            ready_poll_interval=0.02,
            ready_timeout=2.0,
            enable_image_observation=False,
            bitmap_size=128,
            img_width=128,
            img_height=128,
        )
        if env_id == "SuikaEnv-v0":
            env_kwargs["port"] = port_base + rank
        env = gym.make(env_id, **env_kwargs)
        env = SuikaGnnObsWrapper(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--rollout-steps-total", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--env-id",
        type=str,
        default="SuikaEnvNode-v0",
        choices=["SuikaEnv-v0", "SuikaEnvNode-v0"],
    )
    p.add_argument("--port-base", type=int, default=8923)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--save-path", type=Path, default=Path("models/gnn/ppo_suika_gnn"))
    p.add_argument("--wandb-project", type=str, default="suika-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--save-every-steps", type=int, default=20_000)
    p.add_argument("--restart-browser-every-steps", type=int, default=3_000)
    p.add_argument("--gif-eval-every-steps", type=int, default=10_000)
    p.add_argument("--gif-eval-steps", type=int, default=10000)
    p.add_argument("--gif-fps", type=int, default=20)
    p.add_argument("--gif-dir", type=Path, default=Path("gifs/gnn"))
    p.add_argument("--device", type=str, default="cuda", help="auto|cpu|cuda|mps")
    p.add_argument("--gpu-id", type=int, default=None)
    p.add_argument(
        "--resume-path",
        type=Path,
        default=None,
        help="Path to existing PPO .zip model for continued training.",
    )
    p.add_argument(
        "--reset-num-timesteps",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="If True, reset timestep counter on learn() when resuming.",
    )
    p.add_argument("--check", type=lambda x: str(x).lower() == "true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    actual_device = resolve_device(args.device)
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    effective_n_steps = args.n_steps
    if args.rollout_steps_total > 0:
        effective_n_steps = max(1, args.rollout_steps_total // args.n_envs)
    effective_rollout_total = effective_n_steps * args.n_envs
    print(f"[train_gnn] n_envs={args.n_envs}, n_steps={effective_n_steps}, rollout_total={effective_rollout_total}")

    wandb_enabled = bool(args.wandb_run_name)
    run_name = args.wandb_run_name or f"ppo-suika-gnn-seed{args.seed}"
    tb_dir = Path("runs/tb") / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env_gnn(
            i,
            args.seed,
            args.headless,
            args.port_base,
            args.env_id,
        )
        for i in range(args.n_envs)
    ]
    if args.check:
        env = env_fns[0]()
        try:
            obs, _ = env.reset(seed=args.seed)
            print("[check] model input preview (train_gnn.py)")
            for k, v in obs.items():
                arr = np.asarray(v)
                print(
                    f"- {k}: shape={arr.shape}, dtype={arr.dtype}, "
                    f"min={float(np.min(arr)):.6f}, max={float(np.max(arr)):.6f}"
                )
        finally:
            env.close()
        return
    vec_env = DummyVecEnv(env_fns) if args.n_envs == 1 else SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    run = None
    if wandb_enabled:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "algo": "PPO",
                "extractor": "GNN",
                "obs": "graph(node/edge) + hand_onehot",
                "env_id": args.env_id,
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "seed": args.seed,
                "learning_rate": 1e-4,
                "n_steps": effective_n_steps,
                "rollout_steps_total": effective_rollout_total,
                "batch_size": 256,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "device": actual_device,
                "gpu_id": args.gpu_id,
                "save_every_steps": args.save_every_steps,
                "restart_browser_every_steps": args.restart_browser_every_steps,
                "gif_eval_every_steps": args.gif_eval_every_steps,
                "gif_eval_steps": args.gif_eval_steps,
                "gif_fps": args.gif_fps,
                "resume_path": str(args.resume_path) if args.resume_path is not None else None,
                "reset_num_timesteps": bool(args.reset_num_timesteps),
            },
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
        )

    interrupted = False
    try:
        if args.resume_path is not None:
            print(f"[train_gnn] Resuming from: {args.resume_path}")
            model = PPO.load(
                str(args.resume_path),
                env=vec_env,
                device=actual_device,
            )
        else:
            model = PPO(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs=dict(
                    features_extractor_class=SuikaGnnExtractor,
                    share_features_extractor=False,
                ),
                learning_rate=1e-4,
                n_steps=effective_n_steps,
                batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log=str(tb_dir),
                verbose=1,
                seed=args.seed,
                device=actual_device,
            )

        callbacks = []
        if wandb_enabled:
            save_freq = args.save_every_steps if args.save_every_steps > 0 else 0
            callbacks.append(
                WandbCallback(
                    gradient_save_freq=0,
                    model_save_path=str(args.save_path.parent / "wandb_checkpoints"),
                    model_save_freq=save_freq,
                    verbose=2,
                )
            )
        callbacks.append(FinalScoreLoggingCallback(verbose=0))
        callbacks.append(EpisodeLengthMaxLoggingCallback(verbose=0))
        callbacks.append(ActionStatsLoggingCallback(verbose=0))
        callbacks.append(PolicyStdLoggingCallback(verbose=0))
        if args.gif_eval_every_steps > 0:
            callbacks.append(
                PolicyGifCallback(
                    every_steps=args.gif_eval_every_steps,
                    max_steps_per_episode=args.gif_eval_steps,
                    fps=args.gif_fps,
                    out_dir=args.gif_dir,
                    seed=args.seed,
                    headless=args.headless,
                    port_base=args.port_base,
                    env_id=args.env_id,
                    total_timesteps=args.total_timesteps,
                    verbose=1,
                )
            )
        if args.env_id == "SuikaEnv-v0" and args.restart_browser_every_steps > 0:
            callbacks.append(
                BrowserRestartCallback(
                    every_steps=args.restart_browser_every_steps,
                    verbose=1,
                )
            )

        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=True,
            log_interval=1,
            callback=CallbackList(callbacks),
            reset_num_timesteps=bool(args.reset_num_timesteps),
        )
        model.save(str(args.save_path))
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received. Finishing cleanup...")
        if "model" in locals():
            interrupted_path = args.save_path.parent / f"{args.save_path.name}_interrupted"
            model.save(str(interrupted_path))
            print(f"Saved interrupted model to: {interrupted_path}.zip")
        restore_terminal_cursor()
    finally:
        vec_env.close()
        if run is not None:
            run.finish()
        restore_terminal_cursor()

    if not interrupted:
        print(f"Saved model to: {args.save_path}.zip")


if __name__ == "__main__":
    main()
