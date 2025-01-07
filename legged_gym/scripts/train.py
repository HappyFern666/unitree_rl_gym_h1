import os
import numpy as np
from datetime import datetime
import sys
import statistics

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb

def train(args):
    wandb.init(
        project="unitree_H1_training",
        name=args.experiment_name if args.experiment_name else f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "task": args.task,
            "num_envs": args.num_envs,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
        }
    )
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.original_log = ppo_runner.log
    def new_log(self, locs, width=80, pad=35):
        # 调用原始的 log 方法
        self.original_log(locs, width, pad)
        
        # 额外记录到 wandb
        wandb_dict = {}
        
        # 记录损失值和其他指标
        wandb_dict.update({
            'Loss/value_function': locs['mean_value_loss'],
            'Loss/surrogate': locs['mean_surrogate_loss'],
            'Policy/mean_noise_std': ppo_runner.alg.actor_critic.std.mean().item(),
            'Perf/total_fps': int(ppo_runner.num_steps_per_env * ppo_runner.env.num_envs / 
                                (locs['collection_time'] + locs['learn_time']))
        })
        
        # 记录训练指标
        if len(locs['rewbuffer']) > 0:
            wandb_dict.update({
                'Train/mean_reward': statistics.mean(locs['rewbuffer']),
                'Train/mean_episode_length': statistics.mean(locs['lenbuffer'])
            })
        # 记录 episode 信息
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=ppo_runner.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(ppo_runner.device)))
                value = torch.mean(infotensor)
                wandb_dict[f'Episode/{key}'] = value.item()
        
        wandb.log(wandb_dict, step=locs['it'])
    
    # 替换 log 方法
    ppo_runner.log = new_log.__get__(ppo_runner)

    # 开始训练
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

    # 结束 wandb
    wandb.finish()

if __name__ == '__main__':
    args = get_args()
    train(args)