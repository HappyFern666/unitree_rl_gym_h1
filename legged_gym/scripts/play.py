import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch
import time
import cv2
from datetime import datetime

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 创建相机属性
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1920
    camera_props.height = 1080
    camera_props.enable_tensors = True
    
    # 创建相机传感器
    camera = env.gym.create_camera_sensor(env.envs[0], camera_props)
    
    # 设置相机位置（调整这些值以获得更好的视角）
    camera_position = gymapi.Vec3(2.0, 2.0, 1.5)  # 调整相机位置
    camera_target = gymapi.Vec3(0.0, 0.0, 0.5)    # 调整观察目标
    env.gym.set_camera_location(camera, env.envs[0], camera_position, camera_target)
    
    # 设置视频保存
    video_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', args.task + '_' + args.load_run.split('/')[-2] + '_' + "checkpoint" + str(args.checkpoint) + '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (camera_props.width, camera_props.height))

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # 获取初始观察
    obs = env.get_observations()

    try:
        for i in range(1000):  # 设置适当的步数
            # 执行策略
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            
            # 更新图形和渲染
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            
            # 获取相机图像数据
            frame = env.gym.get_camera_image(env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR)
            frame = frame.reshape(camera_props.height, camera_props.width, 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            out.write(frame)
            
            if env.gym.query_viewer_has_closed(env.viewer):
                break
                
    finally:
        out.release()
        print(f"Video saved to: {video_path}")

if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    play(args)