import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import metaworld
import random
import time

from natsort import natsorted
from termcolor import cprint
from gym import spaces
from diffusion_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling

TASK_BOUDNS = {
    # x_min, y_min, z_min, x_max, y_max, z_max
    'default': [-0.5, -1.65, -0.70, 1, 100, 100],
}

class MetaWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, is_state_based=False, device="cuda:0", 
                 use_point_crop=True,
                 num_points=512,
                 ):
        super(MetaWorldEnv, self).__init__()

        if '-v2' not in task_name:
            task_name = task_name + '-v2-goal-observable'

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        self.env._freeze_rand_vec = False # this is important for domain randomization

        # adjust camera as Modem: https://arxiv.org/abs/2212.05698
        self.env.sim.model.cam_pos[2] = [0.75, 0.075, 0.7]
        
        # !!! this is very important, otherwise the point cloud will be wrong.
        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5
        
        self.device_id = int(device.split(":")[-1])
        
        # set the device of mujoco simulation

        
        self.image_size = 84
        
        self.pc_generator = PointCloudGenerator(sim=self.env.sim, cam_names=['corner2'])
        self.use_point_crop = use_point_crop
        cprint("[MetaWorldEnv] use_point_crop: {}".format(self.use_point_crop), "cyan")
        self.num_points = num_points # 512
        
        # angle = 74.5 # for corner
        x_angle = 61.4 # for corner2
        y_angle = -7
        self.pc_transform = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(x_angle)), np.sin(np.deg2rad(x_angle))],
            [0, -np.sin(np.deg2rad(x_angle)), np.cos(np.deg2rad(x_angle))]
        ]) @ np.array([
            [np.cos(np.deg2rad(y_angle)), 0, np.sin(np.deg2rad(y_angle))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(y_angle)), 0, np.cos(np.deg2rad(y_angle))]
        ])
        
        self.pc_scale = np.array([1, 1, 1])
        self.pc_offset = np.array([0, 0, 0])
        # self.min_bound = [-100, -0.5, -0.05]
        if task_name in TASK_BOUDNS:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[task_name]
        else:
            # cprint("Task {} not found in TASK_BOUNDS, using default bounds".format(task_name), "red")
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]
        
        
        # self.episode_length = self.env.max_path_length # this is 500
        self.episode_length = self._max_episode_steps = 200
        
        self.is_state_based = is_state_based
        
        self.action_space = self.env.action_space
        
        # cprint("[MetaWorldEnv] action_space: {}".format(self.env.action_space.shape), "yellow")

        self.obs_sensor_dim = self.get_robot_state().shape[0]

        
        if self.is_state_based:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self.image_size, self.image_size),
                    dtype=np.float32
                ),
                'depth': spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.image_size, self.image_size),
                    dtype=np.float32
                ),
                'agent_pos': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.obs_sensor_dim,),
                    dtype=np.float32
                ),
                'point_cloud': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_points, 3),
                    dtype=np.float32
                ),
            })

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        # import pdb; pdb.set_trace()
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
        # img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name="corner2",
        #                         #   device_id=self.device_id, 
        #                           mode='offscreen')
        img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name="corner2", device_id=self.device_id)
        return img
    
    def get_rgb_high_resolution(self):
        img = self.env.sim.render(width=1024, height=1024, camera_name="corner2", device_id=self.device_id)
        return img

    def render_high_res(self, resolution=1024):
        img = self.env.sim.render(width=resolution, height=resolution, camera_name="corner2", device_id=self.device_id)
        return img
    

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(device_id=self.device_id) # raw point cloud, Nx3
        
        
        if not use_rgb:
            point_cloud = point_cloud[..., :3]

        # do transform, scale, offset, and crop
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset
        
        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]
        

        point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps')
        
        depth = depth[::-1] # flip vertically
        
        return point_cloud, depth
        

    def get_visual_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
        }
        return obs_dict
            
            
    def step(self, action: np.array):

        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1
        
        if not self.is_state_based:
            obs_pixels = self.get_rgb()
            robot_state = self.get_robot_state()
            point_cloud, depth = self.get_point_cloud()
            
            if obs_pixels.shape[0] != 3:  # make channel first
                obs_pixels = obs_pixels.transpose(2, 0, 1)

            obs_dict = {
                'image': obs_pixels,
                'depth': depth,
                'agent_pos': robot_state,
                'point_cloud': point_cloud,
            }

        else:
            obs_dict = raw_state

        done = done or self.cur_step >= self.episode_length
        
        return obs_dict, reward, done, env_info

    def reset(self):
        self.env.reset()
        self.env.reset_model() # this is important for domain randomization
        raw_obs = self.env.reset()
        self.cur_step = 0

        if not self.is_state_based:
            
            obs_pixels = self.get_rgb()
            robot_state = self.get_robot_state()
            point_cloud, depth = self.get_point_cloud()
            
            if obs_pixels.shape[0] != 3:  # make channel first
                obs_pixels = obs_pixels.transpose(2, 0, 1)

            obs_dict = {
                'image': obs_pixels,
                'depth': depth,
                'agent_pos': robot_state,
                'point_cloud': point_cloud,
            }
        else:
            obs_dict = raw_obs

        return obs_dict

    def seed(self, seed=None):
        # self.env.seed(seed)
        pass

    def set_seed(self, seed=None):
        # self.env.seed(seed)
        pass

    def render(self, mode='rgb_array'):
        img = self.get_rgb()
        return img

    def close(self):
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='assembly')
    args = parser.parse_args()
    env_name = args.env + '-v2'
    
    env = MetaWorldEnv(env_name, is_state_based=True)
    from metaworld.policies import SawyerAssemblyV2Policy
    policy = SawyerAssemblyV2Policy()


    obs = env.reset()
    for i in range(100):
        # action = env.action_space.sample()
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)
        env.get_point_cloud()
        rgb = env.get_rgb()
        plt.imsave("debug.png", rgb)
        time.sleep(0.5)
        print(i, reward, done)
        if done:
            break
    env.close()
    cprint("MetaWorld env successfully closed", "green")
