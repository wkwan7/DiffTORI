# bash scripts/metaworld/gen_demonstration_expert.sh reach 5
import argparse
import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
from utilis.config import ARGConfig
from utilis.default_config import default_config, dmc_config
from diffusion_policy_3d.env import MetaWorldEnv
from model.algorithm import BAC
from termcolor import cprint
import copy
import imageio
from metaworld.policies import *
import cv2
# import faulthandler
# faulthandler.enable()

seed = np.random.randint(0, 100)

def load_mw_policy(task_name):
	if task_name == 'peg-insert-side':
		agent = SawyerPegInsertionSideV2Policy()
	else:
		task_name = task_name.split('-')
		task_name = [s.capitalize() for s in task_name]
		task_name = "Sawyer" + "".join(task_name) + "V2Policy"
		agent = eval(task_name)()
	return agent

def main(args):
	env_name = args.env_name

	
	# save_dir = os.path.join(args.root_dir, 'metaworld_'+args.env_name+'_expert.zarr')
	save_dir = os.path.join(args.root_dir, 'temp_'+args.env_name)
	if os.path.exists(save_dir):
		cprint('Data already exists at {}'.format(save_dir), 'red')
		cprint("If you want to overwrite, delete the existing directory first.", "red")
		cprint("Do you want to overwrite? (y/n)", "red")
		user_input = 'y'
		if user_input == 'y':
			cprint('Overwriting {}'.format(save_dir), 'red')
			os.system('rm -rf {}'.format(save_dir))
		else:
			cprint('Exiting', 'red')
			return
	os.makedirs(save_dir, exist_ok=True)
	use_point_crop = args.use_point_crop
	num_points = args.num_points
	e = MetaWorldEnv(env_name, is_state_based=True, device="cuda:0", use_point_crop=use_point_crop, num_points=num_points)
	
 
	# create policy
	policy_path = f"expert_ckpt/{env_name}.torch"
	print("Policy path: ", policy_path)
	# config.update({"device": "3"})
	pi = BAC(e.observation_space.shape[0], e.action_space, config)
	pi.load_checkpoint(policy_path, evaluate=True)

	num_episodes = args.num_episodes
	cprint(f"Number of episodes : {num_episodes}", "yellow")
	

	total_count = 0
	img_arrays = []
	point_cloud_arrays = []
	depth_arrays = []
	state_arrays = []
	action_arrays = []
	episode_ends_arrays = []
    
	
	episode_idx = 0
	


	mw_policy = load_mw_policy(env_name)
	cprint(f"Using script policy: {args.use_script_policy}", "yellow")
	
	# loop over episodes
	while episode_idx < num_episodes:
		raw_state = e.reset()

		obs_dict = e.get_visual_obs()

		
		done = False
		
		ep_reward = 0.
		ep_success = False
		ep_success_times = 0
		

		img_arrays_sub = []
		point_cloud_arrays_sub = []
		depth_arrays_sub = []
		state_arrays_sub = []
		action_arrays_sub = []
		total_count_sub = 0
  
		while not done:

			total_count_sub += 1
			
			obs_img = obs_dict['image']
			obs_robot_state = obs_dict['agent_pos']
			obs_point_cloud = obs_dict['point_cloud']
			obs_depth = obs_dict['depth']
   

			img_arrays_sub.append(obs_img)
			point_cloud_arrays_sub.append(obs_point_cloud)
			depth_arrays_sub.append(obs_depth)
			state_arrays_sub.append(obs_robot_state)

			high_res_img = e.get_rgb_high_resolution()
			cv2.imwrite(os.path.join(save_dir, f"high_res_{total_count_sub}.png"), high_res_img[:,:,::-1])


			# get action
			mw_action = mw_policy.get_action(raw_state)
			action = pi.select_action(raw_state)
			if args.use_script_policy:
				action = mw_action

			action_arrays_sub.append(action)
			raw_state, reward, done, info = e.step(action)
			obs_dict = e.get_visual_obs()

			ep_reward += reward
   

			ep_success = ep_success or info['success']
			ep_success_times += info['success']
   
			if done:
				break
		if not ep_success or ep_success_times < 15:
			cprint(f'Episode: {episode_idx} failed with reward {ep_reward} and success times {ep_success_times}', 'red')
			continue
		else:
			total_count += total_count_sub
			episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode    
			img_arrays.extend(copy.deepcopy(img_arrays_sub))
			point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
			depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
			state_arrays.extend(copy.deepcopy(state_arrays_sub))
			action_arrays.extend(copy.deepcopy(action_arrays_sub))
			cprint('Episode: {}, Reward: {}, Success Times: {}'.format(episode_idx, ep_reward, ep_success_times), 'green')
			episode_idx += 1
			import pdb; pdb.set_trace()
	

	# save data
 	###############################
    # save data
    ###############################
    # create zarr file
	zarr_root = zarr.group(save_dir)
	zarr_data = zarr_root.create_group('data')
	zarr_meta = zarr_root.create_group('meta')
	# save img, state, action arrays into data, and episode ends arrays into meta
	img_arrays = np.stack(img_arrays, axis=0)
	if img_arrays.shape[1] == 3: # make channel last
		img_arrays = np.transpose(img_arrays, (0,2,3,1))
	state_arrays = np.stack(state_arrays, axis=0)
	point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
	depth_arrays = np.stack(depth_arrays, axis=0)
	action_arrays = np.stack(action_arrays, axis=0)
	episode_ends_arrays = np.array(episode_ends_arrays)

	compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
	img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
	state_chunk_size = (100, state_arrays.shape[1])
	point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
	depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
	action_chunk_size = (100, action_arrays.shape[1])
	zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)


	# print shape
	cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
	cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
	cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
	cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
	cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
	cprint(f'Saved zarr file to {save_dir}', 'green')

	# clean up
	del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
	del zarr_root, zarr_data, zarr_meta
	del e, pi


 
if __name__ == "__main__":
    
	arg = ARGConfig()
	arg.add_arg("env_name", "coffee-button", "Environment name")
	arg.add_arg("device", "0", "Computing device")
	arg.add_arg("policy", "Gaussian", "Policy Type: Gaussian | Deterministic (default: Gaussian)")
	arg.add_arg("tag", "default", "Experiment tag")
	arg.add_arg("algo", "BAC", "choose algorithm (BAC, SAC, TD3, TD3-BEE)")
	arg.add_arg("start_steps", 10000, "Number of start steps")
	arg.add_arg("automatic_entropy_tuning", True, "Automaically adjust α (default: True)")
	arg.add_arg("quantile", 0.7, "the quantile regression for value function (default: 0.9)")
	arg.add_arg("seed", 0, "experiment seed")
	arg.add_arg("lambda", "fixed_0.5", "method to calculated lambda, fixed_x, ada, min, max")
	arg.add_arg("des", "", "short description for the experiment")
	arg.add_arg("num_steps", 1000001, "total number of steps")
	arg.add_arg("save_checkpoint", False, "save checkpoint or not")
	arg.add_arg("replay_size", 1000000, "size of replay buffer")


	arg.add_arg("num_episodes", 10, "number of total collected episodes")
	arg.add_arg("root_dir", "data", "directory to save data")
 
	arg.add_arg('use_point_crop', True, "use point crop or not")
	arg.add_arg('num_points', 512, "number of points in point cloud")
	arg.add_arg('use_script_policy', False, "use script policy or not")

	arg.parser()

	config = dmc_config  
	config.update(arg)
	main(config)
