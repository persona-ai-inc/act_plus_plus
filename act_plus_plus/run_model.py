# This runs a model once for Aloha trained from imitate_episodes.py
#
# Supported models: ACT


import argparse
from copy import deepcopy
from itertools import repeat
import os
import pickle
import time

# from aloha.constants import FPS, FOLLOWER_GRIPPER_JOINT_OPEN, TASK_CONFIGS
FPS = 50
FOLLOWER_GRIPPER_JOINT_OPEN = 1.6214
from constants import TASK_CONFIGS
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import wandb

#from detr.models.latent_model import Latent_Model_Transformer
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from utils import (
    load_data,
    compute_dict_mean,
    set_seed,
)


def main(args):

    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_steps = args["num_steps"]
    validate_every = args["validate_every"]
    save_every = args["save_every"]
    resume_ckpt_path = args["resume_ckpt_path"]

    # get task parameters
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    # num_episodes = task_config['num_episodes']
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    stats_dir = task_config.get("stats_dir", None)
    sample_weights = task_config.get("sample_weights", None)
    train_ratio = task_config.get("train_ratio", 0.99)
    name_filter = task_config.get("name_filter", lambda n: True)

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "action_dim": 16,
            "backbone": backbone,
            "camera_names": camera_names,
            "dec_layers": dec_layers,
            "dim_feedforward": args["dim_feedforward"],
            "enc_layers": enc_layers,
            "hidden_dim": args["hidden_dim"],
            "kl_weight": args["kl_weight"],
            "lr_backbone": lr_backbone,
            "lr": args["lr"],
            "nheads": nheads,
            "no_encoder": args["no_encoder"],
            "num_queries": args["chunk_size"],
            "vq_class": args["vq_class"],
            "vq_dim": args["vq_dim"],
            "vq": args["use_vq"],
        }

        actuator_config = {
            "actuator_network_dir": args["actuator_network_dir"],
            "history_len": args["history_len"],
            "future_len": args["future_len"],
            "prediction_len": args["prediction_len"],
        }

    config = {
        "actuator_config": actuator_config,
        "camera_names": camera_names,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "lr": args["lr"],
        "num_steps": num_steps,
        "onscreen_render": onscreen_render,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "real_robot": True,
        "resume_ckpt_path": resume_ckpt_path,
        "save_every": save_every,
        "seed": args["seed"],
        "state_dim": state_dim,
        "task_name": task_name,
        "temporal_agg": args["temporal_agg"],
        "validate_every": validate_every,
    }

    if is_eval:
        ckpt_name = "policy_last.ckpt"
        results = []
        eval_bc(config, ckpt_name)
        print()
        exit()

    return


def eval_bc(config, ckpt_name):

    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    vq = config['policy_config']['vq']
    actuator_config = config['actuator_config']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    if vq:
        # vq_dim = config['policy_config']['vq_dim']
        # vq_class = config['policy_config']['vq_class']
        # latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        # latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
        # latent_model.deserialize(torch.load(latent_model_ckpt_path))
        # latent_model.eval()
        # latent_model.cuda()
        #print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
        print("WARNING: VQ flag set but latent model not instantiated.")
    else:
        print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']


    from aloha.real_env import make_real_env # requires aloha
    from aloha.robot_utils import move_grippers # requires aloha
    from interbotix_common_modules.common_robot.robot import (
        create_interbotix_global_node,
        get_interbotix_global_node,
        robot_startup,
    )
    from interbotix_common_modules.common_robot.exceptions import InterbotixException
    try:
        node = get_interbotix_global_node()
    except:
        node = create_interbotix_global_node('aloha')
    env = make_real_env(node=node, setup_robots=True, setup_base=True)
    try:
        robot_startup(node)
    except InterbotixException:
        pass
    env_max_reward = 0

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    if real_robot:
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks


    ts = env.reset()
    # evaluation loop
    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 16]).cuda()

    qpos_history_raw = np.zeros((max_timesteps, state_dim))
    image_list = [] # for visualization
    qpos_list = []
    target_qpos_list = []
    rewards = []
    with torch.inference_mode():
        time0 = time.time()
        DT = 1 / FPS
        culmulated_delay = 0
        for t in range(max_timesteps):
            time1 = time.time()
            ### process previous timestep to get qpos and image_list
            time2 = time.time()
            obs = ts.observation
            if 'images' in obs:
                image_list.append(obs['images'])
            else:
                image_list.append({'main': obs['image']})
            qpos_numpy = np.array(obs['qpos'])
            qpos_history_raw[t] = qpos_numpy
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            if t % query_frequency == 0:
                curr_image = get_image(
                    ts,
                    camera_names,
                    rand_crop_resize=(config['policy_class'] == 'Diffusion')
                )

            if t == 0:
                # warm up
                for _ in range(10):
                    policy(qpos, curr_image)
                print('network warm up done')
                time1 = time.time()

            ### query policy
            time3 = time.time()
            if config['policy_class'] == "ACT":
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                    if real_robot:
                        all_actions = torch.cat(
                            [
                                all_actions[:, :-BASE_DELAY, :-2],
                                all_actions[:, BASE_DELAY:, -2:]
                            ],
                            dim=2
                        )
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries-BASE_DELAY] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
            elif config['policy_class'] == "Diffusion":
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                    if real_robot:
                        all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                raw_action = all_actions[:, t % query_frequency]
            elif config['policy_class'] == "CNNMLP":
                raw_action = policy(qpos, curr_image)
                all_actions = raw_action.unsqueeze(0)
            else:
                raise NotImplementedError

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            target_qpos = action[:-2]

            base_action = action[-2:]

            # step the environment
            if real_robot:
                ts = env.step(target_qpos, base_action)
            else:
                ts = env.step(target_qpos)

            ### for visualization
            qpos_list.append(qpos_numpy)
            target_qpos_list.append(target_qpos)
            rewards.append(ts.reward)
            duration = time.time() - time1
            sleep_time = max(0, DT - duration)
            time.sleep(sleep_time)
            if duration >= DT:
                culmulated_delay += (duration - DT)
                print((
                    f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: '
                    f'{DT} s, culmulated delay: {culmulated_delay:.3f} s'
                ))

        print(f'Avg fps {max_timesteps / (time.time() - time0)}')
    if real_robot:
        move_grippers(
            [env.follower_bot_left, env.follower_bot_right],
            [FOLLOWER_GRIPPER_JOINT_OPEN] * 2,
            moving_time=0.5,
        )  # open
        # save qpos_history_raw

    return


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)

    return curr_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the selected model checkpoint",
    )
    parser.add_argument(
        "--onscreen_render",
        action="store_true",
        help="Render training onscreen",
    )
    parser.add_argument(
        "--ckpt_dir",
        action="store",
        type=str,
        help="Checkpoint directory",
        required=True,
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        default="ACT",
        help="The desired policy class",
        choices=["ACT", "Diffusion", "CNNMLP"],
    )
    parser.add_argument(
        "--task_name",
        action="store",
        type=str,
        help="Name of the task. Must be in task configurations",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        help="Training batch size",
        required=True,
    )
    parser.add_argument(
        "--seed", action="store", type=int, help="Training seed", required=True
    )
    parser.add_argument(
        "--num_steps",
        action="store",
        type=int,
        help="Number of training steps",
        required=True,
    )
    parser.add_argument(
        "--lr", action="store", type=float, help="Training learning rate", required=True
    )
    parser.add_argument(
        "--validate_every",
        action="store",
        type=int,
        default=500,
        help="Number of steps between validations during training",
        required=False,
    )
    parser.add_argument(
        "--save_every",
        action="store",
        type=int,
        default=500,
        help="Number of steps between checkpoints during training",
        required=False,
    )
    parser.add_argument(
        "--resume_ckpt_path",
        action="store",
        type=str,
        help="Path to checkpoint to resume training from",
        required=False,
    )
    parser.add_argument(
        "--skip_mirrored_data",
        action="store_true",
        help="Skip mirrored data during training",
        required=False,
    )
    parser.add_argument(
        "--actuator_network_dir",
        action="store",
        type=str,
        help="actuator_network_dir",
        required=False,
    )
    parser.add_argument(
        "--history_len",
        action="store",
        type=int,
    )
    parser.add_argument(
        "--future_len",
        action="store",
        type=int,
    )
    parser.add_argument(
        "--prediction_len",
        action="store",
        type=int,
    )

    # for ACT
    parser.add_argument(
        "--kl_weight",
        action="store",
        type=int,
        help="KL Weight",
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        help="chunk_size",
        required=False,
    )
    parser.add_argument(
        "--hidden_dim",
        action="store",
        type=int,
        help="hidden_dim",
        required=False,
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument(
        "--temporal_agg",
        action="store_true",
    )
    parser.add_argument(
        "--use_vq",
        action="store_true",
    )
    parser.add_argument(
        "--vq_class",
        action="store",
        type=int,
        help="vq_class",
    )
    parser.add_argument(
        "--vq_dim",
        action="store",
        type=int,
        help="vq_dim",
    )
    parser.add_argument(
        "--no_encoder",
        action="store_true",
    )

    main(vars(parser.parse_args()))
