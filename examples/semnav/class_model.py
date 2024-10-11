import sys

pirlnav_directory = '/home/gram/repositorios/ros4vsn/examples/pirlnav/data'
sys.path.append(pirlnav_directory)

import random
import numpy as np
import torch
import os

from habitat_baselines.common.baseline_registry import baseline_registry


#Packages in order to prepare the action space and the observation spacee
from gym import spaces
from habitat.core.spaces import EmptySpace, ActionSpace
from collections import OrderedDict
##############

from habitat.utils.env_utils import construct_envs
from habitat.core.environments import get_env_class
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)

from pirlnav.algos.agent import DDPILAgent
from pirlnav.algos.agent import Semantic_DDPILAgent

from habitat_baselines.utils.common import (
    action_array_to_dict,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from pirlnav.config import get_config

from pirlnav.algos.ppo import DDPPO, PPO
class Pirlnav(PPOTrainer):
    def __init__(self):

        self.constant = 414534
        self.seed = 17
        self.run_type = "eval"
        archive_path = os.path.abspath(__file__)
        relative_path = os.path.dirname(archive_path)
        self.path = relative_path + "/configs/experiments/il_objectnav.yaml"
        print(self.path)
        self.opt = ['TENSORBOARD_DIR', 'tb/objectnav_il_rl_ft/ovrl_resnet50/seed_1/','EVAL_CKPT_PATH_DIR','/home/gram/rafarepos/ros4vsn/examples/semnav/model/ckpt.13.pth', 'NUM_UPDATES', '20000', 'NUM_ENVIRONMENTS', '1', 'RL.DDPPO.pretrained', 'False', 'EVAL.USE_CKPT_CONFIG', False]


        # Load configuration

        self.config = get_config(self.path, self.opt)
        self.config.defrost()
        self.config.RUN_TYPE = self.run_type
        self.config.TASK_CONFIG.SEED = self.seed
        self.config.freeze()
        super().__init__(self.config)
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        if self.config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
            torch.set_num_threads(1)

        self.device = torch.device("cpu")
        #torch.cuda.set_device(self.device)
        #self._init_envs()
        action_spaces = OrderedDict([
            ('LOOK_DOWN', EmptySpace()),
            ('LOOK_UP', EmptySpace()),
            ('MOVE_FORWARD', EmptySpace()),
            ('STOP', EmptySpace()),
            ('TURN_LEFT', EmptySpace()),
            ('TURN_RIGHT', EmptySpace()),
        ])
        self.action_space = ActionSpace(action_spaces)
        self.policy_action_space = self.action_space
        self.observation_space = spaces.Dict({
            "compass": spaces.Box(low=-3.1415927, high=3.1415927, shape=(1,), dtype='float32'),
            "gps": spaces.Box(low=-3.4028235e+38, high=3.4028235e+38, shape=(2,), dtype='float32'),
            "objectgoal": spaces.Box(low=0, high=5, shape=(1,), dtype='int64'),
            "rgb": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype='uint8'),
            "semantic_rgb": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype='uint8'),
            "semantic": spaces.Box(low=0, high=-1, shape=(480, 640, 1), dtype='int32')
        })
        self._init_checkpoint()
        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.TYPE = "ObjectNav-v1"
        self.config.freeze()


    def _init_checkpoint(self):
        self.il_cfg = self.config.IL.BehaviorCloning
        self.policy_cfg = self.config.POLICY
        self.obs_transforms = get_active_obs_transforms(self.config)
        self.observation_space = apply_obs_transforms_obs_space(self.observation_space, self.obs_transforms)
        self.obs_space = self.observation_space

        #### SET ACTOR CRITIC
        self.policy = baseline_registry.get_policy(self.config.IL.POLICY.name)
        self.actor_critic = self.policy.from_config(
            self.config, self.observation_space, self.policy_action_space
        )
        self.actor_critic.to(self.device)
        if 'semantic' in self.observation_space.spaces:
            self.agent = Semantic_DDPILAgent(
                actor_critic=self.actor_critic,
                num_envs=1,
                num_mini_batch=self.il_cfg.num_mini_batch,
                lr=self.il_cfg.lr,
                encoder_lr=self.il_cfg.encoder_lr,
                eps=self.il_cfg.eps,
                max_grad_norm=self.il_cfg.max_grad_norm,
                wd=self.il_cfg.wd,
                entropy_coef=self.il_cfg.entropy_coef,
            )
        else:
            self.agent = DDPILAgent(
                actor_critic=self.actor_critic,
                num_envs=1,
                num_mini_batch=self.il_cfg.num_mini_batch,
                lr=self.il_cfg.lr,
                encoder_lr=self.il_cfg.encoder_lr,
                eps=self.il_cfg.eps,
                max_grad_norm=self.il_cfg.max_grad_norm,
                wd=self.il_cfg.wd,
                entropy_coef=self.il_cfg.entropy_coef,
            )
        ##########
        self.actor_critic = self.agent.actor_critic.to(self.device)

        self.ckpt_dict = self.load_checkpoint(self.opt[3], map_location="cpu")
        self.config = self._setup_eval_config(self.ckpt_dict["config"])






        #Cargar entornos para conseguir el action space

        # self.envs = construct_envs(
        #     self.config,
        #     get_env_class(self.config.ENV_NAME),
        #     workers_ignore_signals=is_slurm_batch_job(),
        # )
        self.policy_action_space = self.action_space



        self.observation_space = self.observation_space





        self.agent.load_state_dict(self.ckpt_dict["state_dict"])
        print("Device")
        print(self.device)
        self.test_recurrent_hidden_states = torch.zeros(self.config.NUM_ENVIRONMENTS,
                                                        self.actor_critic.num_recurrent_layers,
                                                        self.policy_cfg.STATE_ENCODER.hidden_size, device=self.device, )
        action_shape = (1,)
        self.prev_actions = torch.zeros(self.config.NUM_ENVIRONMENTS, *action_shape, device=self.device,
                                        dtype=torch.long)

        self.not_done_masks = torch.zeros(self.config.NUM_ENVIRONMENTS, 1, device=self.device, dtype=torch.bool, )
        self.aux = False
    def evaluation(self, observation):

        batch = batch_obs(observation, device=self.device, cache=self._obs_batching_cache)


        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        self.actor_critic.eval()
        (actions, self.test_recurrent_hidden_states,) = self.actor_critic.act(batch,
                                                                                    self.test_recurrent_hidden_states,
                                                                                    self.prev_actions,
                                                                                    self.not_done_masks,
                                                                                    deterministic=True,
                                                                                    )

        self.prev_actions.copy_(actions)

        self.not_done_masks = torch.ones(self.config.NUM_ENVIRONMENTS,
                                         1,
                                         device=self.device,
                                         dtype=torch.bool, )

        self.not_done_masks = self.not_done_masks.to(device=self.device)
        print(actions.item())
        return actions.item()

