#!/usr/bin/env python3
import time
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import os
import torch
from habitat import logger
from habitat.utils import profiling_wrapper
from torch import Tensor
from torch import nn as nn
from torch import optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
class ILAgent(nn.Module):
    def __init__(
        self,
        actor_critic: nn.Module,
        num_envs: int,
        num_mini_batch: int,
        lr: Optional[float] = None,
        encoder_lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        wd: Optional[float] = None,
        optimizer: Optional[str] = "AdamW",
        entropy_coef: Optional[float] = 0.0,
    ) -> None:

        super().__init__()

        self.contador = 0

        self.actor_critic = actor_critic

        self.num_mini_batch = num_mini_batch

        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs
        self.entropy_coef = entropy_coef

        # use different lr for visual encoder and other networks
        visual_encoder_params, other_params = [], []
        for name, param in actor_critic.named_parameters():
            if param.requires_grad:
                if "net.visual_encoder.backbone" in name:
                    visual_encoder_params.append(param)
                else:
                    other_params.append(param)
        logger.info(
            "Visual Encoder params: {}".format(len(visual_encoder_params))
        )
        logger.info("Other params: {}".format(len(other_params)))

        if optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                [
                    {"params": visual_encoder_params, "lr": encoder_lr},
                    {"params": other_params, "lr": lr},
                ],
                lr=lr,
                eps=eps,
                weight_decay=wd,
            )
        else:
            self.optimizer = optim.Adam(
                list(
                    filter(
                        lambda p: p.requires_grad, actor_critic.parameters()
                    )
                ),
                lr=lr,
                eps=eps,
            )
        self.device = torch.device("cpu")

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts) -> Tuple[float, float, float]:
        total_loss_epoch = 0.0
        total_entropy = 0.0
        total_action_loss = 0.0

        profiling_wrapper.range_push("BC.update epoch")
        data_generator = rollouts.recurrent_generator(self.num_mini_batch)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        hidden_states = []


        for batch in data_generator:

            # Reshape to do in a single forward pass for all steps

            (logits, rnn_hidden_states, dist_entropy) = self.actor_critic(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
            )

            N = batch["recurrent_hidden_states"].shape[0]
            T = batch["actions"].shape[0] // N
            actions_batch = batch["actions"].view(T, N, -1)
            logits = logits.view(T, N, -1)

            action_loss = cross_entropy_loss(
                logits.permute(0, 2, 1), actions_batch.squeeze(-1)
            )
            entropy_term = dist_entropy * self.entropy_coef

            self.optimizer.zero_grad()
            inflections_batch = batch["observations"][
                "inflection_weight"
            ].view(T, N, -1)

            action_loss_term = (
                (inflections_batch * action_loss.unsqueeze(-1)).sum(0)
                / inflections_batch.sum(0)
            ).mean()
            total_loss = action_loss_term - entropy_term

            self.before_backward(total_loss)
            total_loss.backward()
            self.after_backward(total_loss)

            self.before_step()
            self.optimizer.step()
            self.after_step()

            total_loss_epoch += total_loss.item()
            total_action_loss += action_loss_term.item()
            total_entropy += dist_entropy.item()
            hidden_states.append(rnn_hidden_states)

        profiling_wrapper.range_pop()

        hidden_states = torch.cat(hidden_states, dim=0).detach()

        total_loss_epoch /= self.num_mini_batch
        total_entropy /= self.num_mini_batch
        total_action_loss /= self.num_mini_batch

        return (
            total_loss_epoch,
            hidden_states,
            total_entropy,
            total_action_loss,
        )

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass

class SemanticILAgent(nn.Module):
    def __init__(
        self,
        actor_critic: nn.Module,
        num_envs: int,
        num_mini_batch: int,
        lr: Optional[float] = None,
        encoder_lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        wd: Optional[float] = None,
        optimizer: Optional[str] = "AdamW",
        entropy_coef: Optional[float] = 0.0,
    ) -> None:

        super().__init__()

        self.contador = 0

        self.actor_critic = actor_critic

        self.num_mini_batch = num_mini_batch

        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs
        self.entropy_coef = entropy_coef

        # use different lr for visual encoder and other networks
        visual_encoder_params, other_params = [], []
        for name, param in actor_critic.named_parameters():
            if param.requires_grad:
                if "net.visual_encoder.backbone" in name:
                    visual_encoder_params.append(param)
                else:
                    other_params.append(param)
        logger.info(
            "Visual Encoder params: {}".format(len(visual_encoder_params))
        )
        logger.info("Other params: {}".format(len(other_params)))

        if optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                [
                    {"params": visual_encoder_params, "lr": encoder_lr},
                    {"params": other_params, "lr": lr},
                ],
                lr=lr,
                eps=eps,
                weight_decay=wd,
            )
        else:
            self.optimizer = optim.Adam(
                list(
                    filter(
                        lambda p: p.requires_grad, actor_critic.parameters()
                    )
                ),
                lr=lr,
                eps=eps,
            )
        self.device = torch.device("cpu")

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts) -> Tuple[float, float, float]:
        total_loss_epoch = 0.0
        total_entropy = 0.0
        total_action_loss = 0.0

        profiling_wrapper.range_push("BC.update epoch")
        data_generator = rollouts.recurrent_generator(self.num_mini_batch)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        hidden_states = []


        for batch in data_generator:
           # for image in batch["observations"]["semantic_rgb"]:
           #     cv2.imshow('Logo OpenCV',image.detach().cpu().numpy())
           #     cv2.waitKey(500)

            # Reshape to do in a single forward pass for all steps

            constant = 414534
            #constant = 9994

            #print(batch["observations"]["rgb"])
            #print(torch.any(batch["observations"]["semantic"]))
            observations_mult = batch["observations"]["semantic"]*constant


            rgb_matrix = torch.zeros((observations_mult.size(0), 480, 640, 3), dtype=torch.uint8, device=observations_mult.device)
            rgb_matrix[:, :, :, 0] = (observations_mult[:, :, :, 0] >> 16) & 0xFF  # R
            rgb_matrix[:, :, :, 1] = (observations_mult[:, :, :, 0] >> 8) & 0xFF  # G
            rgb_matrix[:, :, :, 2] = observations_mult[:, :, :, 0] & 0xFF  # B


            batch["observations"]["semantic_rgb"] = rgb_matrix
            # for num_im in range(batch["observations"]["semantic_rgb"].size(0)):
            #     print(batch["observations"]["semantic"][num_im])
            #     print(batch["observations"]["depth"][num_im])
           # # Uncomment the following code if you want to obtain train images for a dataset
           #  for num_im in range(batch["observations"]["semantic_rgb"].size(0)):
           # #     cv2.imshow('Logo OpenCV',image.detach().cpu().numpy())
           # #     cv2.waitKey(500)
           #
           #      self.contador += 1
           #      ruta_imagen_rgb_sem =  str(self.contador) + "semantic.png"
           #      ruta_imagen_rgb_sem = os.path.join("train_images_depthcorr","semantic", ruta_imagen_rgb_sem)
           #      # ruta_imagen_depth = str(self.contador) + "depth.png"
           #      # ruta_imagen_depth = os.path.join("train_images_depthcorr","depth", ruta_imagen_depth)
           #      image = batch["observations"]["semantic_rgb"][num_im]
           #      plt.imsave(ruta_imagen_rgb_sem, image.detach().cpu().numpy())
           #      # depth_image = batch["observations"]["depth"][num_im]*65535
           #      # cv2.imwrite(ruta_imagen_depth, depth_image.detach().cpu().numpy().astype(np.uint16))
           #      ruta_imagen_rgb = str(self.contador) + "rgb.png"
           #      ruta_imagen_rgb = os.path.join("train_images_depthcorr","rgb", ruta_imagen_rgb)
           #      image = batch["observations"]["rgb"][num_im]
           #      plt.imsave(ruta_imagen_rgb, image.detach().cpu().numpy())
           #      print(self.contador)


            (logits, rnn_hidden_states, dist_entropy) = self.actor_critic(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
            )

            N = batch["recurrent_hidden_states"].shape[0]
            T = batch["actions"].shape[0] // N
            actions_batch = batch["actions"].view(T, N, -1)
            logits = logits.view(T, N, -1)

            action_loss = cross_entropy_loss(
                logits.permute(0, 2, 1), actions_batch.squeeze(-1)
            )
            entropy_term = dist_entropy * self.entropy_coef

            self.optimizer.zero_grad()
            inflections_batch = batch["observations"][
                "inflection_weight"
            ].view(T, N, -1)

            action_loss_term = (
                (inflections_batch * action_loss.unsqueeze(-1)).sum(0)
                / inflections_batch.sum(0)
            ).mean()
            total_loss = action_loss_term - entropy_term

            self.before_backward(total_loss)
            total_loss.backward()
            self.after_backward(total_loss)

            self.before_step()
            self.optimizer.step()
            self.after_step()

            total_loss_epoch += total_loss.item()
            total_action_loss += action_loss_term.item()
            total_entropy += dist_entropy.item()
            hidden_states.append(rnn_hidden_states)

        profiling_wrapper.range_pop()

        hidden_states = torch.cat(hidden_states, dim=0).detach()

        total_loss_epoch /= self.num_mini_batch
        total_entropy /= self.num_mini_batch
        total_action_loss /= self.num_mini_batch

        return (
            total_loss_epoch,
            hidden_states,
            total_entropy,
            total_action_loss,
        )

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass


EPS_PPO = 1e-5


class DecentralizedDistributedMixin:
    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model

        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model

        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, actor_critic, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        actor_critic, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        actor_critic
                    )

        self._ddp_hooks = Guard(self.actor_critic, self.device)  # type: ignore
        # self.get_advantages = self._get_advantages_distributed

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss: Tensor) -> None:
        super().before_backward(loss)  # type: ignore

        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])  # type: ignore
        else:
            self.reducer.prepare_for_backward([])  # type: ignore


class Semantic_DDPILAgent(DecentralizedDistributedMixin, SemanticILAgent):
    pass
class DDPILAgent(DecentralizedDistributedMixin, ILAgent):
    pass
