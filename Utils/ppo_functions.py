import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import sympy
import ast
import copy

sys.path.append("SymbolicMathematics")
from SymbolicMathematics.src.envs import char_sp
from SymbolicMathematics.main import get_parser
from sympy import Symbol, Basic, sympify
from sympy import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import transformers
#from sympy.integrals import manualintegrate
import SymbolicMathematics
from transformers import DataCollatorWithPadding
import manualintegrate
import numexpr
import tqdm
from torch.utils.data import TensorDataset
from SymbolicMathematics.src.envs import char_sp
from SymbolicMathematics.main import get_parser

from transformer_utils import *


def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)
    returns = torch.tensor(returns)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    else:
        returns = returns * 0
    return returns


def calculate_advantages(returns, values):
    values = values.squeeze()
    advantages = returns - values

    if advantages.ndim == 0:
        advantages = advantages.unsqueeze(0)

    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages * 0
    return advantages


def calculate_surrogate_loss(
    actions_log_probability_old,
    actions_log_probability_new,
    epsilon,
    adv):

    if adv.ndim == 0:
        adv = adv.unsqueeze(0)

    policy_ratio = (actions_log_probability_new - actions_log_probability_old).exp()
    surrogate_loss_1 = policy_ratio * adv
    surrogate_loss_2 = torch.clamp(policy_ratio, min=1.0 - epsilon, max=1.0 + epsilon) * adv
    surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
    return surrogate_loss


def calculate_losses(
        surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
    returns = torch.nan_to_num(returns, nan=0.0, posinf=1e6, neginf=-1e6)
    value_pred = torch.nan_to_num(value_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    entropy_bonus = entropy_coefficient * entropy
    total_loss = surrogate_loss + entropy_bonus
    policy_loss = -total_loss.sum()

    value_loss = torch.nn.functional.smooth_l1_loss(returns, value_pred).sum()
    return policy_loss, value_loss


def update_policy(
        agent,
        token_ids,
        attention_mask,
        actions,
        actions_log_probability_old,
        advantages,
        returns,
        optimizer,
        ppo_steps,
        epsilon,
        entropy_coefficient):
    
    BATCH_SIZE = 16
    total_policy_loss = 0
    total_value_loss = 0
    actions_log_probability_old = actions_log_probability_old.detach()
    actions = actions.detach()
    device = next(agent.parameters()).device  

    token_ids = token_ids.to(device)
    attention_mask = attention_mask.to(device)
    actions = actions.to(device)
    returns = returns.to(device)
    advantages = advantages.to(device)

    returns    = returns.view(-1)     
    advantages = advantages.squeeze(-1) 
    if advantages.ndim == 0:
        advantages = advantages.unsqueeze(0)
    if actions.ndim == 0:
        actions = actions.unsqueeze(0)

    training_results_dataset = TensorDataset(
        token_ids,
        attention_mask,
        actions,
        actions_log_probability_old,
        advantages,
        returns,
    )

    batch_dataset = DataLoader(
            training_results_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False)

    for _ in range(ppo_steps):
        for token_ids, attention_mask, actions, log_probs_old, adv, ret in batch_dataset:
            
            device = next(agent.parameters()).device
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            actions = actions.to(device)
            log_probs_old = log_probs_old.to(device)
            adv = adv.to(device)
            ret = ret.to(device)
        
            attention_mask = attention_mask.squeeze(1)
            action_pred, value_pred = agent(token_ids, attention_mask)
            value_pred = value_pred.squeeze(-1)
            adv = adv.squeeze(-1)
    
            probs = torch.nn.functional.softmax(action_pred, dim=-1)
            dist  = torch.distributions.Categorical(probs)
            entropy = dist.entropy()
            log_probs_new = dist.log_prob(actions)

            surrogate = calculate_surrogate_loss(
                log_probs_old,
                log_probs_new,
                epsilon,
                adv
            )

            policy_loss, value_loss = calculate_losses(
                surrogate,
                entropy,
                entropy_coefficient,
                ret,
                value_pred
            )
    
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    return total_policy_loss, total_value_loss