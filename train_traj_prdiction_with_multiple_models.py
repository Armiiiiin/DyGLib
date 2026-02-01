"""
Fast weight added
MR@2m penalty added



DyGFormer traj prediction with cig alg

1. map fusion (ALG)
2. multi-modal (k traj) 
3. temporal weighting for loss 
4. TARGET AGENT only eval


V2X-Graph ALG

1. lane_vectors in agent local coord sys
2. lane_actor_vector = agent to lane relative pos
3. semantics: intersection, turn dir, traffic
4. rotation invariance
5. gate


History Encoder

1. rotation invariance
2. transformer instead of mlp

"""

import os
import sys
import time
import json
import pickle
import logging
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.DataLoader import Data, get_link_prediction_data, get_idx_data_loader
import matplotlib.pyplot as plt

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.MambaEncoder import MambaEncoder  
from utils.utils import (set_random_seed, convert_to_gpu, get_neighbor_sampler, create_optimizer)
from utils.DataLoader import get_link_prediction_data, get_idx_data_loader
from utils.EarlyStopping import EarlyStopping
import pandas as pd
from utils.DataLoader import Data


warnings.filterwarnings('ignore')


# > Part 1: Decoders 


class MLPTrajectoryDecoder(nn.Module):
    
    def __init__(self, embed_dim: int, pred_horizon: int = 30, 
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pred_horizon = pred_horizon
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 5, hidden_dim),  # +5 for current state
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_horizon * 2)
        )
        
    def forward(self, node_embed, current_state=None, **kwargs):
        B = node_embed.size(0)
    
        if current_state is not None:
            x = torch.cat([node_embed, current_state], dim=-1)
        else:
            x = node_embed
        
        offsets = self.mlp(x).view(B, self.pred_horizon, 2)
    
        # > the original point is already (0, 0)
        trajectory = torch.cumsum(offsets, dim=1)
    
        return {'trajectory': trajectory, 'offsets': offsets}


class HistoryEncoder(nn.Module):

    
    def __init__(self, hist_horizon: int = 50, input_dim: int = 4, 
                 embed_dim: int = 128, num_heads: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hist_horizon = hist_horizon
        self.embed_dim = embed_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # > positional encoding 
        self.pos_embed = nn.Parameter(torch.randn(1, hist_horizon + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # > causal mask
        self.register_buffer('causal_mask', self._generate_causal_mask(hist_horizon + 1))
        
    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(self, hist_traj: torch.Tensor, heading: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hist_traj: history traj (x, y, vx, vy) in relative coords
            heading: agent heading for rotation invariance
        """
        B, T, _ = hist_traj.shape
        
        # > rotation invariance
        if heading is not None:
            cos_h = torch.cos(heading).view(B, 1, 1)
            sin_h = torch.sin(heading).view(B, 1, 1)
            
            # > rotate pos (x, y)
            x = hist_traj[:, :, 0:1]
            y = hist_traj[:, :, 1:2]
            rot_x = cos_h * x + sin_h * y
            rot_y = -sin_h * x + cos_h * y
            
            # > rotate v (vx, vy)
            vx = hist_traj[:, :, 2:3]
            vy = hist_traj[:, :, 3:4]
            rot_vx = cos_h * vx + sin_h * vy
            rot_vy = -sin_h * vx + cos_h * vy
            
            hist_traj = torch.cat([rot_x, rot_y, rot_vx, rot_vy], dim=-1)
        
        x = self.input_proj(hist_traj)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # > add positional encoding
        x = x + self.pos_embed[:, :T+1, :]
        
        # > transformer with causal mask
        x = self.transformer(x, mask=self.causal_mask[:T+1, :T+1])
        
        return x[:, 0, :]


class InterModalSelfAttention(nn.Module):
    """
    from Co-MTP
    Competitive relationship among modes: when one mode's probability increases others should decrease
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=False 
            ) for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        
        for i in range(self.num_layers):
            # > self-attention among K modes
            # > each mode attends to all other modes
            attn_out, _ = self.attn_layers[i](x, x, x)
            x = self.norms[i](x + attn_out)
            
            # > FFN
            ffn_out = self.ffns[i](x)
            x = self.ffn_norms[i](x + ffn_out)
        
        return x


class MultiModalMLPDecoder(nn.Module):
    """
    k trajs from multihead proj
    
    with inter-modal self-attention
    """
    
    def __init__(self, embed_dim: int, pred_horizon: int = 50, 
                 hidden_dim: int = 256, num_modes: int = 6, dropout: float = 0.1,
                 use_history: bool = True, hist_horizon: int = 50,
                 hist_encoder_type: str = 'mlp',
                 use_intermodal_attn: bool = False, intermodal_attn_layers: int = 2):
        super().__init__()
        self.use_history = use_history
        self.hist_encoder_type = hist_encoder_type
        self.use_intermodal_attn = use_intermodal_attn
        
        if use_history:
            if hist_encoder_type == 'transformer':
                # > v2x graph motionencoder
                self.hist_encoder = HistoryEncoder(
                    hist_horizon=hist_horizon,
                    input_dim=4,
                    embed_dim=hidden_dim,
                    num_heads=8,
                    num_layers=4,
                    dropout=dropout
                )
            else:
                # > fall back to mlp
                self.hist_encoder = nn.Sequential(
                    nn.Linear(hist_horizon * 4, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            input_dim = embed_dim + 5 + hidden_dim  # > embed + current + history
        else:
            input_dim = embed_dim + 5
        
        self.pred_horizon = pred_horizon
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        
        
        # > multihead proj: generate k different hidden
        self.multihead_proj = nn.Linear(input_dim, num_modes * hidden_dim)
        
        if use_intermodal_attn:
            self.intermodal_attn = InterModalSelfAttention(
                hidden_dim=hidden_dim,
                num_heads=4,
                num_layers=intermodal_attn_layers,
                dropout=dropout
            )
        
        # > shared traj output layer
        self.loc_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_horizon * 2)
        )
        
        self.pi_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_embed, current_state=None, history_traj=None, **kwargs):
        B = node_embed.size(0)
        
        if current_state is not None:
            x = torch.cat([node_embed, current_state], dim=-1)
        else:
            x = node_embed
            
        if self.use_history and history_traj is not None:
            if self.hist_encoder_type == 'transformer':
                # > pass heading for rot invariance
                heading = current_state[:, 4] if current_state is not None else None
                hist_enc = self.hist_encoder(history_traj, heading=heading)
            else:
                # > mlp flatten input
                hist_flat = history_traj.view(B, -1) 
                hist_enc = self.hist_encoder(hist_flat)
            x = torch.cat([x, hist_enc], dim=-1)
        
        # > project to K modes
        x_multi = self.multihead_proj(x).view(B, self.num_modes, self.hidden_dim).transpose(0, 1)
        
        # > inter-modal self-attention
        # > K modes attend to each other
        if self.use_intermodal_attn:
            x_multi = self.intermodal_attn(x_multi)
        
        offsets = self.loc_head(x_multi).view(self.num_modes, B, self.pred_horizon, 2)
        trajectories = torch.cumsum(offsets, dim=2).transpose(0, 1)
        
        pi = self.pi_head(x_multi).squeeze(-1).t()
        
        return {
            'trajectories': trajectories,
            'confidences': pi,
            'trajectory': trajectories[:, 0]
        }

def create_decoder(embed_dim: int, pred_horizon: int = 30, 
                   num_modes: int = 1, **kwargs):
    if num_modes > 1:
        return MultiModalMLPDecoder(embed_dim, pred_horizon, num_modes=num_modes, **kwargs)
    return MLPTrajectoryDecoder(embed_dim, pred_horizon, **kwargs)

def get_embeddings(encoder, src, dst, times, edge_ids=None, 
                  history=None, current=None, needs_edge_ids=False, needs_history=False):
    """Unified function to get embeddings for any encoder type"""
    if needs_edge_ids:
        return encoder.compute_src_dst_node_temporal_embeddings(
            src, dst, times, edge_ids,
            history_traj=history if needs_history else None,
            current_state=current if needs_history else None
        )
    else:
        return encoder.compute_src_dst_node_temporal_embeddings(
            src, dst, times,
            history_traj=history if needs_history else None,
            current_state=current if needs_history else None
        )
# > Part 1.5: Map Fusion Modules (V2X-Graph ALG Style)


class LaneEncoder(nn.Module):
    """
    alg
    """
    
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.lane_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # > semantic embeddings
        self.is_intersection_embed = nn.Embedding(2, hidden_dim) 
        self.turn_direction_embed = nn.Embedding(4, hidden_dim) # > 0 = none, 1 = left, 2 = right, 3 = u turn
        self.traffic_control_embed = nn.Embedding(2, hidden_dim)
        
        self.aggr_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.is_intersection_embed.weight, mean=0., std=0.02)
        nn.init.normal_(self.turn_direction_embed.weight, mean=0., std=0.02)
        nn.init.normal_(self.traffic_control_embed.weight, mean=0., std=0.02)
    
    def forward(self, lane_vectors, lane_actor_vectors=None, 
                is_intersections=None, turn_directions=None, traffic_controls=None,
                lane_mask=None, lane_points=None):

        # > if only lane points
        if lane_vectors is None and lane_points is not None:
            B, N_lanes, N_points, _ = lane_points.shape
            lane_vectors = lane_points[:, :, -1, :] - lane_points[:, :, 0, :]  # [B, L, 2]
            lane_actor_vectors = lane_points[:, :, 0, :]  # [B, L, 2]
            is_intersections = torch.zeros(B, N_lanes, dtype=torch.long, device=lane_points.device)
            turn_directions = torch.zeros(B, N_lanes, dtype=torch.long, device=lane_points.device)
            traffic_controls = torch.zeros(B, N_lanes, dtype=torch.long, device=lane_points.device)
        
        # > alg
        lane_embed = self.lane_encoder(lane_vectors)  # [B, L, D]
        
        if lane_actor_vectors is not None:
            edge_embed = self.edge_encoder(lane_actor_vectors)  # [B, L, D]
        else:
            edge_embed = torch.zeros_like(lane_embed)
        
        if is_intersections is not None:
            is_inter_embed = self.is_intersection_embed(is_intersections.clamp(0, 1))
        else:
            is_inter_embed = torch.zeros_like(lane_embed)
            
        if turn_directions is not None:
            turn_dir_embed = self.turn_direction_embed(turn_directions.clamp(0, 3))
        else:
            turn_dir_embed = torch.zeros_like(lane_embed)
            
        if traffic_controls is not None:
            traffic_embed = self.traffic_control_embed(traffic_controls.clamp(0, 1))
        else:
            traffic_embed = torch.zeros_like(lane_embed)
        
        fused = lane_embed + edge_embed + is_inter_embed + turn_dir_embed + traffic_embed
        
        lane_embeds = self.aggr_embed(fused) 
        
        return lane_embeds


class MapFusionModule(nn.Module):
    """
    1. multi head attention (agent as query, lane as key/value)
    2. gate
    3. ffn + residual
    """
    
    def __init__(self, agent_dim: int, lane_dim: int = 128, 
                 num_heads: int = 4, dropout: float = 0.1,
                 use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.lane_dim = lane_dim
        self.num_heads = num_heads
        self.head_dim = lane_dim // num_heads
        
        self.agent_proj = nn.Linear(agent_dim, lane_dim) if agent_dim != lane_dim else nn.Identity()
        self.output_proj = nn.Linear(lane_dim, agent_dim)
        
        # > multi head attention
        self.lin_q = nn.Linear(lane_dim, lane_dim)
        self.lin_k = nn.Linear(lane_dim, lane_dim)
        self.lin_v = nn.Linear(lane_dim, lane_dim)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
        # > gate
        if use_gate:
            self.lin_ih = nn.Linear(lane_dim, lane_dim)
            self.lin_hh = nn.Linear(lane_dim, lane_dim)
            self.lin_self = nn.Linear(lane_dim, lane_dim)
        
        # > ffn
        self.norm1 = nn.LayerNorm(lane_dim)
        self.norm2 = nn.LayerNorm(lane_dim)
        self.mlp = nn.Sequential(
            nn.Linear(lane_dim, lane_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lane_dim * 4, lane_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, agent_embed, lane_embeds, lane_mask=None):

        if lane_mask is None:
            return agent_embed
    
        has_lanes = lane_mask.any(dim=1)
    
        if not has_lanes.any():
            return agent_embed
    
        B, L, D = lane_embeds.shape
        fused_embed = agent_embed.clone()
        
        if has_lanes.all():
            valid_agent = self.agent_proj(agent_embed)
            valid_lanes = lane_embeds
            valid_mask = lane_mask
            valid_idx = None
        else:
            valid_idx = has_lanes.nonzero(as_tuple=True)[0]
            valid_agent = self.agent_proj(agent_embed[valid_idx])
            valid_lanes = lane_embeds[valid_idx]
            valid_mask = lane_mask[valid_idx]
        
        B_valid = valid_agent.size(0)
        
        # > multi head attention
        q = self.lin_q(valid_agent).view(B_valid, 1, self.num_heads, self.head_dim)
        k = self.lin_k(valid_lanes).view(B_valid, L, self.num_heads, self.head_dim)
        v = self.lin_v(valid_lanes).view(B_valid, L, self.num_heads, self.head_dim)
        
        scale = self.head_dim ** 0.5
        attn = (q * k).sum(dim=-1) / scale  
        
        if valid_mask is not None:
            attn_mask = ~valid_mask.unsqueeze(-1).expand(-1, -1, self.num_heads)
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=1)
        attn = self.attn_drop(attn)
        
        attended = (attn.unsqueeze(-1) * v).sum(dim=1).view(B_valid, -1)
        
        # > gate
        if self.use_gate:
            gate = torch.sigmoid(self.lin_ih(attended) + self.lin_hh(valid_agent))
            updated = attended + gate * (self.lin_self(valid_agent) - attended)
        else:
            updated = attended
        
        updated = self.proj_drop(updated)
        
        x = valid_agent + updated
        x = x + self.mlp(self.norm2(x))
        
        output = self.output_proj(x)
        
        if valid_idx is None:
            fused_embed = output + agent_embed
        else:
            fused_embed[valid_idx] = output + agent_embed[valid_idx]
    
        return fused_embed


# > part 2: loss function and eval metrics

class MultiModalLoss(nn.Module):
    
    def __init__(self, fde_weight=0.5, time_weight_type='none', cls_weight=1.0,
                 fast_weight=1.0, mr_penalty_weight=0.0):
        super().__init__()
        self.fde_weight = fde_weight
        self.time_weight_type = time_weight_type  # 'none', 'linear', 'exp'
        self.cls_weight = cls_weight
        self.fast_weight = fast_weight
        self.mr_penalty_weight = mr_penalty_weight
    
    def _get_time_weights(self, T, device):
        if self.time_weight_type == 'linear':
            weights = torch.arange(1, T + 1, dtype=torch.float32, device=device)
            weights = weights / weights.sum() * T
        elif self.time_weight_type == 'exp':
            weights = torch.exp(torch.arange(T, dtype=torch.float32, device=device) / T)
            weights = weights / weights.sum() * T
        else:
            weights = torch.ones(T, device=device)
        return weights
        
    def forward(self, pred_trajs, gt_traj, confidences, current_state=None):
        B, K, T, _ = pred_trajs.shape
        device = pred_trajs.device
        
        time_weights = self._get_time_weights(T, device)
        
        gt_expanded = gt_traj.unsqueeze(1)
        l2_norm = torch.norm(pred_trajs - gt_expanded, p=2, dim=-1)  
        
        weighted_l2 = l2_norm * time_weights.view(1, 1, T)
        ade_per_mode = l2_norm.mean(dim=-1) 
        weighted_ade_per_mode = weighted_l2.mean(dim=-1)  
        fde_per_mode = l2_norm[:, :, -1]    
        
        ade_best_mode = ade_per_mode.argmin(dim=1)
        min_ade = ade_per_mode[torch.arange(B, device=device), ade_best_mode]
        min_weighted_ade = weighted_ade_per_mode[torch.arange(B, device=device), ade_best_mode]
        
        fde_best_mode = fde_per_mode.argmin(dim=1)
        min_fde = fde_per_mode[torch.arange(B, device=device), fde_best_mode]
        
        # > speed weighting for fast vehicles (continuous weighting)
        if self.fast_weight != 1.0 and current_state is not None:
            speeds = torch.norm(current_state[:, 2:4], dim=-1)

            speed_weights = 1.0 + (self.fast_weight - 1.0) * torch.clamp(speeds / 15.0, 0, 1)
            speed_weights = speed_weights / speed_weights.mean()
        else:
            speed_weights = torch.ones(B, device=device)
        
        # > MR penalty: huber loss for fde > 2m
        if self.mr_penalty_weight > 0:
            mr_penalty = F.huber_loss(min_fde, torch.full_like(min_fde, 2.0), reduction='none', delta=1.0)
            mr_penalty = F.relu(min_fde - 2.0) * mr_penalty  # > effective only when fde > 2m
        else:
            mr_penalty = torch.zeros_like(min_fde)
        
        # > apply speed weight to all terms except mr penalty
        reg_loss = (min_weighted_ade * speed_weights).mean() + self.fde_weight * (min_fde * speed_weights).mean() + self.mr_penalty_weight * mr_penalty.mean()

        
        with torch.no_grad():
            soft_target = F.softmax(-ade_per_mode, dim=1)
        log_probs = F.log_softmax(confidences, dim=1)
        cls_loss = -(soft_target * log_probs).sum(dim=1).mean()
        
        total_loss = reg_loss + self.cls_weight * cls_loss
        
        return {
            'loss': total_loss,
            'reg_loss': reg_loss,
            'cls_loss': cls_loss,
            'ade': min_ade.mean(),
            'fde': min_fde.mean(),
            'min_ade': min_ade.mean(),
            'min_fde': min_fde.mean(),
        }

class TrajectoryMetrics:
    
    @staticmethod
    def compute_ade(pred, gt):
        return torch.norm(pred - gt, dim=-1).mean().item()
    
    @staticmethod
    def compute_fde(pred, gt):
        return torch.norm(pred[:, -1] - gt[:, -1], dim=-1).mean().item()
    
    @staticmethod
    def compute_miss_rate(pred, gt, threshold=2.0):
        fde = torch.norm(pred[:, -1] - gt[:, -1], dim=-1)
        return (fde > threshold).float().mean().item()
    
    @staticmethod
    def compute_direction_accuracy(pred, gt, threshold=0.5):
        gt_disp = gt[:, -1] - gt[:, 0]
        pred_disp = pred[:, -1] - pred[:, 0]
        gt_dist = torch.norm(gt_disp, dim=-1)
        moving_mask = gt_dist > threshold
    
        if moving_mask.sum() == 0:
            return 1.0
    
        cos_sim = F.cosine_similarity(pred_disp[moving_mask], gt_disp[moving_mask], dim=-1)
        return (cos_sim > 0.5).float().mean().item()
    
    @staticmethod
    def get_quality_rating(ade, fde):
    
        if ade < 0.5 and fde < 1.0:
            return "EXCELLENT", "ade < 0.5 and fde < 1.0"
        elif ade < 1.0 and fde < 2.0:
            return "GOOD", "ade < 1.0 and fde < 2.0"
        elif ade < 2.0 and fde < 4.0:
            return "ACCEPTABLE", "ade < 2.0 and fde < 4.0"
        else:
            return "POOR", "ade >= 2.0 and fde >= 4.0"


# > Part 3: explainability 

def visualize_predictions(predictions, ground_truth, current_states,
                         num_samples=10, save_path=None):

    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    n = min(num_samples, len(predictions))
    indices = np.random.choice(len(predictions), n, replace=False)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        pred = predictions[idx].cpu().numpy()
        gt = ground_truth[idx].cpu().numpy()
        curr = current_states[idx].cpu().numpy()
        
        # > current pos
        ax.scatter(curr[0], curr[1], c='green', s=100, marker='o', 
                  label='Current', zorder=5)
        
        # > gt
        ax.plot(gt[:, 0], gt[:, 1], 'b-', lw=2, label='GT', alpha=0.8)
        ax.scatter(gt[-1, 0], gt[-1, 1], c='blue', s=50, marker='x')
        
        # > predict
        ax.plot(pred[:, 0], pred[:, 1], 'r--', lw=2, label='Pred', alpha=0.8)
        ax.scatter(pred[-1, 0], pred[-1, 1], c='red', s=50, marker='x')
        
        # > error
        ade = np.linalg.norm(pred - gt, axis=1).mean()
        fde = np.linalg.norm(pred[-1] - gt[-1])
        rating, _ = TrajectoryMetrics.get_quality_rating(ade, fde)
        
        ax.set_title(f'ADE:{ade:.2f}m FDE:{fde:.2f}m [{rating}]', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def analyze_errors(predictions, ground_truth, current_states, save_path=None):

    errors = torch.norm(predictions - ground_truth, dim=-1) 
    ade_samples = errors.mean(dim=1)
    fde_samples = errors[:, -1]
    speeds = torch.norm(current_states[:, 2:4], dim=-1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # > error distribution
    ax = axes[0, 0]
    ax.hist(ade_samples.cpu().numpy(), bins=50, alpha=0.7, label='ADE')
    ax.hist(fde_samples.cpu().numpy(), bins=50, alpha=0.7, label='FDE')
    ax.set_xlabel('Error (m)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    
    # > error, velocity
    ax = axes[0, 1]
    ax.scatter(speeds.cpu().numpy(), ade_samples.cpu().numpy(), alpha=0.3, s=10)
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('ADE (m)')
    ax.set_title('Error vs Speed')
    
    # > error, time
    ax = axes[1, 0]
    mean_err = errors.mean(dim=0).cpu().numpy()
    ax.plot(range(len(mean_err)), mean_err, 'b-', lw=2)
    ax.fill_between(range(len(mean_err)), 
                    mean_err - errors.std(dim=0).cpu().numpy(),
                    mean_err + errors.std(dim=0).cpu().numpy(),
                    alpha=0.3)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Error (m)')
    ax.set_title('Error Accumulation Over Time')
    ax.grid(True, alpha=0.3)
    
    # 4> best vs worst
    ax = axes[1, 1]
    sorted_idx = torch.argsort(ade_samples)
    best = ade_samples[sorted_idx[:10]].mean().item()
    worst = ade_samples[sorted_idx[-10:]].mean().item()
    ax.bar(['Best 10', 'Worst 10'], [best, worst], color=['green', 'red'])
    ax.set_ylabel('ADE (m)')
    ax.set_title('Best vs Worst Predictions')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    
    slow = speeds < 5
    medium = (speeds >= 5) & (speeds < 15)
    fast = speeds >= 15
    
    results = {
        'ade_mean': ade_samples.mean().item(),
        'ade_std': ade_samples.std().item(),
        'fde_mean': fde_samples.mean().item(),
        'fde_std': fde_samples.std().item(),
    }
    
    if slow.sum() > 0:
        results['ade_slow'] = ade_samples[slow].mean().item()
    if medium.sum() > 0:
        results['ade_medium'] = ade_samples[medium].mean().item()
    if fast.sum() > 0:
        results['ade_fast'] = ade_samples[fast].mean().item()
    
    
    print("Error Analysis")
    print(f"ADE: {results['ade_mean']:.3f} ± {results['ade_std']:.3f} m")
    print(f"FDE: {results['fde_mean']:.3f} ± {results['fde_std']:.3f} m")
    if 'ade_slow' in results:
        print(f"Slow (<5m/s): ADE = {results['ade_slow']:.3f}m")
    if 'ade_medium' in results:
        print(f"Medium (5-15m/s): ADE = {results['ade_medium']:.3f}m")
    if 'ade_fast' in results:
        print(f"Fast (>15m/s): ADE = {results['ade_fast']:.3f}m")
    
    rating, expl = TrajectoryMetrics.get_quality_rating(
        results['ade_mean'], results['fde_mean']
    )
    print(f"\nQuality: {rating} - {expl}")
    
    return results


# > Part 4: load data

class SplitDataLoader:
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.splits = {}
        
        for split in ['train', 'val', 'test']:
            csv_path = self.data_dir / f'{split}.csv'
            if csv_path.exists():
                self.splits[split] = self._load_split(split)
                print(f"Loaded {split}: {len(self.splits[split]['edges'])} edges")
    
    def _load_split(self, split: str):
        
        
        # > edge list
        df = pd.read_csv(self.data_dir / f'{split}.csv')
        
        # > edge features
        edge_feats = np.load(self.data_dir / f'{split}.npy')
        
        # > node features
        node_feats = np.load(self.data_dir / f'{split}_node.npy')
        
        # > gt
        with open(self.data_dir / f'{split}_trajectory.pkl', 'rb') as f:
            traj_data = pickle.load(f)
        
        # > lane
        lane_path = self.data_dir / f'{split}_lanes.pkl'
        lane_data = None
        if lane_path.exists():
            with open(lane_path, 'rb') as f:
                lane_data = pickle.load(f)
        
        return {
            'edges': df,
            'edge_feats': edge_feats,
            'node_feats': node_feats,
            'traj_data': traj_data,
            'lane_data': lane_data
        }
    
    def get_split(self, split: str):
        return self.splits.get(split)
    
    def create_data_object(self, split: str):
         
        split_data = self.splits[split]
        df = split_data['edges']
        
        return Data(
            src_node_ids=df['u'].values.astype(np.int64),
            dst_node_ids=df['i'].values.astype(np.int64),
            node_interact_times=df['ts'].values.astype(np.float64),
            edge_ids=df['idx'].values.astype(np.int64),
            labels=df['label'].values.astype(np.int64)
        )
    
    def get_node_feats(self):
        max_feats = None
        for split_data in self.splits.values():
            feats = split_data['node_feats']
            if max_feats is None or feats.shape[0] > max_feats.shape[0]:
                max_feats = feats
        return max_feats
    
    def get_edge_feats(self):
        all_feats = []
        for split in ['train', 'val', 'test']:
            if split in self.splits:
                all_feats.append(self.splits[split]['edge_feats'])
        return np.vstack(all_feats)


class SplitTrajectoryLoader:    
    def __init__(self, split_data: dict):
        traj_data = split_data['traj_data']
        self.future_traj = traj_data['future_traj']
        self.current_state = traj_data['current_state']
        self.history_traj = traj_data.get('history_traj', {})
        self.is_target = traj_data.get('is_target', {})
        self.pred_horizon = traj_data['pred_horizon']
        self.hist_horizon = traj_data.get('hist_horizon', 50)
        
        target_count = sum(1 for v in self.is_target.values() if v)
        print(f"  Trajectory samples: {len(self.future_traj)}")
        print(f"  History samples: {len(self.history_traj)}")
        print(f"  TARGET_AGENT samples: {target_count}")
    
    def get_batch(self, node_ids, timestamps):

        B = len(node_ids)
        
        future = np.zeros((B, self.pred_horizon, 5), dtype=np.float32)
        current = np.zeros((B, 5), dtype=np.float32)
        history = np.zeros((B, self.hist_horizon, 4), dtype=np.float32)
        valid = np.zeros(B, dtype=bool)
        is_target = np.zeros(B, dtype=bool)
        
        for i, (nid, ts) in enumerate(zip(node_ids, timestamps)):
            key = (int(nid), float(ts))
            if key in self.future_traj:
                curr_state = self.current_state[key].copy()
                fut_traj = self.future_traj[key].copy()
                
                if np.isnan(fut_traj).any() or np.isnan(curr_state).any():
                    continue
                
                # > convert to relative coordinates
                curr_x, curr_y = curr_state[0], curr_state[1]
                fut_traj[:, 0] -= curr_x
                fut_traj[:, 1] -= curr_y
                
                if key in self.history_traj:
                    hist = self.history_traj[key].copy()
                    hist[:, 0] -= curr_x
                    hist[:, 1] -= curr_y
                    history[i] = hist
                    
                    
                curr_state[0] = 0
                curr_state[1] = 0
                
                future[i] = fut_traj
                current[i] = curr_state
                valid[i] = True
                is_target[i] = self.is_target.get(key, False)  
        
        return {
            'future_traj': torch.from_numpy(future),
            'current_state': torch.from_numpy(current),
            'history_traj': torch.from_numpy(history),
            'valid_mask': torch.from_numpy(valid),
            'is_target': torch.from_numpy(is_target)  
        }


class SplitLaneLoader:
    
    def __init__(self, split_data: dict):
        self.has_data = False
        self.is_alg_style = False
        
        if split_data.get('lane_data') is not None:
            lane_data = split_data['lane_data']
            self.lanes = lane_data['lanes']
            self.max_lanes = lane_data['max_lanes']
            self.points_per_lane = lane_data['points_per_lane']
            self.has_data = True
            
            # > if using v2xg alg
            self.is_alg_style = lane_data.get('style') == 'v2xg-alg'
            if self.is_alg_style:
                print(f"  Lane samples: {len(self.lanes)} (V2X-Graph ALG style)")
            else:
                print(f"  Lane samples: {len(self.lanes)}")
    
    def get_batch(self, node_ids, timestamps, current_states, device='cpu'):
        B = len(node_ids)
        
        if not self.has_data:
            return None, None
        
        if self.is_alg_style:
            return self._get_batch_alg(node_ids, timestamps, device)
        else:
            return self._get_batch_legacy(node_ids, timestamps, current_states, device)
    
    def _get_batch_alg(self, node_ids, timestamps, device):
        B = len(node_ids)
        
        lane_vectors = np.zeros((B, self.max_lanes, 2), dtype=np.float32)
        lane_actor_vectors = np.zeros((B, self.max_lanes, 2), dtype=np.float32)
        is_intersections = np.zeros((B, self.max_lanes), dtype=np.int64)
        turn_directions = np.zeros((B, self.max_lanes), dtype=np.int64)
        traffic_controls = np.zeros((B, self.max_lanes), dtype=np.int64)
        lane_mask = np.zeros((B, self.max_lanes), dtype=bool)
        
        for i, (nid, ts) in enumerate(zip(node_ids, timestamps)):
            key = (int(nid), float(ts))
            if key in self.lanes:
                lanes = self.lanes[key]
                n_lanes = min(lanes.get('num_lanes', 0), self.max_lanes)
                
                if n_lanes > 0:
                    lane_vectors[i, :n_lanes] = lanes['lane_vectors'][:n_lanes]
                    lane_actor_vectors[i, :n_lanes] = lanes['lane_actor_vectors'][:n_lanes]
                    is_intersections[i, :n_lanes] = lanes['is_intersections'][:n_lanes]
                    turn_directions[i, :n_lanes] = lanes['turn_directions'][:n_lanes]
                    traffic_controls[i, :n_lanes] = lanes['traffic_controls'][:n_lanes]
                    lane_mask[i, :n_lanes] = True
        
        return {
            'lane_vectors': torch.from_numpy(lane_vectors).to(device),
            'lane_actor_vectors': torch.from_numpy(lane_actor_vectors).to(device),
            'is_intersections': torch.from_numpy(is_intersections).to(device),
            'turn_directions': torch.from_numpy(turn_directions).to(device),
            'traffic_controls': torch.from_numpy(traffic_controls).to(device),
            'lane_mask': torch.from_numpy(lane_mask).to(device)
        }
    
    def _get_batch_legacy(self, node_ids, timestamps, current_states, device):
        """older version returns centerlines only"""
        B = len(node_ids)
        
        lane_points = np.zeros((B, self.max_lanes, self.points_per_lane, 2), dtype=np.float32)
        lane_mask = np.zeros((B, self.max_lanes), dtype=bool)
        
        for i, (nid, ts) in enumerate(zip(node_ids, timestamps)):
            key = (int(nid), float(ts))
            if key in self.lanes:
                lanes = self.lanes[key]
                centerlines = lanes['centerlines']
                n_lanes = min(len(centerlines), self.max_lanes)
                
                curr_x = current_states[i, 0]
                curr_y = current_states[i, 1]
                heading = current_states[i, 4]  
                
                # > rotation for local frame
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)
                
                for j in range(n_lanes):
                    # > agent center
                    dx = centerlines[j, :, 0] - curr_x
                    dy = centerlines[j, :, 1] - curr_y
                    # > rotate to agent's local frame
                    lane_points[i, j, :, 0] = cos_h * dx + sin_h * dy
                    lane_points[i, j, :, 1] = -sin_h * dx + cos_h * dy
                    lane_mask[i, j] = True
        
        return (torch.from_numpy(lane_points).to(device),
                torch.from_numpy(lane_mask).to(device))


# > Part 5: training

def get_args():
    parser = argparse.ArgumentParser('Trajectory Prediction')
    
    parser.add_argument('--dataset_name', type=str, default='v2x_traj')
    parser.add_argument('--data_dir', type=str, default='processed_data/v2x_full')
    parser.add_argument('--use_map', action='store_true')
    
    parser.add_argument('--pred_horizon', type=int, default=30)
    
    parser.add_argument('--num_neighbors', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--channel_embedding_dim', type=int, default=50)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model_name', type=str, default='DyGFormer',
                   choices=['DyGFormer', 'TGAT', 'TGN', 'JODIE', 'DyRep', 'CAWN', 'TCL', 'GraphMixer', 'Mamba'])
    
    parser.add_argument('--num_modes', type=int, default=1,
                       help='Number of traj modes (1=single, 6=v2x-graph setting)')
    parser.add_argument('--fde_weight', type=float, default=0.5)
    parser.add_argument('--time_weight', type=str, default='none',
                       choices=['none', 'linear', 'exp'])
    parser.add_argument('--cls_weight', type=float, default=1.0)
    parser.add_argument('--fast_weight', type=float, default=1.0)
    parser.add_argument('--mr_penalty_weight', type=float, default=0.0)
    parser.add_argument('--hist_encoder_type', type=str, default='mlp',
                       choices=['mlp', 'transformer'])
    parser.add_argument('--use_intermodal_attn', action='store_true')
    parser.add_argument('--intermodal_attn_layers', type=int, default=2)

    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    
    parser.add_argument('--gpu', type=int, default=0)
    
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vis_interval', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--target_only', action='store_true')
    
    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    return args


def main():
    args = get_args()
    
    print(f"Prediction horizon: {args.pred_horizon} steps")
    print(f"Map fusion: {args.use_map}")
    print(f"Target-only eval: {args.target_only}")
    print(f"Device: {args.device}")
    
    
    set_random_seed(89)
    
    print("\nLoading data splits")
    split_loader = SplitDataLoader(args.data_dir)
    
    train_data = split_loader.create_data_object('train')
    val_data = split_loader.create_data_object('val')
    if 'test' in split_loader.splits:
        test_data = split_loader.create_data_object('test')
    else:
        test_data = None
    
    node_feats = split_loader.get_node_feats()
    edge_feats = split_loader.get_edge_feats()
    
    print(f"\nData split summary:")
    print(f"  Train: {len(train_data.src_node_ids):,} samples")
    print(f"  Val: {len(val_data.src_node_ids):,} samples")
    if test_data is not None:
        print(f"  Test: {len(test_data.src_node_ids):,} samples")
    else:
        print(f"  Test: No test data")

    
    print("\nLoading traj data")
    train_traj_loader = SplitTrajectoryLoader(split_loader.get_split('train'))
    val_traj_loader = SplitTrajectoryLoader(split_loader.get_split('val'))
    if 'test' in split_loader.splits:
        test_traj_loader = SplitTrajectoryLoader(split_loader.get_split('test'))
    else:
        test_traj_loader = None
    
    train_lane_loader = None
    val_lane_loader = None
    test_lane_loader = None
    if args.use_map:
        print("\nLoading lane data")
        train_lane_loader = SplitLaneLoader(split_loader.get_split('train'))
        val_lane_loader = SplitLaneLoader(split_loader.get_split('val'))
        test_lane_loader = SplitLaneLoader(split_loader.get_split('test'))
    
    # > full_data for neighbor sampler
    # > neighbor sampler needs the entire history (train + val)
    src_list = [train_data.src_node_ids, val_data.src_node_ids]
    dst_list = [train_data.dst_node_ids, val_data.dst_node_ids]
    times_list = [train_data.node_interact_times, val_data.node_interact_times]
    labels_list = [train_data.labels, val_data.labels]

    edge_offset = train_data.edge_ids.max() + 1
    edge_list = [train_data.edge_ids, val_data.edge_ids + edge_offset]

    if test_data is not None:
        src_list.append(test_data.src_node_ids)
        dst_list.append(test_data.dst_node_ids)
        times_list.append(test_data.node_interact_times)
        labels_list.append(test_data.labels)
        edge_offset += val_data.edge_ids.max() + 1
        edge_list.append(test_data.edge_ids + edge_offset)

    full_data = Data(
        src_node_ids=np.concatenate(src_list),
        dst_node_ids=np.concatenate(dst_list),
        node_interact_times=np.concatenate(times_list),
        edge_ids=np.concatenate(edge_list),
        labels=np.concatenate(labels_list)
    )
    
    neighbor_sampler = get_neighbor_sampler(full_data, 'recent', 1e-6, 42)
    
    train_loader = get_idx_data_loader(
        list(range(len(train_data.src_node_ids))), 
        args.batch_size, shuffle=True
    )
    val_loader = get_idx_data_loader(
        list(range(len(val_data.src_node_ids))),
        args.batch_size, shuffle=False
    )
    if test_data is not None:
        test_loader = get_idx_data_loader(
            list(range(len(test_data.src_node_ids))),
            args.batch_size, shuffle=False
        )
    else:
        test_loader = None
    
    print(f"\nUsing encoder: {args.model_name}")
    
    if args.model_name == 'DyGFormer':
        encoder = DyGFormer(
            node_raw_features=node_feats,
            edge_raw_features=edge_feats,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=100,
            channel_embedding_dim=args.channel_embedding_dim,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            max_input_sequence_length=512,
            device=args.device
        )
    elif args.model_name in ['TGN', 'JODIE', 'DyRep']:
        encoder = MemoryModel(
            node_raw_features=node_feats,
            edge_raw_features=edge_feats,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=100,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            model_name=args.model_name,
            device=args.device
        )
    elif args.model_name == 'TGAT':
        encoder = TGAT(
            node_raw_features=node_feats,
            edge_raw_features=edge_feats,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=100,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=args.device
        )
    elif args.model_name == 'CAWN':
        encoder = CAWN(
            node_raw_features=node_feats,
            edge_raw_features=edge_feats,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=100,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=args.device
        )
    elif args.model_name == 'TCL':
        encoder = TCL(
            node_raw_features=node_feats,
            edge_raw_features=edge_feats,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=100,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_depths=args.num_neighbors, 
            dropout=args.dropout,
            device=args.device
        )
    elif args.model_name == 'GraphMixer':
        encoder = GraphMixer(
            node_raw_features=node_feats,
            edge_raw_features=edge_feats,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=100,
            num_tokens=args.num_neighbors,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=args.device
        )
    elif args.model_name == 'Mamba':  
        print("Initializing Mamba encoder...")
        encoder = MambaEncoder(
            hist_horizon=train_traj_loader.hist_horizon,  
            node_dim=4,  
            embed_dim=args.channel_embedding_dim * args.num_heads,  
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=args.device
        )
        
        print(f"Mamba encoder initialized with:")
        print(f"  hist_horizon: {train_traj_loader.hist_horizon}")
        print(f"  embed_dim: {args.channel_embedding_dim}")
        print(f"  num_layers: {args.num_layers}")
    else:
        raise ValueError(f"Unknown model: {args.model_name}")
    
    encoder = convert_to_gpu(encoder, device=args.device)
    
    needs_edge_ids = args.model_name in ['TGN', 'JODIE', 'DyRep']
    needs_history = args.model_name == 'Mamba'
    
    print(f"Using encoder: {type(encoder).__name__}")
    print(f"needs_edge_ids: {needs_edge_ids}")
    print(f"needs_history: {needs_history}")
    # > backup memory before forward
    initial_memory_backup = None
    if needs_edge_ids and hasattr(encoder, 'memory_bank'):
        initial_memory_backup = encoder.memory_bank.backup_memory_bank()

    with torch.no_grad():
        test_src = train_data.src_node_ids[:2]
        test_dst = train_data.dst_node_ids[:2]
        test_times = train_data.node_interact_times[:2]
        test_edge_ids = train_data.edge_ids[:2]
        if needs_edge_ids:
            test_embed, _ = encoder.compute_src_dst_node_temporal_embeddings(
                test_src, test_dst, test_times, test_edge_ids
            )
        else:
            test_embed, _ = encoder.compute_src_dst_node_temporal_embeddings(
                test_src, test_dst, test_times
            )
        embed_dim = test_embed.shape[1]
        print(f"node feature dim: {node_feats.shape[1]}")
        print(f"actual embedding dim: {embed_dim}")
        if torch.isnan(test_embed).any():
            print("[WARNING] NAN in encoder output")
        else:
            print(f"Embedding range: [{test_embed.min():.4f}, {test_embed.max():.4f}]")
    
    # > reset to initial memory
    if needs_edge_ids and initial_memory_backup is not None:
        encoder.memory_bank.reload_memory_bank(initial_memory_backup)
    
    lane_encoder = None
    map_fusion = None
    if args.use_map and train_lane_loader is not None and train_lane_loader.has_data:

        lane_encoder = LaneEncoder(hidden_dim=128)
        lane_encoder = convert_to_gpu(lane_encoder, device=args.device)
        map_fusion = MapFusionModule(agent_dim=embed_dim, lane_dim=128)
        map_fusion = convert_to_gpu(map_fusion, device=args.device)
        print("Map fusion modules initialized")
    
    decoder = create_decoder(embed_dim, args.pred_horizon, num_modes=args.num_modes,
                             hist_encoder_type=args.hist_encoder_type,
                             use_intermodal_attn=args.use_intermodal_attn,
                             intermodal_attn_layers=args.intermodal_attn_layers)
    decoder = convert_to_gpu(decoder, device=args.device)

    loss_fn = MultiModalLoss(
        fde_weight=args.fde_weight, 
        time_weight_type=args.time_weight, 
        cls_weight=args.cls_weight,
        fast_weight=args.fast_weight,
        mr_penalty_weight=args.mr_penalty_weight
    )
    
    # > resume from checkpoint
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        if lane_encoder is not None and 'lane_encoder' in checkpoint:
            lane_encoder.load_state_dict(checkpoint['lane_encoder'])
        if map_fusion is not None and 'map_fusion' in checkpoint:
            map_fusion.load_state_dict(checkpoint['map_fusion'])
    
    params = list(encoder.parameters()) + list(decoder.parameters())
    if lane_encoder is not None:
        params += list(lane_encoder.parameters())
    if map_fusion is not None:
        params += list(map_fusion.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    log_dir = args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    
    best_val_fde = float('inf')
    best_val_info = None
    patience_counter = 0
    last_val_info = None
    
    if not args.eval_only:
        for epoch in range(args.num_epochs):
            encoder.train()
            decoder.train()
            if lane_encoder is not None:
                lane_encoder.train()
            if map_fusion is not None:
                map_fusion.train()
            
            # > reset memory at epoch start if needed
            if needs_edge_ids and initial_memory_backup is not None:
                encoder.memory_bank.reload_memory_bank(initial_memory_backup)
        
            train_losses = []
            train_ades = []
        
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
            for batch_idx, indices in enumerate(pbar):
                indices = indices.numpy()
            
                src = train_data.src_node_ids[indices]
                dst = train_data.dst_node_ids[indices]
                times = train_data.node_interact_times[indices]
                edge_ids = train_data.edge_ids[indices]
                
                # > traj gt
                traj_batch = train_traj_loader.get_batch(src, times)
                valid = traj_batch['valid_mask']
            
                if valid.sum() == 0:
                    continue
            
                # > filter valid samples
                valid_np = valid.numpy()
                src = src[valid_np]
                dst = dst[valid_np]
                times = times[valid_np]
                edge_ids = edge_ids[valid_np]
            
                current = traj_batch['current_state'][valid].to(args.device)
                gt_traj = traj_batch['future_traj'][valid].to(args.device)
                history = traj_batch['history_traj'][valid].to(args.device)
                gt_xy = gt_traj[:, :, :2]
                
            
                # > forward encoder
                if needs_edge_ids:
                    src_embed, _ = encoder.compute_src_dst_node_temporal_embeddings(
                        src, dst, times, edge_ids,
                        history_traj=history if needs_history else None,
                        current_state=current if needs_history else None
                    )
                else:
                    src_embed, _ = encoder.compute_src_dst_node_temporal_embeddings(
                        src, dst, times,
                        history_traj=history if needs_history else None,
                        current_state=current if needs_history else None
                    )
                if batch_idx == 0 and epoch == 0:
                    print(f"First batch - history shape: {history.shape}")
                    print(f"First batch - current shape: {current.shape}")
                    print(f"First batch - src_embed shape: {src_embed.shape}")
                # > forward map fusion
                if map_fusion is not None and train_lane_loader is not None:
                    abs_current = train_traj_loader.current_state
                    abs_states = np.array([abs_current.get((int(nid), float(ts)), np.zeros(5)) 
                                           for nid, ts in zip(src, times)])
                    lane_data = train_lane_loader.get_batch(
                        src, times, abs_states, device=args.device
                    )
                    
                    if isinstance(lane_data, dict):
                        lane_mask = lane_data['lane_mask']
                        if lane_mask.sum() > 0:
                            lane_embeds = lane_encoder(
                                lane_vectors=lane_data['lane_vectors'],
                                lane_actor_vectors=lane_data['lane_actor_vectors'],
                                is_intersections=lane_data['is_intersections'],
                                turn_directions=lane_data['turn_directions'],
                                traffic_controls=lane_data['traffic_controls'],
                                lane_mask=lane_mask
                            )
                            src_embed = map_fusion(src_embed, lane_embeds, lane_mask)
                    else:
                        lane_points, lane_mask = lane_data
                        if lane_mask is not None and lane_mask.sum() > 0:
                            lane_embeds = lane_encoder(lane_points=lane_points, lane_mask=lane_mask)
                            src_embed = map_fusion(src_embed, lane_embeds, lane_mask)
            
                # > forward decoder
                output = decoder(src_embed, current, history_traj=history)
                    
                pred_traj = output['trajectory']
            
                if args.num_modes > 1:
                    loss_dict = loss_fn(output['trajectories'], gt_xy, output['confidences'], current_state=current)
                else:
                    loss_dict = loss_fn(output['trajectory'].unsqueeze(1), gt_xy, torch.zeros(gt_xy.size(0), 1, device=args.device))
                loss = loss_dict['loss']
            
                # > backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                # > detach memory if needed
                if needs_edge_ids and hasattr(encoder, 'memory_bank'):
                    encoder.memory_bank.node_raw_messages.clear()
                    # detach memory tensors
                    encoder.memory_bank.node_memories.detach_()
            
                train_losses.append(loss.item())
                train_ades.append(loss_dict['ade'].item())
            
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'ADE': f"{loss_dict['ade'].item():.2f}m",
                    'FDE': f"{loss_dict['fde'].item():.2f}m"
                })
        
            avg_loss = np.mean(train_losses)
            avg_ade = np.mean(train_ades)
            print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, ADE={avg_ade:.2f}m")
        
            # > validation
            if (epoch + 1) % args.vis_interval == 0:
                encoder.eval()
                decoder.eval()
                if lane_encoder is not None:
                    lane_encoder.eval()
                if map_fusion is not None:
                    map_fusion.eval()
                
                # > reset memory for val
                if needs_edge_ids and initial_memory_backup is not None:
                    encoder.memory_bank.reload_memory_bank(initial_memory_backup)
            
                val_preds = []
                val_gts = []
                val_states = []
                val_is_target = [] 
            
                with torch.no_grad():
                    for indices in val_loader:
                        indices = indices.numpy()
                    
                        src = val_data.src_node_ids[indices]
                        dst = val_data.dst_node_ids[indices]
                        times = val_data.node_interact_times[indices]
                        edge_ids = val_data.edge_ids[indices]
                    
                        traj_batch = val_traj_loader.get_batch(src, times)
                        valid = traj_batch['valid_mask']
                    
                        if valid.sum() == 0:
                            continue
                    
                        valid_np = valid.numpy()
                        src = src[valid_np]
                        dst = dst[valid_np]
                        times = times[valid_np]
                        edge_ids = edge_ids[valid_np]
                    
                        current = traj_batch['current_state'][valid].to(args.device)
                        gt_traj = traj_batch['future_traj'][valid].to(args.device)
                        history = traj_batch['history_traj'][valid].to(args.device)
                    
                        if needs_edge_ids:
                            src_embed, _ = encoder.compute_src_dst_node_temporal_embeddings(
                                src, dst, times, edge_ids,
                                history_traj=history if needs_history else None,
                                current_state=current if needs_history else None
                            )
                        else:
                            src_embed, _ = encoder.compute_src_dst_node_temporal_embeddings(
                                src, dst, times,
                                history_traj=history if needs_history else None,
                                current_state=current if needs_history else None
                            )
                    
                        # > map fusion
                        if map_fusion is not None and val_lane_loader is not None:
                            abs_current = val_traj_loader.current_state
                            abs_states = np.array([abs_current.get((int(nid), float(ts)), np.zeros(5)) 
                                                   for nid, ts in zip(src, times)])
                            lane_data = val_lane_loader.get_batch(
                                src, times, abs_states, device=args.device
                            )
                            
                            if isinstance(lane_data, dict):
                                lane_mask = lane_data['lane_mask']
                                if lane_mask.sum() > 0:
                                    lane_embeds = lane_encoder(
                                        lane_vectors=lane_data['lane_vectors'],
                                        lane_actor_vectors=lane_data['lane_actor_vectors'],
                                        is_intersections=lane_data['is_intersections'],
                                        turn_directions=lane_data['turn_directions'],
                                        traffic_controls=lane_data['traffic_controls'],
                                        lane_mask=lane_mask
                                    )
                                    src_embed = map_fusion(src_embed, lane_embeds, lane_mask)
                            else:
                                lane_points, lane_mask = lane_data
                                if lane_mask is not None and lane_mask.sum() > 0:
                                    lane_embeds = lane_encoder(lane_points=lane_points, lane_mask=lane_mask)
                                    src_embed = map_fusion(src_embed, lane_embeds, lane_mask)
                    
                        output = decoder(src_embed, current, history_traj=history)
                    
                        if args.num_modes > 1:
                            # > pick the best by FDE 
                            trajs = output['trajectories']
                            gt = gt_traj[:, :, :2].unsqueeze(1)
                            l2_norm = torch.norm(trajs - gt, dim=-1) 
                            fde_per_mode = l2_norm[:, :, -1] 
                            best_mode = fde_per_mode.argmin(dim=1) 
                            best_traj = trajs[torch.arange(trajs.size(0), device=trajs.device), best_mode]
                            val_preds.append(best_traj)
                        else:
                            val_preds.append(output['trajectory'])
                            
                        val_gts.append(gt_traj[:, :, :2])
                        val_states.append(current)
                        
                        val_is_target.append(traj_batch['is_target'][valid].to(args.device))
            
                if len(val_preds) > 0:
                    preds = torch.cat(val_preds)
                    gts = torch.cat(val_gts)
                    states = torch.cat(val_states)
                    is_target_val = torch.cat(val_is_target)
                
                    val_ade_all = TrajectoryMetrics.compute_ade(preds, gts)
                    val_fde_all = TrajectoryMetrics.compute_fde(preds, gts)
                    
                    # > TARGET_AGENT only metrics (for early stopping)
                    target_mask = is_target_val.bool()
                    target_count = target_mask.sum().item()
                    if target_count > 0:
                        val_ade = TrajectoryMetrics.compute_ade(preds[target_mask], gts[target_mask])
                        val_fde = TrajectoryMetrics.compute_fde(preds[target_mask], gts[target_mask])
                        miss_rate = TrajectoryMetrics.compute_miss_rate(preds[target_mask], gts[target_mask])
                    else:
                        val_ade = val_ade_all
                        val_fde = val_fde_all
                        
                        miss_rate = TrajectoryMetrics.compute_miss_rate(preds, gts)
                        
                    dir_acc = TrajectoryMetrics.compute_direction_accuracy(preds, gts)
                
                    rating, expl = TrajectoryMetrics.get_quality_rating(val_ade, val_fde)
                
                    print(f"\nValidation:")
                    print(f"   All: ADE={val_ade_all:.3f}m, FDE={val_fde_all:.3f}m")
                    print(f"   TARGET_AGENT (n={target_count}): ADE={val_ade:.3f}m, FDE={val_fde:.3f}m, MR={miss_rate*100:.1f}%")
                    print(f"   Quality: {rating} - {expl}\n")
                
                
                    if args.visualize:
                        visualize_predictions(
                            preds, gts, states,
                            save_path=f"{log_dir}/epoch{epoch+1}_traj.png"
                        )
                    # > document las val
                    last_val_info = {
                        'epoch': epoch + 1,
                        'ade': val_ade,
                        'fde': val_fde,
                        'mr_2m': miss_rate,
                        'direction_acc': dir_acc,
                    }
                    # > early stopping
                    if val_fde < best_val_fde:
                        best_val_fde = val_fde
                        best_val_info = {
                            'epoch': epoch + 1,
                            'ade': val_ade,
                            'fde': val_fde,
                            'mr_2m': miss_rate,
                            'direction_acc': dir_acc,
                        }
                        patience_counter = 0
                        save_dict = {
                            'encoder': encoder.state_dict(),
                            'decoder': decoder.state_dict()
                        }
                        if lane_encoder is not None:
                            save_dict['lane_encoder'] = lane_encoder.state_dict()
                        if map_fusion is not None:
                            save_dict['map_fusion'] = map_fusion.state_dict()
                        torch.save(save_dict, f"{log_dir}/best_model.pt")
                        print(f"New best FDE: {best_val_fde:.3f}m")
                    else:
                        patience_counter += 1
                        if patience_counter >= args.patience:
                            print("Early stopping")
                            break
    
    
    if 'test' in split_loader.splits and test_data is not None:
        print("Final Evaluation on TEST SET")
        
        if args.eval_only and args.resume:
            checkpoint = torch.load(args.resume)
        else:
            checkpoint = torch.load(f"{log_dir}/best_model.pt")
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        if lane_encoder is not None and 'lane_encoder' in checkpoint:
            lane_encoder.load_state_dict(checkpoint['lane_encoder'])
        if map_fusion is not None and 'map_fusion' in checkpoint:
            map_fusion.load_state_dict(checkpoint['map_fusion'])
        
        encoder.eval()
        decoder.eval()
        if lane_encoder is not None:
            lane_encoder.eval()
        if map_fusion is not None:
            map_fusion.eval()
        
        # > reset memory for testing
        if needs_edge_ids and initial_memory_backup is not None:
            encoder.memory_bank.reload_memory_bank(initial_memory_backup)
        
        processed_keys = set() 
        
        all_ade_per_mode = []
        all_fde_per_mode = []
        all_preds = []
        all_gts = []
        all_states = []
        all_is_target = []  
        
        with torch.no_grad():
            for indices in tqdm(test_loader, desc="Test"):
                indices = indices.numpy()
                
                src = test_data.src_node_ids[indices]
                dst = test_data.dst_node_ids[indices]
                times = test_data.node_interact_times[indices]
                edge_ids = test_data.edge_ids[indices]
                
                # > Handle Mamba (needs history) vs others
                nodes_to_process = [('src', src), ('dst', dst)]
                precomputed_embeds = {}
                
                if not needs_history:
                    if needs_edge_ids:
                        src_embed, dst_embed = encoder.compute_src_dst_node_temporal_embeddings(
                            src, dst, times, edge_ids
                        )
                    else:
                        src_embed, dst_embed = encoder.compute_src_dst_node_temporal_embeddings(
                            src, dst, times
                        )
                    precomputed_embeds['src'] = src_embed
                    precomputed_embeds['dst'] = dst_embed
                
                for node_type, node_ids in nodes_to_process:
                    
                    batch_node_ids = node_ids  
                    batch_times = times
            
                    # 1. Deduplication
                    keep_mask = []
                    for nid, ts in zip(batch_node_ids, batch_times):
                        key = (int(nid), float(ts))
                        if key not in processed_keys:
                            processed_keys.add(key)
                            keep_mask.append(True)
                        else:
                            keep_mask.append(False)
            
                    keep_mask = torch.tensor(keep_mask, dtype=torch.bool)
                    if keep_mask.sum() == 0:
                        continue
            
                    batch_node_ids = batch_node_ids[keep_mask.numpy()]
                    batch_times = batch_times[keep_mask.numpy()]
            
                    if not needs_history:
                        batch_embeds = precomputed_embeds[node_type][keep_mask]
            
                    # 2. Load trajectories
                    traj_batch = test_traj_loader.get_batch(batch_node_ids, batch_times)
                    valid = traj_batch['valid_mask']
                
                    if valid.sum() == 0:
                        continue
                
                    valid_np = valid.numpy()
                    batch_node_ids = batch_node_ids[valid_np]
                    batch_times = batch_times[valid_np] 
                    
                    if not needs_history:
                        batch_embeds = batch_embeds[valid]
                
                    current = traj_batch['current_state'][valid].to(args.device)
                    gt_traj = traj_batch['future_traj'][valid].to(args.device)
                    history = traj_batch['history_traj'][valid].to(args.device)
                    is_target = traj_batch['is_target'][valid].to(args.device) 
                
                    # 3. Compute embeddings for Mamba (needs history)
                    if needs_history:
                        batch_embeds, _ = encoder.compute_src_dst_node_temporal_embeddings(
                            batch_node_ids, batch_node_ids, batch_times,
                            history_traj=history,
                            current_state=current
                        )

                    # > map fusion
                    if map_fusion is not None and test_lane_loader is not None:
                        abs_current = test_traj_loader.current_state
                        abs_states = np.array([abs_current.get((int(nid), float(ts)), np.zeros(5))
                                               for nid, ts in zip(batch_node_ids, batch_times)])
                        lane_data = test_lane_loader.get_batch(
                            batch_node_ids, batch_times, abs_states, device=args.device
                        )
                        
                        if isinstance(lane_data, dict):
                            lane_mask = lane_data['lane_mask']
                            if lane_mask.sum() > 0:
                                lane_embeds = lane_encoder(
                                    lane_vectors=lane_data['lane_vectors'],
                                    lane_actor_vectors=lane_data['lane_actor_vectors'],
                                    is_intersections=lane_data['is_intersections'],
                                    turn_directions=lane_data['turn_directions'],
                                    traffic_controls=lane_data['traffic_controls'],
                                    lane_mask=lane_mask
                                )
                                batch_embeds = map_fusion(batch_embeds, lane_embeds, lane_mask)
                        else:
                            lane_points, lane_mask = lane_data
                            if lane_mask is not None and lane_mask.sum() > 0:
                                lane_embeds = lane_encoder(lane_points=lane_points, lane_mask=lane_mask)
                                batch_embeds = map_fusion(batch_embeds, lane_embeds, lane_mask)

                
                    output = decoder(batch_embeds, current, history_traj=history)
                
                    if args.num_modes > 1:
                        trajs = output['trajectories']  
                        gt = gt_traj[:, :, :2].unsqueeze(1) 
                        l2_norm = torch.norm(trajs - gt, dim=-1)  
    
                        ade_per_mode = l2_norm.mean(dim=-1)  
                        fde_per_mode = l2_norm[:, :, -1] 
    
                        all_ade_per_mode.append(ade_per_mode)
                        all_fde_per_mode.append(fde_per_mode)
                    
                        best_mode = fde_per_mode.argmin(dim=1)
                        best_traj = trajs[torch.arange(trajs.size(0), device=trajs.device), best_mode]
                        all_preds.append(best_traj)  
                    else:
                        all_ade_per_mode.append(torch.norm(output['trajectory'] - gt_traj[:, :, :2], dim=-1).mean(dim=-1, keepdim=True))
                        all_fde_per_mode.append(torch.norm(output['trajectory'][:, -1] - gt_traj[:, -1, :2], dim=-1, keepdim=True))
                        all_preds.append(output['trajectory'])
                    
                    all_gts.append(gt_traj[:, :, :2])
                    all_states.append(current)
                    all_is_target.append(is_target) 
        
        if len(all_ade_per_mode) > 0:
            all_ade = torch.cat(all_ade_per_mode) 
            all_fde = torch.cat(all_fde_per_mode) 
            preds = torch.cat(all_preds)
            gts = torch.cat(all_gts)
            states = torch.cat(all_states)
            is_target_all = torch.cat(all_is_target)
            
            N = all_ade.size(0)
            device = all_ade.device
            
            # > target agent filter
            target_mask = is_target_all.bool()
            target_count = target_mask.sum().item()
            
            print(f"\n[Agent Distribution]")
            print(f"  Total samples: {N}")
            print(f"  TARGET_AGENT: {target_count} ({target_count/N*100:.1f}%)")
            print(f"  Other agents: {N - target_count} ({(N-target_count)/N*100:.1f}%)")
            
            # > v2x-graph protocol: select mode by FDE, compute all metrics on that trajectory
            fde_best_mode = all_fde.argmin(dim=1) 
            v2x_ade = all_ade[torch.arange(N, device=device), fde_best_mode]
            v2x_fde = all_fde[torch.arange(N, device=device), fde_best_mode]
            v2x_mr = (v2x_fde > 2).float().mean().item()
            
            speeds = torch.norm(states[:, 2:4], dim=-1)
            
            # > velocity stratification
            slow_mask = speeds < 5
            med_mask = (speeds >= 5) & (speeds < 15)
            fast_mask = speeds >= 15
            
            test_dir_acc = TrajectoryMetrics.compute_direction_accuracy(preds, gts)
            
            # > v2x protocol metrics
            v2x_ade_mean = v2x_ade.mean().item()
            v2x_ade_std = v2x_ade.std().item()
            v2x_fde_mean = v2x_fde.mean().item()
            v2x_fde_std = v2x_fde.std().item()
            
            # > v2x protocol metrics with speed-stratified metrics  (only for reference)
            v2x_ade_slow = v2x_ade[slow_mask].mean().item() if slow_mask.sum() > 0 else None
            v2x_ade_med = v2x_ade[med_mask].mean().item() if med_mask.sum() > 0 else None
            v2x_ade_fast = v2x_ade[fast_mask].mean().item() if fast_mask.sum() > 0 else None
            
            v2x_fde_slow = v2x_fde[slow_mask].mean().item() if slow_mask.sum() > 0 else None
            v2x_fde_med = v2x_fde[med_mask].mean().item() if med_mask.sum() > 0 else None
            v2x_fde_fast = v2x_fde[fast_mask].mean().item() if fast_mask.sum() > 0 else None
            
            v2x_mr_slow = (v2x_fde[slow_mask] > 2).float().mean().item() if slow_mask.sum() > 0 else None
            v2x_mr_med = (v2x_fde[med_mask] > 2).float().mean().item() if med_mask.sum() > 0 else None
            v2x_mr_fast = (v2x_fde[fast_mask] > 2).float().mean().item() if fast_mask.sum() > 0 else None
            
            # > TARGET_AGENT only metrics 
            target_ade = None
            target_fde = None
            target_mr = None
            if target_count > 0:
                target_ade = v2x_ade[target_mask].mean().item()
                target_fde = v2x_fde[target_mask].mean().item()
                target_mr = (v2x_fde[target_mask] > 2).float().mean().item()
            
            print("TEST SET RESULTS")
            

            print(f"\n[All Agents under V2X-Graph Protocol] (for reference only)")
            print(f"  ADE:    {v2x_ade_mean:.3f} ± {v2x_ade_std:.3f} m")
            print(f"  FDE:    {v2x_fde_mean:.3f} ± {v2x_fde_std:.3f} m")
            print(f"  MR@2m:  {v2x_mr*100:.1f}%")
            
            # > TARGET_AGENT only results
            if target_count > 0:
                print(f"\n[TARGET_AGENT Only under V2X-Graph Protocol] (n={target_count}) ")
                print(f"comparable to v2x-graph")
                print(f"  ADE: {target_ade:.3f} m")
                print(f"  FDE: {target_fde:.3f} m")
                print(f"  MR@2m: {target_mr*100:.1f}%")

            
            print(f"\n  DirectionAcc: {test_dir_acc*100:.1f}%")
            

            
            print(f"\n[By Speed under V2X-Graph Protocol] (for reference only)")
            print(f"  {'Speed':<18} {'ADE':>8} {'FDE':>8} {'MR@2m':>8} {'Count':>8}")

            if v2x_ade_slow is not None:
                print(f"  {'Slow (<5m/s)':<18} {v2x_ade_slow:>8.3f} {v2x_fde_slow:>8.3f} {v2x_mr_slow*100:>7.1f}% {slow_mask.sum().item():>8}")
            if v2x_ade_med is not None:
                print(f"  {'Medium (5-15m/s)':<18} {v2x_ade_med:>8.3f} {v2x_fde_med:>8.3f} {v2x_mr_med*100:>7.1f}% {med_mask.sum().item():>8}")
            if v2x_ade_fast is not None:
                print(f"  {'Fast (>15m/s)':<18} {v2x_ade_fast:>8.3f} {v2x_fde_fast:>8.3f} {v2x_mr_fast*100:>7.1f}% {fast_mask.sum().item():>8}")
            
            
            # > visualization
            visualize_predictions(preds, gts, states, num_samples=10, save_path=f"{log_dir}/final_predictions.png")
            analyze_errors(preds, gts, states, save_path=f"{log_dir}/error_analysis.png")
            
            
            results = {
                'config': {
                    'num_modes': args.num_modes,
                    'pred_horizon': args.pred_horizon,
                    'fde_weight': args.fde_weight,
                    'cls_weight': args.cls_weight,
                    'time_weight': args.time_weight,
                    'use_intermodal_attn': args.use_intermodal_attn,
                    'use_map': args.use_map,
                    'target_only': args.target_only,
                },
                'val': {
                    'best': best_val_info,
                    'last': last_val_info, 
                },
                

                'test_v2x_protocol': {
                    'description': 'all agents under v2X-graph protocol (for reference only)',
                    'ADE_mean': v2x_ade_mean,
                    'ADE_std': v2x_ade_std,
                    'FDE_mean': v2x_fde_mean,
                    'FDE_std': v2x_fde_std,
                    'MR_2m': v2x_mr,
                    'direction_acc': test_dir_acc,
                    'n_samples': len(gts),
                    'by_speed': {
                        'slow': {'ade': v2x_ade_slow, 'fde': v2x_fde_slow, 'mr_2m': v2x_mr_slow, 'n': int(slow_mask.sum().item())},
                        'medium': {'ade': v2x_ade_med, 'fde': v2x_fde_med, 'mr_2m': v2x_mr_med, 'n': int(med_mask.sum().item())},
                        'fast': {'ade': v2x_ade_fast, 'fde': v2x_fde_fast, 'mr_2m': v2x_mr_fast, 'n': int(fast_mask.sum().item())},
                    },
                },
                
                'test_target_agent_only': {
                    'description': 'TARGET_AGENT only eval under v2x-graph protocol (comparable with v2x-graph)',
                    'ADE': target_ade,
                    'FDE': target_fde,
                    'MR_2m': target_mr,
                    'n_samples': target_count,
                },
            }
            
            with open(f"{log_dir}/results.json", 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {log_dir}/results.json")
    
    else:
        print("\nNo test data available, skipping test evaluation")
        print(f"Best validation FDE: {best_val_fde:.3f}m")
        
        results = {
            'config': {
                'num_modes': args.num_modes,
                'use_map': args.use_map,
            },
            'val': {
                    'best': best_val_info,
                    'last': last_val_info, 
                },
        }
        with open(f"{log_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\nDone. Results saved to {log_dir}/")


if __name__ == "__main__":
    main()
