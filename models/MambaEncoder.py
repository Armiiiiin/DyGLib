import torch
import torch.nn as nn
from typing import Optional
from models.mamba_layers import VMRNNCell  

def init_weights(m):
    """Initialize network weights using Xavier uniform initialization."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


class MLPEmbedding(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel)
        )
    
    def forward(self, x):
        return self.mlp(x)


class TemporalData:
    def __init__(self):
        pass


class MotionNodeEncoder(nn.Module):
    """Two-layer MLP with LayerNorm and ReLU activation."""
    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(MotionNodeEncoder, self).__init__()
        
        self.historical_steps = historical_steps
        self.center_embed = MLPEmbedding(in_channel=node_dim, out_channel=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
 
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))

        nn.init.normal_(self.bos_token, mean=0., std=.02)
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)

        self.vm_rnn = VMRNNCell(
            hidden_dim=embed_dim,
            input_resolution=(1, self.historical_steps + 1),
            depth=num_layers,
            drop=dropout,
            attn_drop=dropout,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            d_state=16
        )
        
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                bos_mask: torch.Tensor,
                padding_mask: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
       

        if rotate_mat is not None:
            B, T, _ = x.shape
            pos = x[:, :, :2] 
            vel = x[:, :, 2:4] 
            
            pos_flat = pos.reshape(B * T, 2, 1)  
            vel_flat = vel.reshape(B * T, 2, 1) 

            rotate_mat_expanded = rotate_mat.unsqueeze(1).repeat(1, T, 1, 1)  
            rotate_mat_flat = rotate_mat_expanded.reshape(B * T, 2, 2)  

            pos_rotated_flat = torch.bmm(rotate_mat_flat, pos_flat)  
            vel_rotated_flat = torch.bmm(rotate_mat_flat, vel_flat)  
            
            pos_rotated = pos_rotated_flat.reshape(B, T, 2)
            vel_rotated = vel_rotated_flat.reshape(B, T, 2)
            
            x_rotated = torch.cat([pos_rotated, vel_rotated], dim=-1)  
            center_embed = self.center_embed(x_rotated)
        else:
            center_embed = self.center_embed(x)
 
        center_embed = torch.where(
            bos_mask.unsqueeze(-1),
            self.bos_token,
            center_embed
        )
 
        center_embed = center_embed + self.mlp(self.norm(center_embed))

        x = torch.where(
            padding_mask.t().unsqueeze(-1),
            self.padding_token,
            center_embed.transpose(0, 1)
        )
        
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0) 
        x = x + self.pos_embed

        x_transposed = x.transpose(0, 1)  
        out, _ = self.vm_rnn(x_transposed, None)  

        return out[:, -1, :]  


class MambaEncoder(nn.Module):
    """Mamba-based temporal encoder for trajectory prediction."""
    
    def __init__(self, hist_horizon=50, node_dim=4, embed_dim=128,
                 num_heads=8, num_layers=4, dropout=0.1, device='cpu'):
        super().__init__()
        
        self.hist_horizon = hist_horizon
        self.embed_dim = embed_dim
        self.device = device

        self.motion_encoder = MotionNodeEncoder(
            historical_steps=hist_horizon,
            node_dim=node_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def compute_src_dst_node_temporal_embeddings(self, src, dst, times,
                                                   history_traj=None,
                                                   current_state=None,
                                                   **kwargs):
        
        batch_size = len(src)
        
        if history_traj is None:
            print("WARNING: No history provided, using zeros")
            history_traj = torch.zeros(batch_size, self.hist_horizon, 4,
                                      device=self.device)

        bos_mask = torch.zeros(batch_size, self.hist_horizon,
                              dtype=torch.bool, device=self.device)
        padding_mask = torch.zeros(batch_size, self.hist_horizon,
                                   dtype=torch.bool, device=self.device)

        if current_state is not None and current_state.shape[1] >= 5:
            headings = current_state[:, 4]
            rotate_mat = torch.zeros(batch_size, 2, 2, device=self.device)
            cos_h = torch.cos(headings)
            sin_h = torch.sin(headings)
            rotate_mat[:, 0, 0] = cos_h
            rotate_mat[:, 0, 1] = -sin_h
            rotate_mat[:, 1, 0] = sin_h
            rotate_mat[:, 1, 1] = cos_h
        else:
            rotate_mat = torch.eye(2, device=self.device).unsqueeze(0).repeat(
                batch_size, 1, 1
            )

        embeddings = self.motion_encoder(
            x=history_traj,
            bos_mask=bos_mask,
            padding_mask=padding_mask,
            rotate_mat=rotate_mat
        )
        
        return embeddings, embeddings  


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    encoder = MambaEncoder(
        hist_horizon=50,
        node_dim=4,
        embed_dim=128,
        num_layers=4,
        device=device
    )
    
    encoder.to(device)
    batch_size = 32
    src = torch.arange(batch_size)
    dst = torch.arange(batch_size)
    times = torch.randn(batch_size)
    history_traj = torch.randn(batch_size, 50, 4, device=device)
    current_state = torch.randn(batch_size, 5, device=device)

    src_embed, dst_embed = encoder.compute_src_dst_node_temporal_embeddings(
        src, dst, times,
        history_traj=history_traj,
        current_state=current_state
    )
    
    print(f"Input history shape: {history_traj.shape}")
    print(f"Output embedding shape: {src_embed.shape}")
    
