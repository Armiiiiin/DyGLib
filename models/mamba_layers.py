
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, Tuple
from functools import partial
from einops import repeat, rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except ImportError:
    print("Warning: mamba_ssm not found. Install with: pip install mamba-ssm")
    selective_scan_fn = None
    selective_scan_ref = None

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except ImportError:
    pass


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, 
                             with_Z=False, with_Group=True, with_complex=False):
    
    import numpy as np
    
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
        return 0
    
    assert not with_complex
    flops = 0
    
    # selective scan operations
    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], 
                                 "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], 
                                 "bdl,bdnl,bdl->bdln")
    
    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    
    return flops


class SS2D(nn.Module):
    """Selective State Space 2D layer with bidirectional scanning."""
    
    def __init__(
        self,
        d_model: int,           
        d_state: int = 16,      
        d_conv: int = 3,        
        expand: int = 2,        
        dt_rank: str = "auto",  
        dt_min: float = 0.001, 
        dt_max: float = 0.1,   
        dt_init: str = "random",  
        dt_scale: float = 1.0,  
        dt_init_floor: float = 1e-4,  
        dropout: float = 0.,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Store configuration
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
    
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, 
                                **factory_kwargs)
        
      
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,  
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()  
       
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), 
                     bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), 
                     bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), 
                     bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), 
                     bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  
        del self.x_proj  
        
      
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, 
                        dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, 
                        dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, 
                        dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, 
                        dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        ) 
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        ) 
        del self.dt_projs
        
    
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, 
                                      copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        
       
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, 
                                  **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        self.forward_core = self.forward_corev0
    
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", 
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * 
            (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        
        return dt_proj
    
    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  
        
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True  
        return A_log
    
    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
       
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def forward_corev0(self, x: torch.Tensor) -> torch.Tensor:
       
        if selective_scan_fn is None:
            raise ImportError("mamba_ssm not installed. Run: pip install mamba-ssm")
        
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W  
        K = 4      
        
       
        x_hwwh = torch.stack([
            x.view(B, -1, L), 
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)  # Vertical
        ], dim=1).view(B, 2, -1, L)
        
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # [B, 4, C, L]
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", 
                            xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, 
                                  [self.dt_rank, self.d_state, self.d_state], 
                                  dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", 
                          dts.view(B, K, -1, L), self.dt_projs_weight)
        
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
       
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        
        assert out_y.dtype == torch.float
        
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(
            out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3
        ).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(
            inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3
        ).contiguous().view(B, -1, L)
        
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)
        
        return y
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  
        
        x = x.permute(0, 3, 1, 2).contiguous()  
        x = self.act(self.conv2d(x))
        
      
        y = self.forward_core(x)
        y = y * F.silu(z)
        
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out




class VSSBlock(nn.Module):
    """Vision State Space Block with residual connection."""
    
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            **kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x




class VSB(VSSBlock):
  
    
    def __init__(
        self,
        hidden_dim: int = 0,
        input_resolution: Tuple[int, int] = (1, 51), 
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        self.input_resolution = input_resolution
        
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input size mismatch: {L} != {H}*{W}"
        
        shortcut = x
        x = self.ln_1(x)
        
        if hx is not None:
            hx = self.ln_1(hx)
            x = torch.cat((x, hx), dim=-1)  
            x = self.linear(x)  
        
        x = x.view(B, H, W, C)
        
        x = self.drop_path(self.self_attention(x))
 
        x = x.view(B, H * W, C)
        x = shortcut + x
        
        return x



class VMRNNCell(nn.Module):
    """Vision State Space Block with temporal state integration."""
    
    def __init__(
        self, 
        hidden_dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        drop: float = 0., 
        attn_drop: float = 0., 
        drop_path: float = 0., 
        norm_layer: Callable = nn.LayerNorm, 
        d_state: int = 16, 
        **kwargs
    ):
        
        super(VMRNNCell, self).__init__()
        
        self.VSBs = nn.ModuleList([
            VSB(
                hidden_dim=hidden_dim,
                input_resolution=input_resolution,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                **kwargs
            )
            for i in range(depth)
        ])
    
    def forward(
        self, 
        xt: torch.Tensor, 
        hidden_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
       
        B, L, C = xt.shape
        
        if hidden_states is None:
            hx = torch.zeros(B, L, C, device=xt.device, dtype=xt.dtype)
            cx = torch.zeros(B, L, C, device=xt.device, dtype=xt.dtype)
        else:
            hx, cx = hidden_states

        outputs = []
        for index, layer in enumerate(self.VSBs):
            if index == 0:
                x = layer(xt, hx)
            else:
                x = layer(outputs[-1], None)
            outputs.append(x)

        o_t = outputs[-1]
        Ft = torch.sigmoid(o_t)
        cell = torch.tanh(o_t)
        Ct = Ft * (cx + cell)
        Ht = Ft * torch.tanh(Ct)
        
        return Ht, (Ht, Ct)

