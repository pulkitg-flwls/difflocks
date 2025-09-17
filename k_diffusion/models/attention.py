from functools import reduce
from inspect import isfunction
import math
from k_diffusion.models.modules import AxialRoPE, apply_rotary_emb_
from k_diffusion.models.modules import AdaRMSNorm, FeedForwardBlock, LinearGEGLU, RMSNorm, apply_wd, use_flash_2, zero_init
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

import flash_attn



try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

    
def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)

def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)

def scale_for_cosine_sim_single(q, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    return q * scale_q.to(q.dtype) 

class SpatialTransformerSimpleV2(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head,
                 global_cond_dim,
                 do_self_attention=True,
                 dropout=0.,
                 context_dim=None,
                 ):
        super().__init__()
       

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.n_heads = n_heads
        self.d_head = d_head
        self.do_self_attention=do_self_attention


        self.x_in_norm = AdaRMSNorm(in_channels, global_cond_dim)

        #x to qkv
        if self.do_self_attention:
            self.x_qkv_proj = apply_wd(torch.nn.Linear(in_channels, inner_dim * 3, bias=False))
        else:
            self.x_q_proj = apply_wd(torch.nn.Linear(in_channels, inner_dim, bias=False))
        self.x_scale = nn.Parameter(torch.full([self.n_heads], 10.0))

        self.x_pos_emb = AxialRoPE(d_head // 2, self.n_heads)


        #context to kv
        self.cond_kv_proj = apply_wd(torch.nn.Linear(context_dim, inner_dim * 2, bias=False))
        self.cond_scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.cond_pos_emb = AxialRoPE(d_head // 2, self.n_heads)

        self.ff = FeedForwardBlock(in_channels, d_ff=int(in_channels*2), cond_features=global_cond_dim, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.proj_out = apply_wd(zero_module(nn.Linear(in_channels, inner_dim)))
    

    def forward(self, x, pos, global_cond, context=None, context_pos=None):
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(x, 'b c h w -> b h w c')
        context = rearrange(context, 'b c h w -> b h w c')
        x = self.x_in_norm(x, global_cond)

        if self.do_self_attention:
            #x to qkv
            x_qkv = self.x_qkv_proj(x)
            pos = rearrange(pos, "... h w e -> ... (h w) e").to(x_qkv.dtype)
            x_theta = self.x_pos_emb(pos)
            if use_flash_2(x_qkv):
                x_qkv = rearrange(x_qkv, "n h w (t nh e) -> n (h w) t nh e", t=3, e=self.d_head)
                x_qkv = scale_for_cosine_sim_qkv(x_qkv, self.x_scale, 1e-6)
                x_theta = torch.stack((x_theta, x_theta, torch.zeros_like(x_theta)), dim=-3)
                x_qkv = apply_rotary_emb_(x_qkv, x_theta)
                x_q, x_k, x_v = x_qkv.chunk(3,dim=-3)
            else:
                print("we couldnt run flash2, maybe it's not installed or the input si not bfloat16")
                exit(1)
        else:
            #x to q
            x_q = self.x_q_proj(x)
            pos = rearrange(pos, "... h w e -> ... (h w) e").to(x_q.dtype)
            x_theta = self.x_pos_emb(pos)
            if use_flash_2(x_q):
                x_q = rearrange(x_q, "n h w (nh e) -> n (h w) nh e", e=self.d_head)
                x_q = scale_for_cosine_sim_single(x_q, self.x_scale[:, None], 1e-6)
                x_q=x_q.unsqueeze(2) #n (h w) 1 nh e
                x_theta=x_theta.unsqueeze(1)
                x_q = apply_rotary_emb_(x_q, x_theta)
            else:
                print("we couldnt run flash2, maybe it's not installed or the input si not bfloat16")
                exit(1)


        #context to kv
        cond_kv = self.cond_kv_proj(context)
        # print("cond_kv init",cond_kv.shape)
        context_pos = rearrange(context_pos, "... h w e -> ... (h w) e").to(cond_kv.dtype)
        cond_theta = self.cond_pos_emb(context_pos)
        if use_flash_2(cond_kv):
            cond_kv = rearrange(cond_kv, "n h w (t nh e) -> n (h w) t nh e", t=2, e=self.d_head)
            cond_k, cond_v = cond_kv.unbind(2) # makes each n (h w) nh e
            cond_k = scale_for_cosine_sim_single(cond_k, self.cond_scale[:, None], 1e-6)
            cond_k=cond_k.unsqueeze(2) #n (h w) 1 nh e
            cond_theta=cond_theta.unsqueeze(1)
            cond_k = apply_rotary_emb_(cond_k, cond_theta)
            cond_k=cond_k.squeeze(2)
        else:
            print("we couldnt run flash2, maybe it's not installed or the input si not bfloat16")
            exit(1)

        #doing self attention by concating K and V between X and cond
        if self.do_self_attention:
            k = torch.cat([x_k, cond_k.unsqueeze(2)], dim=1)
            v = torch.cat([x_v, cond_v.unsqueeze(2)], dim=1)
        else:
            # print("not doing self attention")
            k=cond_k.unsqueeze(2)
            v=cond_v.unsqueeze(2)
        q=x_q
        

        #rearange a bit
        q=q.squeeze(2)
        kv=torch.cat([k,v],2)
        # print("final q before giving to flash",q.shape)
        # print("final kv before giving to flash",kv.shape)

        x = flash_attn.flash_attn_kvpacked_func(q, kv, softmax_scale=1.0)

        x = rearrange(x, 'b (h w) nh e -> b (h w) (nh e)', nh=self.n_heads, e=self.d_head, h=h, w=w)

        #last ff
        x = self.dropout(x)
        x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, c=c)

        x=  x + x_in 

        #attention part finished------

        #linear feed forward
        x = rearrange(x, 'b c h w -> b h w c', h=h, w=w, c=c)

        # print("x before ff is ", x.shape)
        x = self.ff(x, global_cond)


        x = rearrange(x, 'b h w c -> b c h w', h=h, w=w, c=c)

        return x
            

