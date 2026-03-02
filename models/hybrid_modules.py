import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from einops.layers.torch import Reduce

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class block(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(2, 2, 1)
        self.conv_proj = nn.Conv2d(dim // 2, dim, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        
        self.dim = dim

    def forward(self, x):
        batch_size = x.size(0)
        dim = x.size(1)
        attn1 = self.conv0(x)  
        attn2 = self.conv_spatial(attn1)  

        attn1 = self.conv1(attn1) 
        attn2 = self.conv2(attn2) 

        attn = torch.cat([attn1, attn2], dim=1)  
        avg_attn = torch.mean(attn, dim=1, keepdim=True) 
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) 
        agg = torch.cat([avg_attn, max_attn], dim=1) 
        agg = self.conv(agg).sigmoid()

        ch_attn1 = self.global_pool(attn) 
        
        if ch_attn1.size(0) > 1 or ch_attn1.size(2) > 1 or ch_attn1.size(3) > 1:
            z = self.fc1(ch_attn1)
        else:
            z = self.fc1[0](ch_attn1)  
            z = self.fc1[2](z)  
        
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)
        a_b = self.softmax(a_b)

        a1,a2 =  a_b.chunk(2, dim=1)
        a1 = a1.reshape(batch_size,dim // 2,1,1)
        a2 = a2.reshape(batch_size, dim // 2, 1, 1)

        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
        w2 = a2 * agg[:, 1, :, :].unsqueeze(1)

        attn = attn1 * w1 + attn2 * w2
        attn = self.conv_proj(attn).sigmoid()

        return x * attn

class Attention_KSB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = block(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)    
        x = self.activation(x) 
        x = self.spatial_gating_unit(x) 
        x = self.proj_2(x)   
        x = x + shorcut
        return x

class Mlp_KSB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.in_features = in_features

    def forward(self, x):
        x = self.fc1(x)     
        if x.size(0) > 1 or x.size(2) > 1 or x.size(3) > 1:
            x = self.dwconv(x)  
        x = self.act(x)     
        x = self.drop(x)
        x = self.fc2(x)    
        x = self.drop(x)
        return x

class Transformer_KSFA(nn.Module):
    def __init__(self, dim, mlp_ratio=1., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention_KSB(dim)
        self.drop_path = nn.Identity()  
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_KSB(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        
        self.dim = dim

    def forward(self, x):
        if x.size(0) > 1 or x.size(2) > 1 or x.size(3) > 1:
            norm1_x = self.norm1(x)
            norm2_x = self.norm2(x)
        else:
            norm1_x = x
            norm2_x = x
        
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(norm1_x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(norm2_x))
        return x

class FFN(nn.Module):
    def __init__(self, dim, bias, kernel_size):
        super(FFN, self).__init__()
        if kernel_size not in [3, 5, 7]:
            raise ValueError("Invalid kernel_size. Must be 3, 5, or 7.")

        self.kernel_size = kernel_size
        hidden_features = 64

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias) 
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3,
                                   groups=hidden_features, bias=bias)
        self.relu3 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        if self.kernel_size == 3:
            x = self.relu3(self.dwconv3x3(x))
        elif self.kernel_size == 5:
            x = self.relu3(self.dwconv5x5(x))
        elif self.kernel_size == 7:
            x = self.relu3(self.dwconv7x7(x))
        x = self.project_out(x)

        return x

class Token_Selective_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, k, group_num, chunk_size=8):
        super(Token_Selective_Attention, self).__init__()
        self.num_heads = num_heads
        self.k = k
        self.group_num = group_num
        self.dim_group = dim // group_num
        self.chunk_size = chunk_size  
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv3d(self.group_num, self.group_num * 3, kernel_size=(1, 1, 1), bias=False)
        self.qkv_conv = nn.Conv3d(self.group_num * 3, self.group_num * 3, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                  groups=self.group_num * 3, bias=bias)  
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, self.group_num, c//self.group_num, h, w)
        b, t, c, h, w = x.shape  

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = rearrange(q, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        k = rearrange(k, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        v = rearrange(v, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, _, N = q.shape  

        if N <= self.chunk_size or not self.training:  
            mask = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)
            attn = (q.transpose(-2, -1) @ k) * self.temperature  
            index = torch.topk(attn, k=int(N * self.k), dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  
        else:
            out = torch.zeros_like(v)  
            mask = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)
            
            for i in range(0, N, self.chunk_size):
                end_idx = min(i + self.chunk_size, N)
                q_chunk = q[:, :, :, i:end_idx]  
                
                attn_chunk = (q_chunk.transpose(-2, -1) @ k) * self.temperature  
                
                index_chunk = torch.topk(attn_chunk, k=int(N * self.k), dim=-1, largest=True)[1]
                mask_chunk = torch.zeros(b, self.num_heads, end_idx - i, N, device=x.device, requires_grad=False)
                mask_chunk.scatter_(-1, index_chunk, 1.)
                mask[:, :, i:end_idx, :] = mask_chunk
                
                attn_chunk = torch.where(mask_chunk > 0, attn_chunk, torch.full_like(attn_chunk, float('-inf')))
                attn_chunk = attn_chunk.softmax(dim=-1)
                
                out_chunk = (attn_chunk @ v.transpose(-2, -1)).transpose(-2, -1)  
                out[:, :, :, i:end_idx] = out_chunk

        out = rearrange(out, 'b head c (h w t) -> b t (head c) h w', head=self.num_heads, h=h, w=w)

        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)

        return out


class Transformer_DCSA(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, kernel_size, k, group_num, chunk_size=32):
        super(Transformer_DCSA, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_selective = Token_Selective_Attention(dim, num_heads, bias, k, group_num, chunk_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn_conv = FFN(dim, bias, kernel_size)

    def forward(self, x):
        x = x + self.attn_selective(self.norm1(x))  

        x = x + self.ffn_conv(self.norm2(x))
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, reduction=16, min_hidden=32, kernel_size=3):
        super().__init__()
                                        
        hidden_dim = max(dim // reduction, min_hidden)
        padding = kernel_size // 2
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=1, bias=False)
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=1, bias=False)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, bias=True),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x3, x5):
                                  
        f3 = self.conv3(x3)
        f5 = self.conv5(x5)
        s = self.pool(f3 + f5)
        w = self.softmax(self.mlp(s))
        w3 = w[:, 0:1]
        w5 = w[:, 1:2]
        return f3 * w3 + f5 * w5


class LeFF(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
                                                 
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.project_in = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
                        
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.project_out(x)
        x = self.dropout(x)
        return x


class CrossAttention2D(nn.Module):
    def __init__(self, dim, num_heads=4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
                                            
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

    def forward(self, q_feat, kv_feat):
                                     
        b, c, h, w = q_feat.shape
        q = self.q(q_feat)
        k = self.k(kv_feat)
        v = self.v(kv_feat)

        q = rearrange(q, 'b (head d) h w -> b head (h w) d', head=self.num_heads)
        k = rearrange(k, 'b (head d) h w -> b head (h w) d', head=self.num_heads)
        v = rearrange(v, 'b (head d) h w -> b head (h w) d', head=self.num_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b head (h w) d -> b (head d) h w', h=h, w=w)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class SelfAttention2D(nn.Module):
    def __init__(self, dim, num_heads=4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
                               
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

    def forward(self, x):
                           
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)

        q = rearrange(q, 'b (head d) h w -> b head (h w) d', head=self.num_heads)
        k = rearrange(k, 'b (head d) h w -> b head (h w) d', head=self.num_heads)
        v = rearrange(v, 'b (head d) h w -> b head (h w) d', head=self.num_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b head (h w) d -> b (head d) h w', h=h, w=w)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class DCSA(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0, layer_norm_type='WithBias'):
        super().__init__()
                                             
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.kv_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=False)

                         
        self.norm_q1 = LayerNorm(dim, layer_norm_type)
        self.norm_kv1 = LayerNorm(dim, layer_norm_type)
        self.self_attn_q = SelfAttention2D(dim, num_heads=num_heads, attn_dropout=dropout, proj_dropout=dropout)
        self.self_attn_kv = SelfAttention2D(dim, num_heads=num_heads, attn_dropout=dropout, proj_dropout=dropout)
        self.norm_q2 = LayerNorm(dim, layer_norm_type)
        self.norm_kv2 = LayerNorm(dim, layer_norm_type)
   
        self.cross_attn = CrossAttention2D(dim, num_heads=num_heads, attn_dropout=dropout, proj_dropout=dropout)
        self.cross_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm_ffn = LayerNorm(dim, layer_norm_type)
        self.leff = LeFF(dim, dropout=dropout)

    def forward(self, aop_feat, dop_feat):
                              
        q_branch = self.q_conv(aop_feat)
        q_branch = q_branch + self.self_attn_q(self.norm_q1(q_branch))
        q = self.norm_q2(q_branch)
                  
        kv_branch = self.kv_conv(dop_feat)
        kv_branch = kv_branch + self.self_attn_kv(self.norm_kv1(kv_branch))
        kv = self.norm_kv2(kv_branch)
                                
        cross = self.cross_attn(q, kv)
        x = q_branch + self.cross_dropout(cross)
        x = x + self.leff(self.norm_ffn(x))
        return x


class SGFT(nn.Module):
    def __init__(self, channels, dop_kernel_size=1, spatial_kernel_size=7):
        super().__init__()
                                       
        if spatial_kernel_size not in (3, 5, 7):
            raise ValueError("spatial_kernel_size must be one of {3,5,7}")
        spatial_padding = spatial_kernel_size // 2
        self.spatial_attn = nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=spatial_padding, bias=False)

        if dop_kernel_size != 1:
            raise ValueError("dop_kernel_size currently only supports 1 for lightweight gating.")
        self.dop_mapper = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.channels = channels

    def forward(self, x, raw_dop_1ch):
                                            
        if raw_dop_1ch.dim() != 4 or raw_dop_1ch.size(1) != 1:
            raise ValueError(f"raw_dop_1ch must be [B,1,H,W], got {tuple(raw_dop_1ch.shape)}")
              
        dop_resized = F.interpolate(raw_dop_1ch, size=x.shape[-2:], mode='bilinear', align_corners=False)
        soft_mask = self.sigmoid(self.dop_mapper(dop_resized))

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        original_attention = self.sigmoid(self.spatial_attn(spatial))

        final_mask = original_attention * soft_mask
        return x * final_mask
