"""PointCloud Diffusion Network For Segmentation."""

from __future__ import annotations
import torch
import torch.nn as nn
from dps.model.network.geometric import PointTransformerNetwork, to_dense_batch, to_flat_batch, batch2offset, offset2batch, knn, PointBatchNorm, KnnTransformer, KnnTransformerDecoder
import torch.nn.functional as F
import torch_scatter
import math
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from dps.utils.pcd_utils import visualize_point_pyramid


class PositionEmbeddingCoordsSine(nn.Module):
    """From Mask3D"""

    def __init__(self, temperature=10000, normalize=False, scale=None, pos_type="fourier", d_pos=None, d_in=3, gauss_scale=1.0):
        super().__init__()
        self.d_pos = d_pos
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        num_channels = self.d_pos
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert ndim % 2 == 0, f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(bsize, npoints, d_out)
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                out = self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                out = self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

        return out


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Apply mask to attention scores
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            attn = attn.masked_fill(mask, float("-inf"))  # Apply mask where mask is True

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PcdSegNoiseNet(nn.Module):
    """Generate noise for point cloud diffusion process for segmentation."""

    def __init__(self, grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, **kwargs) -> None:
        super().__init__()
        num_denoise_layers = kwargs.get("num_denoise_layers", 3)
        out_dim = kwargs.get("out_dim", 1)
        max_timestep = kwargs.get("max_timestep", 100)
        self.normalize_coord = kwargs.get("normalize_coord", True)
        # Module
        self.pcd_encoder = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=False)
        self.dit_blocks = nn.ModuleList()
        self.latent_dim = hidden_dims[0]
        hidden_dim_denoise = 2 * self.latent_dim
        n_heads_denoise = hidden_dim_denoise // 32
        for i in range(num_denoise_layers):
            self.dit_blocks.insert(0, DiTBlock(hidden_size=hidden_dim_denoise, num_heads=n_heads_denoise))
        self.time_embedding = nn.Embedding(max_timestep, 2 * self.latent_dim)
        self.proj_up = nn.Sequential(nn.Linear(out_dim, 2 * self.latent_dim), nn.SiLU(), nn.Linear(2 * self.latent_dim, self.latent_dim))
        self.proj_down = nn.Sequential(nn.Linear(2 * self.latent_dim, 2 * self.latent_dim), nn.SiLU(), nn.Linear(2 * self.latent_dim, out_dim))
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier", d_pos=self.latent_dim * 2, gauss_scale=1.0, normalize=False)

    def initialize_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.dit_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def encode_pcd(self, points: list[torch.Tensor], attrs=None) -> torch.Tensor:
        """
        Encode point cloud into feat pyramid.
        Args:
            points (list[torch.Tensor]): coord, point, offset
        """
        # Assemble feat
        feat_list = [points[0], points[1]]
        if "normal" in attrs:
            normal = attrs["normal"]
            feat_list.append(normal)
        points = [points[0], torch.cat(feat_list, dim=1), points[2]]

        if self.normalize_coord:
            # Normalize the coord to unit cube
            coord, feat, offset = points
            # Convert to batch & mask
            batch_index = offset2batch(offset)
            coord, mask = to_dense_batch(coord, batch_index)
            normal = to_dense_batch(normal, batch_index) if "normal" in attrs else None
            # Normalize the coord
            center = coord.mean(dim=1, keepdim=True)
            coord = coord - center
            points[0] = to_flat_batch(coord, mask)[0]

        points, all_points, cluster_indices, attrs = self.pcd_encoder(points, return_full=True)

        # DEBUG: Visualize the point cloud pyramid
        # self.visualize_pcd_pyramids(all_points, cluster_indices, attrs)
        # Map to the second dense layer
        return points

    def forward(self, noisy_t: torch.Tensor, points: list[torch.Tensor], t: int) -> torch.Tensor:
        """
        FiLM condition.
        """
        if len(t.shape) == 1:
            t = t[:, None]
        time_token = self.time_embedding(t).squeeze(1)
        # Convert to batch & mask
        coord, feat, offset = points
        batch_index = offset2batch(offset)
        coord, _ = to_dense_batch(coord, batch_index)
        noisy_t = self.proj_up(noisy_t)
        noisy_t, _ = to_dense_batch(noisy_t, batch_index)
        feat, batch_mask = to_dense_batch(feat, batch_index)
        noisy_t = torch.cat((feat, noisy_t), dim=-1)  # Concatenate condition feature
        padding_mask = batch_mask == 0

        # Add positional encoding
        pos_embedding = self.pos_enc(coord).permute(0, 2, 1)
        # Do DiT
        for i in range(len(self.dit_blocks)):
            noisy_t = noisy_t + pos_embedding  # In segmentation task, we add positional encoding before each layer
            # Mask out the padding
            noisy_t = self.dit_blocks[i](noisy_t, time_token, mask=padding_mask.unsqueeze(1))
            # Sanity check
            if torch.isnan(noisy_t).any():
                print("Nan exists in the feature")

        # Decode
        noisy_t = self.proj_down(noisy_t)
        return noisy_t

    ################################# DEBUG #################################
    def visualize_pcd_pyramids(self, all_points, cluster_indices, point_attrs=None):
        coord, feat, _ = all_points[-1]
        normal = point_attrs[-1]["normal"] if "normal" in point_attrs[-1] else None
        offsets = [a[2] for a in all_points[1:]]
        cluster_pyramid = [to_dense_batch(cluster_idex, offset2batch(offset))[0] for (cluster_idex, offset) in zip(cluster_indices, offsets)]
        # Reverse the anchor cluster pyramid
        cluster_pyramid = cluster_pyramid[::-1]
        coord = to_dense_batch(coord, offset2batch(offsets[-1]))[0]
        normal = to_dense_batch(normal, offset2batch(offsets[-1]))[0] if normal is not None else None
        for i in range(coord.shape[0]):
            normal_i = normal[i] if normal is not None else None
            visualize_point_pyramid(coord[i], normal_i, [a[i] for a in cluster_pyramid])
            if i >= 1:
                break
