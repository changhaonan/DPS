"""Register pose transfomer"""

from __future__ import annotations
import math
import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from dps.model.embeddings.pct import PointTransformerEncoderSmall, EncoderMLP, DropoutSampler
from dps.model.network.geometric import PointTransformerNetwork, to_dense_batch, offset2batch, to_flat_batch, DualSoftmaxReposition, PointTransformer, knn
from dps.model.network.genpose_modules import Linear
from dps.utils.pcd_utils import visualize_point_pyramid, visualize_tensor_pcd


class PointSAMBlock(nn.Module):
    """Decoder block like SAM"""

    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.query_attn = PointTransformer(embed_channels=hidden_dim, n_heads=n_heads, attn_drop_rate=dropout, drop_path_rate=dropout)
        self.context_attn = PointTransformer(embed_channels=hidden_dim, n_heads=n_heads, attn_drop_rate=dropout, drop_path_rate=dropout)

    def forward(self, query_points, context_points, knn_indices: dict):
        query_coord, query_feat, query_offset = query_points
        context_coord, context_feat, context_offset = context_points
        # self-attention on query
        query_feat = self.query_attn(feat=query_feat, coord=query_coord, knn_indexes=knn_indices["query2query"])
        # cross-attention on query2context
        query_feat = self.context_attn(query_feat=query_feat, coord=query_coord, context_feat=context_feat, context_coord=context_coord, knn_indexes=knn_indices["query2context"])
        # cross-attention on context2query
        context_feat = self.context_attn(query_feat=context_feat, coord=context_coord, context_feat=query_feat, context_coord=query_coord, knn_indexes=knn_indices["context2query"])
        query_points = [query_coord, query_feat, query_offset]
        context_points = [context_coord, context_feat, context_offset]
        return query_points, context_points


class RigPoseTransformer(nn.Module):
    """Rigister pose transformer network"""

    def __init__(self, grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, fusion_projection_dim, **kwargs) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Encode pcd features
        self.target_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=False)
        self.anchor_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=False)
        #
        self.sam_blocks = nn.ModuleList()
        for i in range(2):
            self.sam_blocks.append(PointSAMBlock(hidden_dim=hidden_dims[0], n_heads=n_heads[0], dropout=0.1))
        # Reposition
        self.dual_softmax_reposition = DualSoftmaxReposition(hidden_dim=hidden_dims[0], temperature=1.0)
        self.normalize_coord = True
        self.query2query_knn = 16
        self.query2context_knn = 64

    def encode_cond(self, target_points, anchor_points, target_attrs=None, anchor_attrs=None):
        """
        Encode target and anchor pcd to features
        """
        # Assemble feat
        target_feat_list = [target_points[0], target_points[1]]
        if "normal" in target_attrs:
            target_normal = target_attrs["normal"]
            target_feat_list.append(target_normal)
        anchor_feat_list = [anchor_points[0], anchor_points[1]]
        if "normal" in anchor_attrs:
            anchor_normal = anchor_attrs["normal"]
            anchor_feat_list.append(anchor_normal)
        target_points = [target_points[0], torch.cat(target_feat_list, dim=1), target_points[2]]
        anchor_points = [anchor_points[0], torch.cat(anchor_feat_list, dim=1), anchor_points[2]]

        if self.normalize_coord:
            # Normalize coord to unit cube
            target_coord, anchor_coord = target_points[0], anchor_points[0]
            target_offset, anchor_offset = target_points[2], anchor_points[2]
            # Convert to batch & mask
            anchor_batch_index = offset2batch(anchor_offset)
            anchor_coord, anchor_mask = to_dense_batch(anchor_coord, anchor_batch_index)
            anchor_normal = to_dense_batch(anchor_normal, anchor_batch_index)[0] if "normal" in anchor_attrs else None
            anchor_center = (anchor_coord.max(dim=1).values + anchor_coord.min(dim=1).values) / 2
            anchor_scale = anchor_coord.max(dim=1).values - anchor_coord.min(dim=1).values
            anchor_scale = torch.max(anchor_scale.max(dim=1).values, torch.tensor(1e-3).to(anchor_scale.device))
            anchor_coord = (anchor_coord - anchor_center[:, None]) / anchor_scale[:, None, None]
            target_batch_index = offset2batch(target_offset)
            target_coord, target_mask = to_dense_batch(target_coord, target_batch_index)
            target_normal = to_dense_batch(target_normal, target_batch_index)[0] if "normal" in target_attrs else None
            target_coord = (target_coord - anchor_center[:, None]) / anchor_scale[:, None, None]
            # # # DEBUG
            # anchor_coord_0 = anchor_coord[0].cpu().numpy()
            # anchor_normal_0 = anchor_normal[0].cpu().numpy() if anchor_normal is not None else None
            # target_coord_0 = target_coord[0].cpu().numpy()
            # target_normal_0 = target_normal[0].cpu().numpy() if target_normal is not None else None
            # anchor_pcd = o3d.geometry.PointCloud()
            # anchor_pcd.points = o3d.utility.Vector3dVector(anchor_coord_0)
            # if anchor_normal_0 is not None:
            #     anchor_pcd.normals = o3d.utility.Vector3dVector(anchor_normal_0)
            # anchor_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            # target_pcd = o3d.geometry.PointCloud()
            # target_pcd.points = o3d.utility.Vector3dVector(target_coord_0)
            # if target_normal_0 is not None:
            #     target_pcd.normals = o3d.utility.Vector3dVector(target_normal_0)
            # target_pcd.paint_uniform_color([0.0, 0.0, 1.0])
            # o3d.visualization.draw_geometries([anchor_pcd, target_pcd])
            # # # END DEBUG
            target_points[0] = to_flat_batch(target_coord, target_mask)[0]
            anchor_points[0] = to_flat_batch(anchor_coord, anchor_mask)[0]

        target_points, all_target_points, target_cluster_indices, target_attrs = self.target_pcd_transformer(target_points, return_full=True, **target_attrs)
        # Encode anchor pcd
        anchor_points, all_anchor_points, anchor_cluster_indices, anchor_attrs = self.anchor_pcd_transformer(anchor_points, return_full=True, **anchor_attrs)

        # DEBUG:
        # self.visualize_pcd_pyramids(all_anchor_points, anchor_cluster_indices, anchor_attrs)
        # Check the existence of nan
        if torch.isnan(target_points[1]).any() or torch.isnan(anchor_points[1]).any():
            print("Nan exists in the feature")
        return target_points, anchor_points, target_attrs, anchor_attrs

    def forward(self, anchor_points, target_points):
        anchor_coord, anchor_feat, anchor_offset = anchor_points
        target_coord, target_feat, target_offset = target_points
        # Conduct SAM-like matching
        knn_indices = dict()
        knn_indices["query2query"] = knn(query=target_coord, base=target_coord, query_offset=target_offset, base_offset=target_offset, k=self.query2query_knn)[0]
        knn_indices["query2context"] = knn(query=target_coord, base=anchor_coord, query_offset=target_offset, base_offset=anchor_offset, k=self.query2context_knn)[0]
        knn_indices["context2query"] = knn(query=anchor_coord, base=target_coord, query_offset=anchor_offset, base_offset=target_offset, k=self.query2context_knn)[0]

        for sam_block in self.sam_blocks:
            target_points, anchor_points = sam_block(target_points, anchor_points, knn_indices)

        anchor_coord, anchor_feat, anchor_offset = anchor_points
        target_coord, target_feat, target_offset = target_points
        conf_matrix = self.dual_softmax_reposition.match(
            feat_a=target_feat, feat_b=anchor_feat, coord_a=target_coord, coord_b=anchor_coord, batch_index_a=offset2batch(target_offset), batch_index_b=offset2batch(anchor_offset)
        )
        return conf_matrix

    def to_gt_correspondence_matrix(self, conf_matrix, gt_corr):
        """
        Get the ground truth correspondence from the conf_matrix
        """
        gt_corr = self.dual_softmax_reposition.to_gt_correspondence_matrix(conf_matrix, gt_corr)
        return gt_corr

    ################################# UTILS #################################
    def reposition(self, coord, R, t, normal=None):
        """Reposition the coord with R and t, and normal with R.
        Args:
            coord (torch.Tensor): [B, N, 3]
            R (torch.Tensor): [B, 3, 3]
            t (torch.Tensor): [B, 3]
            normal (torch.Tensor, optional): [B, N, 3]. Defaults to None.
        """
        coord = torch.bmm(coord, R.transpose(1, 2)) + t[:, None]
        if normal is not None:
            normal = torch.bmm(normal, R.transpose(1, 2))
        return coord, normal

    ################################# DEBUG #################################
    def test_gt_corr(self, anchor_points, target_points, gt_corr):
        anchor_coord, anchor_feat, anchor_offset = anchor_points
        target_coord, target_feat, target_offset = target_points
        R, t, condition = self.dual_softmax_reposition.arun(
            conf_matrix=gt_corr, coord_a=target_coord, coord_b=anchor_coord, batch_index_a=offset2batch(target_offset), batch_index_b=offset2batch(anchor_offset)
        )
        return R, t, condition

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
