import torch as th
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch import nn
from .layer.egat import EdgeGATConv
from GenScore.model.layer.subgraph import subgraph
from .transform_AK.transform import SubgraphsTransform
from .transform_AK.config import cfg, update_cfg
from torch_scatter import scatter, scatter_mean

from .transform_AK.element import MLP, DiscreteEncoder, Identity, VNUpdate


def glorot_orthogonal(tensor, scale):
    """Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
    if tensor is not None:
        th.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


class coords_update(nn.Module):
    def __init__(self, dim_dh, num_head, drop_rate=0.15):
        super().__init__()
        self.num_head = num_head
        self.attention2deltax = nn.Sequential(
            nn.Linear(dim_dh, dim_dh // 2),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(dim_dh // 2, self.num_head)
        )
        self.weighted_head_layer = nn.Linear(num_head, 1, bias=False)

    def forward(self, a_ij, coords, edge_index):
        i, j = edge_index
        delta_x = coords[i] - coords[j]
        delta_x = delta_x / (th.norm(delta_x, p=2, dim=-1).unsqueeze(dim=-1) + 1e-6)
        delta_x = delta_x * self.weighted_head_layer(self.attention2deltax(a_ij))
        delta_x = scatter(delta_x, index=i, reduce='sum', dim=0)
        coords += delta_x
        return coords


class MultiHeadAttentionLayer(nn.Module):
    """Compute attention scores with a DGLGraph's node and edge (geometric) features."""

    def __init__(self, num_input_feats, num_output_feats,
                 num_heads, using_bias=False, update_edge_feats=True, update_coords=False):
        super(MultiHeadAttentionLayer, self).__init__()

        # Declare shared variables
        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats
        self.update_coords = update_coords

        # Define node features' query, key, and value tensors, and define edge features' projection tensors
        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.edge_feats_projection = nn.Linear(num_input_feats + 1, self.num_output_feats * self.num_heads, bias=using_bias)
        
        self.coords_update = coords_update(self.num_output_feats * self.num_heads, self.num_heads)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        if self.using_bias:
            glorot_orthogonal(self.Q.weight, scale=scale)
            self.Q.bias.data.fill_(0)

            glorot_orthogonal(self.K.weight, scale=scale)
            self.K.bias.data.fill_(0)

            glorot_orthogonal(self.V.weight, scale=scale)
            self.V.bias.data.fill_(0)

            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
            self.edge_feats_projection.bias.data.fill_(0)
        else:
            glorot_orthogonal(self.Q.weight, scale=scale)
            glorot_orthogonal(self.K.weight, scale=scale)
            glorot_orthogonal(self.V.weight, scale=scale)
            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)

    def propagate_attention(self, edge_index, node_feats_q, node_feats_k, node_feats_v, edge_feats_projection, coords):
        row, col = edge_index
        e_out = None
        # Compute attention scores
        alpha = node_feats_k[row] * node_feats_q[col]
        # Scale and clip attention scores
        alpha = (alpha / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)
        # Use available edge features to modify the attention scores
        alpha = alpha * edge_feats_projection
        # Copy edge features as e_out to be passed to edge_feats_MLP
        if self.update_edge_feats:
            e_out = alpha

            if self.update_coords:
                coords = self.coords_update(alpha, coords, edge_index)

        # Apply softmax to attention scores, followed by clipping
        alphax = th.exp((alpha.sum(-1, keepdim=True)).clamp(-5.0, 5.0))
        # Send weighted values to target nodes
        wV = scatter_add(node_feats_v[row] * alphax, col, dim=0, dim_size=node_feats_q.size(0))
        z = scatter_add(alphax, col, dim=0, dim_size=node_feats_q.size(0))
        return wV, z, e_out, coords

    def forward(self, x, edge_attr, edge_index, coords):
        row, col = edge_index
        node_feats_q = self.Q(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_k = self.K(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_v = self.V(x).view(-1, self.num_heads, self.num_output_feats)
        edge_attr = th.cat([edge_attr, th.norm(coords[row] - coords[col], p=2, dim=-1, keepdim=True)*0.1], dim=-1)
        edge_feats_projection = self.edge_feats_projection(edge_attr).view(-1, self.num_heads, self.num_output_feats)
        wV, z, e_out, coords = self.propagate_attention(edge_index, node_feats_q, node_feats_k, node_feats_v,
                                                edge_feats_projection, coords)

        h_out = wV / (z + th.full_like(z, 1e-6))
        
        return h_out, e_out, coords


class GraphTransformerModule(nn.Module):
    """A Graph Transformer module (equivalent to one layer of graph convolutions)."""

    def __init__(
            self,
            num_hidden_channels,
            activ_fn=nn.SiLU(),
            residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
    ):
        super(GraphTransformerModule, self).__init__()

        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a Geometric Transformer module
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=True
        )

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        # MLP for edge features
        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)
        glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
        self.O_edge_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, data, node_feats, edge_feats):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats  # Cache node representations for first residual connection
        edge_feats_in1 = edge_feats  # Cache edge representations for first residual connection

        # Apply first round of normalization before applying geometric attention, for performance enhancement
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        # Get multi-head attention output using provided node and edge representations
        node_attn_out, edge_attn_out, data.pos = self.mha_module(node_feats, edge_feats, data.edge_index, data.pos)

        node_feats = node_attn_out.view(-1, self.num_output_feats)
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)

        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)

        node_feats = self.O_node_feats(node_feats)
        edge_feats = self.O_edge_feats(edge_feats)

        # Make first residual connection
        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection
            edge_feats = edge_feats_in1 + edge_feats  # Make first edge residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection
        edge_feats_in2 = edge_feats  # Cache edge representations for second residual connection

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
            edge_feats = self.layer_norm2_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm2_node_feats(node_feats)
            edge_feats = self.batch_norm2_edge_feats(edge_feats)

        # Apply MLPs for node and edge features
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)

        # Make second residual connection
        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection
            edge_feats = edge_feats_in2 + edge_feats  # Make second edge residual connection

        # Return edge representations along with node representations (for tasks other than interface prediction)
        return node_feats, edge_feats

    def forward(self, data, node_feats, edge_feats):
        """Perform a forward pass of a Geometric Transformer to get intermediate node and edge representations."""
        node_feats, edge_feats = self.run_gt_layer(data, node_feats, edge_feats)
        return node_feats, edge_feats


class FinalGraphTransformerModule(nn.Module):
    """A (final layer) Graph Transformer module that combines node and edge representations using self-attention."""

    def __init__(self,
                 num_hidden_channels,
                 activ_fn=nn.SiLU(),
                 residual=True,
                 num_attention_heads=4,
                 norm_to_apply='batch',
                 dropout_rate=0.1,
                 num_layers=4):
        super(FinalGraphTransformerModule, self).__init__()

        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a Geometric Transformer module
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=False)

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

    # glorot_orthogonal(self.conformation_module.weight, scale=scale)

    def run_gt_layer(self, data, node_feats, edge_feats):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats  # Cache node representations for first residual connection
        # edge_feats = self.conformation_module(edge_feats)

        # Apply first round of normalization before applying geometric attention, for performance enhancement
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        # Get multi-head attention output using provided node and edge representations
        node_attn_out, _, data.pos = self.mha_module(node_feats, edge_feats, data.edge_index, data.pos)
        node_feats = node_attn_out.view(-1, self.num_output_feats)
        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        node_feats = self.O_node_feats(node_feats)

        # Make first residual connection
        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm2_node_feats(node_feats)

        # Apply MLP for node features
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        # Make second residual connection
        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection

        # Return node representations
        return node_feats

    def forward(self, data, node_feats, edge_feats):
        """Perform a forward pass of a Geometric Transformer to get final node representations."""
        node_feats = self.run_gt_layer(data, node_feats, edge_feats)
        return node_feats


class GraphTransformer(nn.Module):
    """A graph transformer
	"""

    def __init__(
            self,
            in_channels,
            edge_features=10,
            num_hidden_channels=128,
            activ_fn=nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
            **kwargs
    ):
        super(GraphTransformer, self).__init__()

        # Initialize model parameters
        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Initializer Modules
        # --------------------
        # Define all modules related to edge and node initialization
        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)
        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a variable number of Geometric Transformer modules
        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
            num_hidden_channels=num_hidden_channels,
            activ_fn=activ_fn,
            residual=transformer_residual,
            num_attention_heads=num_attention_heads,
            norm_to_apply=norm_to_apply,
            dropout_rate=dropout_rate,
            num_layers=num_layers) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([FinalGraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=transformer_residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate,
                num_layers=num_layers)])
        self.gt_block = nn.ModuleList(gt_block_modules)

    def forward(self, data):
        node_feats = self.node_encoder(data.x)
        edge_feats = self.edge_encoder(data.edge_attr[:, 1:])

        # Apply a given number of intermediate geometric attention layers to the node and edge features given
        for gt_layer in self.gt_block[:-1]:
            node_feats, edge_feats = gt_layer(data, node_feats, edge_feats)

        # Apply final layer to update node representations by merging current node and edge representations
        node_feats = self.gt_block[-1](data, node_feats, edge_feats)
        data.x = node_feats
        data.edge_attr = edge_feats
        # return node_feats
        return data


# ==============================================
from .layer.gatedgcn_layer import GatedGCNLayer


class GatedGCN(nn.Module):
    """A graph transformer
	"""

    def __init__(
            self,
            in_channels,
            edge_features=10,
            num_hidden_channels=128,
            dropout_rate=0.1,
            num_layers=4,
            residual=True,
            equivstable_pe=False,
            **kwargs
    ):
        super(GatedGCN, self).__init__()

        # Initialize model parameters
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)

        gt_block_modules = [GatedGCNLayer(
            num_hidden_channels,
            num_hidden_channels,
            dropout_rate,
            residual,
            equivstable_pe=equivstable_pe) for _ in range(num_layers)]

        self.gt_block = nn.ModuleList(gt_block_modules)

    def forward(self, data):
        data.x = self.node_encoder(data.x)
        data.edge_attr = self.edge_encoder(data.edge_attr)

        # Apply a given number of intermediate geometric attention layers to the node and edge features given
        for gt_layer in self.gt_block:
            data = gt_layer(data)

        # Apply final layer to update node representations by merging current node and edge representations
        # data.x = node_feats
        # data.edge_attr = edge_feats
        # return node_feats
        return data


class SubGT(nn.Module):
    def __init__(self,
                 in_channels,
                 edge_features=10,
                 num_hidden_channels=128,
                 activ_fn=nn.SiLU(),
                 transformer_residual=True,
                 num_attention_heads=4,
                 norm_to_apply='batch',
                 dropout_rate=0.1,
                 num_layers=4,
                 **kwargs
                 ):
        super(SubGT, self).__init__()

        # Initialize model parameters
        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Initializer Modules
        # --------------------
        # Define all modules related to edge and node initialization
        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)
        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a variable number of Geometric Transformer modules
        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
            num_hidden_channels=num_hidden_channels,
            activ_fn=activ_fn,
            residual=transformer_residual,
            num_attention_heads=num_attention_heads,
            norm_to_apply=norm_to_apply,
            dropout_rate=dropout_rate,
            num_layers=num_layers) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([FinalGraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=transformer_residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate,
                num_layers=num_layers)])
        self.gt_block = nn.ModuleList(gt_block_modules)

        self.transform = SubgraphsTransform(cfg.subgraph.hops,
                                            walk_length=cfg.subgraph.walk_length,
                                            p=cfg.subgraph.walk_p,
                                            q=cfg.subgraph.walk_q,
                                            repeat=cfg.subgraph.walk_repeat,
                                            sampling_mode=cfg.sampling.mode,
                                            minimum_redundancy=cfg.sampling.redundancy,
                                            shortest_path_mode_stride=cfg.sampling.stride,
                                            random_mode_sampling_rate=cfg.sampling.random_rate,
                                            random_init=True)

        self.transform_eval = SubgraphsTransform(cfg.subgraph.hops,
                                                 walk_length=cfg.subgraph.walk_length,
                                                 p=cfg.subgraph.walk_p,
                                                 q=cfg.subgraph.walk_q,
                                                 repeat=cfg.subgraph.walk_repeat,
                                                 sampling_mode=None,
                                                 random_init=False)

        self.sub_GT = subgraph(self.gt_block, num_hidden_channels)

    def forward(self, data, model_type=None):
        edge_index = data.edge_index
        data.x = self.node_encoder(data.x)
        data.edge_attr = self.edge_encoder(data.edge_attr[:, :10])

        sample = dict()

        sample['subgraphs_batch'], sample['subgraphs_nodes_mapper'], sample['subgraphs_edges_mapper'], sample[
            'combined_subgraphs'], sample['hop_indicator'], sample['num_nodes'] = self.transform_eval(data.edge_index,
                                                                                                 data.x.size()[0])

        # Apply a given number of intermediate geometric attention layers to the node and edge features given

        data.x = self.sub_GT(data, sample, model_type)
        data.edge_index = edge_index

        return data


class SubEGAT(nn.Module):
    def __init__(
            self,
            in_channels,
            edge_features=10,
            num_hidden_channels=128,
            dropout_rate=0.1,
            num_layers=6,
            **kwargs
    ):
        super(SubEGAT, self).__init__()

        # Initialize model parameters
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)

        gt_block_modules = [EdgeGATConv(
            num_hidden_channels,
            num_hidden_channels,
            edge_dim=num_hidden_channels, heads=1) for _ in range(num_layers)]

        self.gt_block = nn.ModuleList(gt_block_modules)

        self.transform = SubgraphsTransform(cfg.subgraph.hops,
                                            walk_length=cfg.subgraph.walk_length,
                                            p=cfg.subgraph.walk_p,
                                            q=cfg.subgraph.walk_q,
                                            repeat=cfg.subgraph.walk_repeat,
                                            sampling_mode=cfg.sampling.mode,
                                            minimum_redundancy=cfg.sampling.redundancy,
                                            shortest_path_mode_stride=cfg.sampling.stride,
                                            random_mode_sampling_rate=cfg.sampling.random_rate,
                                            random_init=True)

        self.transform_eval = SubgraphsTransform(cfg.subgraph.hops,
                                                 walk_length=cfg.subgraph.walk_length,
                                                 p=cfg.subgraph.walk_p,
                                                 q=cfg.subgraph.walk_q,
                                                 repeat=cfg.subgraph.walk_repeat,
                                                 sampling_mode=None,
                                                 random_init=False)

        self.sub_EGAT = subgraph(self.gt_block, num_hidden_channels)

    def forward(self, data, model_type='dynamic'):
        edge_index = data.edge_index
        data.x = self.node_encoder(data.x)
        data.edge_attr = self.edge_encoder(data.edge_attr)

        sample = dict()

        sample['subgraphs_batch'], sample['subgraphs_nodes_mapper'], sample['subgraphs_edges_mapper'], sample[
            'combined_subgraphs'], sample['hop_indicator'], sample['num_nodes'] = self.transform_eval(data.edge_index,
                                                                                                 data.x.size()[0])

        # Apply a given number of intermediate geometric attention layers to the node and edge features given

        data.x = self.sub_EGAT(data, sample, model_type)
        data.edge_index = edge_index

        return data


# ==============================================
class GenScore(nn.Module):
    def __init__(self, ligand_model, target_model, in_channels, hidden_dim, n_gaussians, dropout_rate=0.15,
                 dist_threhold=1000):
        super(GenScore, self).__init__()

        self.ligand_model = ligand_model
        self.target_model = target_model
        self.MLP = nn.Sequential(nn.Linear(in_channels * 2, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ELU(),
                                 nn.Dropout(p=dropout_rate))

        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)
        self.atom_types = nn.Linear(in_channels, 17)
        self.bond_types = nn.Linear(in_channels * 2, 4)

        # self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        self.dist_threhold = dist_threhold

    def forward(self, data_ligand, data_target, train):
        h_l_pos, h_t_pos = data_ligand.pos, data_target.pos

        h_l = self.ligand_model(data_ligand)
        h_t = self.target_model(data_target)

        h_l_x, l_mask = to_dense_batch(h_l.x, h_l.batch, fill_value=0)
        h_t_x, t_mask = to_dense_batch(h_t.x, h_t.batch, fill_value=0)
        h_l_pos, _ = to_dense_batch(h_l_pos, h_l.batch, fill_value=0)
        h_t_pos, _ = to_dense_batch(h_t_pos, h_t.batch, fill_value=0)

        # assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
        (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
        self.B = B
        self.N_l = N_l
        self.N_t = N_t

        # Combine and mask
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_t, 1)  # [B, N_l, N_t, C_out]

        h_t_x = h_t_x.unsqueeze(-3)
        h_t_x = h_t_x.repeat(1, N_l, 1, 1)  # [B, N_l, N_t, C_out]

        C = th.cat((h_l_x, h_t_x), -1)
        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
        self.C = C = C[C_mask]
        C = self.MLP(C)

        # Get batch indexes for ligand-target combined features
        C_batch = th.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]

        # Outputs
        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C)) + 1.1
        mu = F.elu(self.z_mu(C)) + 1
        atom_types = self.atom_types(h_l.x)
        bond_types = self.bond_types(th.cat([h_l.x[h_l.edge_index[0]], h_l.x[h_l.edge_index[1]]], axis=1))

        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos.view(B, -1, 3))[C_mask]
        return pi, sigma, mu, dist.unsqueeze(1).detach(), atom_types, bond_types, C_batch

    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()

        dists = -2 * th.bmm(X, Y.permute(0, 2, 1)) + th.sum(Y ** 2, axis=-1).unsqueeze(1) + th.sum(X ** 2,
                                                                                                   axis=-1).unsqueeze(
            -1)
        return th.nan_to_num((dists ** 0.5), 10000)

