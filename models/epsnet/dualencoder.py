import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
import numpy as np
from numpy import pi as PI
from tqdm.auto import tqdm

from utils.chem import BOND_TYPES
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, generate_symmetric_edge_noise, extend_graph_order_radius
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance, get_angle, get_dihedral, eq_transform
from models.epsnet.attention import GAT, SelfAttention
from models.epsnet.diffusion import get_timestep_embedding, get_beta_schedule
import pdb


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class DualEncoderEpsNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_timesteps=config.num_diffusion_timesteps
        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.model_type = config.type  # config.type  # 'diffusion'; 'dsm'
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)
        self.is_emb_time = True
        if self.is_emb_time:
            '''
            timestep embedding
            '''
            self.hidden_dim = config.hidden_dim
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(config.hidden_dim,
                                config.hidden_dim*4),
                torch.nn.Linear(config.hidden_dim*4,
                                config.hidden_dim*4),
            ])
            self.temb_proj = torch.nn.Linear(config.hidden_dim*4,
                                            config.hidden_dim)
            self.nonlinearity = nn.ReLU()
        self.is_emb_time = False
        """
        The graph neural network that extracts node-wise features.
        """
        self.encoder_global = SchNetEncoder(
            hidden_channels=config.hidden_dim,
            num_filters=config.hidden_dim,
            num_interactions=config.num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=config.cutoff,
            smooth=config.smooth_conv,
        )
        self.encoder_local = GINEncoder(
            hidden_dim=config.hidden_dim,
            num_convs=config.num_convs_local,
        )
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1], 
            activation=config.mlp_act
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1], 
            activation=config.mlp_act
        )

        ''' header for masked vector prediciton'''
        if self.model_type in ['selected_diffusion', 'subgraph_diffusion']:
            self.edge_encoder_mask = get_edge_encoder(config)
            self.encoder_mask = SchNetEncoder(
                hidden_channels=config.hidden_dim,
                num_filters=config.hidden_dim,
                num_interactions=config.num_convs,
                edge_channels=self.edge_encoder_global.out_channels,
                cutoff=config.cutoff,
                smooth=config.smooth_conv,
        )
            self.mask_pred = self.config.get("mask_pred", "MLP").upper()
            if self.mask_pred.upper()=="GAT":
                self.mask_predictor = GAT(in_channels=config.hidden_dim, hidden_channels=config.hidden_dim//2, out_channels=1, num_heads=2)
            elif self.mask_pred=='MLP':
                self.mask_predictor = MultiLayerPerceptron(
                    config.hidden_dim,
                    [config.hidden_dim, config.hidden_dim // 2, 1], 
                    activation=config.mlp_act
                )
            elif self.mask_pred=='2BMLP':
                self.mask_predictor = MultiLayerPerceptron(
                    config.hidden_dim*2,
                    [config.hidden_dim, config.hidden_dim // 2, 1], 
                    activation=config.mlp_act
                )
            else:
                raise
            self.CEloss = nn.CrossEntropyLoss()
            self.BCEloss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.register_parameter("mask_predictor", None)
            
        self.model_mask = nn.ModuleList([self.mask_predictor,self.temb,self.temb_proj])    

        self.model_global = nn.ModuleList([self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp])        
        '''
        Incorporate parameters together
        '''
        
        self.model_local = nn.ModuleList([self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp])

   

        if self.model_type in ['selected_diffusion','subgraph_diffusion','diffusion']:
            # denoising diffusion
            ## betas
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            betas = torch.from_numpy(betas).float()
            self.betas = nn.Parameter(betas, requires_grad=False)
            ## variances
            alphas = (1. - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            self.num_timesteps = self.betas.size(0)
        elif self.model_type == 'dsm':
            # denoising score matching
            sigmas = torch.tensor(
                np.exp(np.linspace(np.log(config.sigma_begin), np.log(config.sigma_end),
                                config.num_noise_level)), dtype=torch.float32)
            self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)
            self.num_timesteps = self.sigmas.size(0)  # betas.shape[0]



    def forward(self, atom_type, pos, bond_index, bond_type, batch, time_step, 
                edge_index=None, edge_type=None, edge_length=None, return_edges=False, 
                extend_order=True, extend_radius=True, is_sidechain=None):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        N = atom_type.size(0)
        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=N,
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                order=self.config.edge_order,
                cutoff=self.config.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=is_sidechain,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)
        local_edge_mask = is_local_edge(edge_type)  # (E, )

        # Emb time_step
        if self.model_type in ['selected_diffusion','diffusion',"subgraph_diffusion"]:
            # # timestep embedding
            
            if self.is_emb_time:
                temb = get_timestep_embedding(time_step, self.hidden_dim)
                temb = self.temb.dense[0](temb)
                # temb = self.nonlinearity(temb)
                # temb = self.temb.dense[1](temb)
                temb = self.temb_proj(self.nonlinearity(temb))  # (G, dim)

            sigma_edge = torch.ones(size=(edge_index.size(1), 1), device=pos.device)  # (E, 1)

        # Encoding global
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length,
            edge_type=edge_type
        )   
        # Embed edges
        # edge_attr += temb_edge

        # Global
        node_attr_global = self.encoder_global(
            z=atom_type,
            edge_index=edge_index,
            edge_length=edge_length,
            edge_attr=edge_attr_global,
        )
        if self.is_emb_time: node_attr_global = node_attr_global + 0.1*temb[batch]
        ## Assemble pairwise features
        # (h_i,h_j,e_ij)
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )    # (E_global, 2H)
        ## Invariant features of edges (radius graph, global)
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (1.0 / sigma_edge)    # (E_global, 1)
        
        # Encoding local
        edge_attr_local = self.edge_encoder_global(
        # edge_attr_local = self.edge_encoder_local(
            edge_length=edge_length,
            edge_type=edge_type
        )   # Embed edges
        # edge_attr += temb_edge

        # Local
        node_attr_local = self.encoder_local(
            z=atom_type,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )
        if self.is_emb_time: node_attr_local = node_attr_local + 0.1*temb[batch]
        ## Assemble pairwise features
        h_pair_local = assemble_atom_pair_feature(
            node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )    # (E_local, 2H)
        ## Invariant features of edges (bond graph, local)
        if isinstance(sigma_edge, torch.Tensor):
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge[local_edge_mask]) # (E_local, 1)
        else:
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge) # (E_local, 1)

        
            
        self.node_attr_global = node_attr_global
        self.node_attr_local= node_attr_local
        self.batch = batch

        if return_edges:
            if self.model_type in ['selected_diffusion', "subgraph_diffusion"]:
                if self.mask_pred=="MLP":
                    mask_emb=node_attr_global
                    # mask_emb=torch.cat([node_attr_local,node_attr_global],dim=-1)
                    node_mask_pred=self.mask_predictor(mask_emb)
                elif self.mask_pred=='2BMLP':
                    mask_emb=torch.cat([node_attr_local,node_attr_global],dim=-1)
                    node_mask_pred=self.mask_predictor(mask_emb)
                elif self.mask_pred.upper()=="GAT":
                    mask_emb=node_attr_global
                    node_mask_pred = self.mask_predictor(mask_emb,edge_index)
                return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask, node_mask_pred
            return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask
        else:
            return edge_inv_global, edge_inv_local
   
    def get_node_embeddings(self):
        self.node_embs = torch.cat([self.node_attr_local,self.node_attr_global],dim=-1)

        return self.node_embs
    def get_global_node_embeddings(self):


        return self.node_attr_global

    def get_graph_embeddings(self):
        # node_embs = torch.cat([self.node_attr_local,self.node_attr_global],dim=-1)

        # self.node_embs
        self.node_embs = torch.cat([self.node_attr_local,self.node_attr_global],dim=-1)
        self.graph_embs = scatter(self.node_embs, self.batch, dim=0, reduce='sum') # 【"add", "sum", "mean"]
        return self.graph_embs
    def get_graph_embeddings(self):
        self.node_embs = torch.cat([self.node_attr_local,self.node_attr_global],dim=-1)
        self.graph_embs = scatter(self.node_embs, self.batch, dim=0, reduce='sum') # 【"add", "sum", "mean"]
        return self.graph_embs

import torch.nn.functional as F
class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
    
class GraphPooling(torch.nn.Module):
    def __init__(self, hidden_channels, out_dim, readout="mean"):
        super(GraphPooling, self).__init__()
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.readout = readout
        self.output_layer = nn.Linear(hidden_channels, out_dim)

    def forward(self, x, batch):
        h = self.lin1(x)
        h = self.act(h)
        h = self.lin2(h)
        graph_embs = scatter(h, batch, dim=0, reduce=self.readout) # 【"add", "sum", "mean"]
        graph_embs = self.output_layer(graph_embs)
        return graph_embs

def is_bond(edge_type):
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


def is_local_edge(edge_type):
    return edge_type > 0


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -torch.clamp(edge_length - rng, min=0.0, max=float('inf')) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom
