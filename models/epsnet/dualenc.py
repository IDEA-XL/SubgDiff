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
from models.epsnet.attention import GAT
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

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1).
        """
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


        if self.model_type in ['selected_diffusion','diffusion',"subgraph_diffusion"]:
            # DDPM loss implicit handle the noise variance scale conditioning
            # # timestep embedding
            if self.is_emb_time:
                temb = get_timestep_embedding(time_step, self.hidden_dim)
                temb = self.temb.dense[0](temb)
                # temb = self.nonlinearity(temb)
                # temb = self.temb.dense[1](temb)
                temb = self.temb_proj(self.nonlinearity(temb))  # (G, dim)

            sigma_edge = torch.ones(size=(edge_index.size(1), 1), device=pos.device)  # (E, 1)

        # Encoding global
        # 1. h_e_ij =  MLP(e_ij, d_ij)
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length,
            edge_type=edge_type
        )   
        # Embed edges
        # edge_attr += temb_edge

        # Global
        # 2. h_i=MLP(x_i, d_ij, h_e_ij )
        node_attr_global = self.encoder_global(
            z=atom_type,
            edge_index=edge_index,
            edge_length=edge_length,
            edge_attr=edge_attr_global,
        )
        if self.is_emb_time: node_attr_global = node_attr_global + 0.1*temb[batch]
        ## Assemble pairwise features
        # h_ij = (h_i,h_j,h_e_ij)
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )    # (E_global, 2H)
        ## Invariant features of edges (radius graph, global)
        # s_theta_ij = MLP(h_ij)
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (1.0 / sigma_edge)    # (E_global, 1)
        
        # Encoding local
        # 1. h_e_ij =MLP(e_ij, d_ij)
        edge_attr_local = self.edge_encoder_global(
        # edge_attr_local = self.edge_encoder_local(
            edge_length=edge_length,
            edge_type=edge_type
        )   # Embed edges
        # edge_attr += temb_edge

        # Local
        # 2. h_i=MLP(x_i, e_ij)
        node_attr_local = self.encoder_local(
            z=atom_type,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )
        
        if self.is_emb_time: node_attr_local = node_attr_local + 0.1*temb[batch]
        ## Assemble pairwise features
        # 3. h_e_ij= [h_i * h_j, h_e_ij]
        h_pair_local = assemble_atom_pair_feature(
            node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )    # (E_local, 2H)
        ## Invariant features of edges (bond graph, local)
        # s_theta_ij= MLP(h_e_ij)
        if isinstance(sigma_edge, torch.Tensor):
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge[local_edge_mask]) # (E_local, 1)
        else:
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge) # (E_local, 1)

        
        # if self.model_type in ['selected_diffusion']:
            

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
    

    def get_loss(self, data, atom_type, pos, bond_index, bond_type, batch, num_nodes_per_graph, num_graphs, 
                 anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, is_sidechain=None):
        if self.model_type == "subgraph_diffusion":
            return self.get_loss_diffusion_subgraph(data, atom_type, pos, bond_index, bond_type, batch, num_nodes_per_graph, num_graphs, 
                anneal_power, return_unreduced_loss, return_unreduced_edge_loss, extend_order, extend_radius, is_sidechain)

    def get_loss_diffusion_subgraph(self, data, atom_type, pos, bond_index, bond_type, batch, num_nodes_per_graph, num_graphs, 
                 anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, is_sidechain=None):
        N = atom_type.size(0)
        node2graph = batch

        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        
        time_step = data.time_step.squeeze()
        last_select = data.last_select
        a_pos =data.alpha
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        if hasattr(data,"noise_scale"):
            pos_perturbed = pos + pos_noise * (data.noise_scale).sqrt() / a_pos.sqrt()
            if (data.noise_scale==0).any():
                print("noise_scale is 0", (data.noise_scale==0).sum())
            if torch.isnan(pos_perturbed).any():
                print(a_pos)
        else:
            pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()

        
        # Update invariant edge features
        edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask ,node_mask_pred= self(
            atom_type = atom_type,
            pos = pos_perturbed,
            bond_index = bond_index,
            bond_type = bond_type,
            batch = batch,
            time_step = time_step,
            return_edges = True,
            extend_order = extend_order,
            extend_radius = extend_radius,
            is_sidechain = is_sidechain
        )   # (E_global, 1), (E_local, 1)


        a_edge = a_pos[edge_index[0]]
        if hasattr(data,"noise_scale"):
            edge_noise_scale = data.noise_scale[edge_index[0]]
        else:
            edge_noise_scale = 1- a_edge
        

        selected_diffusion=True
         
        part_step_mask= False # for debug
        if selected_diffusion:
            
            last_select_label = last_select>last_select.min()
            loss_mask = 100*self.BCEloss(node_mask_pred, last_select_label.float())
            pred_select = nn.Sigmoid()(node_mask_pred) > 0.5
            #
            # For two part scale 
            if last_select.min()>0:
                self.two_part_p = last_select.min() 
                pred_select= pred_select + last_select.min() *(~pred_select)
            
            
            if part_step_mask:
                ## unmask the first 1/4 steps and leave the  rest steps masked. 
                non_mask_step=(data.time_step > self.config.num_diffusion_timesteps // 4 *3)[batch].unsqueeze(-1)
                pred_select = torch.where(non_mask_step, last_select.bool(), pred_select)
                # to make sure calculate the mask_step values
                loss_mask = torch.where(non_mask_step, (loss_mask[~non_mask_step].mean().detach() * last_select.shape[0] - loss_mask[~non_mask_step].sum(0).detach())/non_mask_step.sum(), loss_mask)
            
            
        else: 
            loss_mask = torch.zeros_like(node_mask_pred)
            pred_select = last_select
        # Compute original and perturbed distances
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)
        d_perturbed = edge_length
        # Filtering for protein
        train_edge_mask = is_train_edge(edge_index, is_sidechain)
        d_perturbed = torch.where(train_edge_mask.unsqueeze(-1), d_perturbed, d_gt)



        if self.config.edge_encoder == 'gaussian':
            # Distances must be greater than 0 
            d_sgn = torch.sign(d_perturbed)
            d_perturbed = torch.clamp(d_perturbed * d_sgn, min=0.01, max=float('inf'))

        d_target = (d_gt - d_perturbed) / edge_noise_scale.sqrt() * a_edge.sqrt()  # (E_global, 1), denoising direction

        if torch.isnan(d_target).any():
            # print(a_pos)
            print((edge_noise_scale==0).sum())
        global_mask = torch.logical_and(
                            torch.logical_or(d_perturbed <= self.config.cutoff, local_edge_mask.unsqueeze(-1)),
                            ~local_edge_mask.unsqueeze(-1)
                        )
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        ## Chain rule approach
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        
        
        if self.training:
            loss_global = (node_eq_global - target_pos_global)**2
            if selected_diffusion:
                loss_global = last_select * loss_global 
        else: # evaluation
            loss_global = (pred_select * node_eq_global - last_select * target_pos_global)**2
            loss_mask= ~torch.eq(pred_select,last_select) 
            
            if part_step_mask:
                loss_mask = loss_mask.float()
                loss_mask = torch.where(non_mask_step, (loss_mask[~non_mask_step].mean().detach() * last_select.shape[0] - loss_mask[~non_mask_step].sum(0).detach())/non_mask_step.sum(), loss_mask)

        loss_global = 2 * torch.sum(loss_global, dim=-1, keepdim=True)
        
        target_pos_local = eq_transform(d_target[local_edge_mask], pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        
        
        if self.training:
            loss_local = (node_eq_local - target_pos_local)**2
            if selected_diffusion:
                loss_local = last_select * loss_local
        else:
            loss_local = (node_eq_local - target_pos_local)**2
            loss_local = (pred_select * node_eq_local - last_select * target_pos_local)**2

        loss_local = 5 * torch.sum(loss_local, dim=-1, keepdim=True)
        

        # loss for atomic eps regression
        loss = loss_global + loss_local + loss_mask
        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss, loss_global, loss_local, loss_mask
        else:
            return loss

    def langevin_dynamics_sample_diffusion_subgraph(self, atom_type, pos_init, bond_index, bond_type, batch, num_graphs, extend_order, extend_radius=True, 
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)  
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # 
            return a
        
        def calculate_mean_sigma(mu1,sigma1_square,sigma2_square):
             mean = sigma1_square / (mu1**2 * sigma2_square+ sigma1_square).sqrt()
             mean = mean/mu1
             sigma_t_square = sigma1_square*sigma2_square/(mu1**2 * sigma2_square+ sigma1_square)
             
             return mean, sigma_t_square
        use_mask = kwargs.get("use_mask", True)
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt() #
        pos_traj = []
        if is_sidechain is not None:
            assert pos_gt is not None, 'need crd of backbone for sidechain prediction'
        with torch.no_grad():
            
            seq = range(max(0,self.num_timesteps-n_steps), self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])
            
            pos = pos_init * sigmas[-1]
            if is_sidechain is not None:
                pos[~is_sidechain] = pos_gt[~is_sidechain]

            # masked 
            self.one_minu_beta_sqrt = (1-self.betas).sqrt()
            p=0.5 # node select probability
            alpha_expect = (p*self.one_minu_beta_sqrt + 1-p)**2
            ## --------------- expectation state + multiple step mask ------------------------- ## 
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)

                edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask, node_mask_pred= self(
                            atom_type = atom_type,
                            pos = pos,
                            bond_index = bond_index,
                            bond_type = bond_type,
                            batch = batch,
                            time_step = t,
                            return_edges = True,
                            extend_order = extend_order,
                            extend_radius = extend_radius,
                            is_sidechain = is_sidechain
                        )   # (E_global, 1), (E_local, 1)
                # Local
                node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
                if clip_local is not None:
                    node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                ## mask vector
                pred_select = nn.Sigmoid()(node_mask_pred)>0.5
                if not use_mask:
                    pred_select[:] = True
                
                # Global
                
                if sigmas[i] < global_start_sigma:
                    # print('mask select zero ratio:',(pred_select==0).sum()/pred_select.shape[0])
                    edge_inv_global = edge_inv_global * (1-local_edge_mask.view(-1, 1).float())
                    node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                    # node_eq_local = 0
                else:
                    node_eq_global = 0

                # Sum
                eps_pos = node_eq_local +  node_eq_global * w_global #   eps_pos is the score, score is -noise
                
                # Update
                sampling_type = kwargs.get("sampling_type", 'same_mask_noisy')  

                noise = torch.randn_like(pos)  #  
                if sampling_type in ["same_mask_noisy"]:
                    b = self.betas # betas
                    t = t[0]
                    next_t = (torch.ones(1) * j).to(pos.device)
                    at = compute_alpha(b, t.long()) # self.alphas[t]  = \bar\alpha_t 
                    at_next = compute_alpha(b, next_t.long()) # self.alphas[t-1] = \bar\alpha_{t-1}
                    if sampling_type == "same_mask_noisy":
                        eps_0 = -eps_pos 
                        self.same_mask_steps =self.config.get("same_mask_steps", 250)


                        k = self.same_mask_steps
                        m = t.div(k,rounding_mode='floor') # time_step//k
                        mask_step = t % k
                        if mask_step==0 or t == self.num_timesteps -1 or t % k == 0:
                            same_mask = pred_select
                            # print('mask select zero ratio:{} || t:{}'.format(((pred_select==0).sum()/pred_select.shape[0]).item(),t.item()))
                        if mask_step==0:
                            m-=1
                            mask_step=k
                                            
                        mu1 = (1-same_mask*self.betas[t]).sqrt()
                        sigma1_square = same_mask*self.betas[t]
                        # computer expect -> t step
                        bern_beta_mask_t = self.betas[t-mask_step:t].repeat(same_mask.shape[0], 1)*same_mask
                        prod_gamma_tm1 = (1. - bern_beta_mask_t).prod(dim=-1,keepdim=True)

                        if not hasattr(self, "prod_one_minu_beta_sqrt"):
                            self.prod_one_minu_beta_sqrt =torch.zeros(self.num_timesteps//k)
                            self.prod_one_minu_beta_sqrt[0] = self.alphas[k-1]
                            for j in range(1, self.num_timesteps//k):
                                self.prod_one_minu_beta_sqrt[j] = (1-self.betas)[j*k:(j+1)*k].prod(dim=-1) #\prod_{i=(j-1)k+1}^{kj}(1-\beta_i)
                            self.prod_one_minu_beta = self.prod_one_minu_beta_sqrt
                            self.prod_one_minu_beta_sqrt = self.prod_one_minu_beta_sqrt.sqrt()

                            alpha_exp_m = (p*self.prod_one_minu_beta_sqrt + 1-p)**2
                            self.alpha_nodes_m= alpha_exp_m.cumprod(dim=-1)
                            self.noise_scale_m = self.alpha_nodes_m * ((1-self.prod_one_minu_beta)/self.alpha_nodes_m).cumsum(-1) * p**2
                        ## Lemma B.1      
                        sigma2_square = prod_gamma_tm1 * self.noise_scale_m[m] + 1-prod_gamma_tm1 

                        mean, sigma_t_square = calculate_mean_sigma(mu1,sigma1_square,sigma2_square)
                        mean = pos/mu1 - mean * eps_0 
                        mask = 1 - (t == 0).float()
                        logvar = (sigma_t_square).log()
                        pos_next = mean + mask* torch.exp(0.5 * logvar) * noise

                elif sampling_type == 'ld':
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = pos + step_size * eps_pos / sigmas[i] + noise * torch.sqrt(step_size*2)

                pos = pos_next

                if is_sidechain is not None:
                    pos[~is_sidechain] = pos_gt[~is_sidechain]

                if torch.isnan(pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                pos = center_pos(pos, batch)
                if clip_pos is not None:
                    pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(pos.clone().cpu())
            
        return pos, pos_traj
    


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
