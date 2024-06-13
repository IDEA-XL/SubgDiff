import copy
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import coalesce

from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np


from .chem import BOND_TYPES, BOND_NAMES, get_atom_symbol




class AddHigherOrderEdges(object):

    def __init__(self, order, num_types=len(BOND_TYPES)):
        super().__init__()
        self.order = order
        self.num_types = num_types

    def binarize(self, x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(self, adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        """
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    self.binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

        for i in range(2, order+1):
            adj_mats.append(self.binarize(adj_mats[i-1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order+1):
            order_mat = order_mat + (adj_mats[i] - adj_mats[i-1]) * i

        return order_mat

    def __call__(self, data: Data):
        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = self.get_higher_order_adj_matrix(adj, self.order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)   # (N, N)
        type_highorder = torch.where(adj_order > 1, self.num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
        data.is_bond = (data.edge_type < self.num_types)
        assert (data.edge_index == edge_index_1).all()

        return data

class AddEdgeLength(object):

    def __call__(self, data: Data):

        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data    


# Add attribute placeholder for data object, so that we can use batch.to_data_list
class AddPlaceHolder(object):
    def __call__(self, data: Data):
        data.pos_gen = -1. * torch.ones_like(data.pos)
        data.d_gen = -1. * torch.ones_like(data.edge_length)
        data.d_recover = -1. * torch.ones_like(data.edge_length)
        return data


class AddEdgeName(object):

    def __init__(self, asymmetric=True):
        super().__init__()
        self.bonds = copy.deepcopy(BOND_NAMES)
        self.bonds[len(BOND_NAMES) + 1] = 'Angle'
        self.bonds[len(BOND_NAMES) + 2] = 'Dihedral'
        self.asymmetric = asymmetric

    def __call__(self, data:Data):
        data.edge_name = []
        for i in range(data.edge_index.size(1)):
            tail = data.edge_index[0, i]
            head = data.edge_index[1, i]
            if self.asymmetric and tail >= head:
                data.edge_name.append('')
                continue
            tail_name = get_atom_symbol(data.atom_type[tail].item())
            head_name = get_atom_symbol(data.atom_type[head].item())
            name = '%s_%s_%s_%d_%d' % (
                self.bonds[data.edge_type[i].item()] if data.edge_type[i].item() in self.bonds else 'E'+str(data.edge_type[i].item()),
                tail_name,
                head_name,
                tail,
                head,
            )
            if hasattr(data, 'edge_length'):
                name += '_%.3f' % (data.edge_length[i].item())
            data.edge_name.append(name)
        return data


class AddAngleDihedral(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def iter_angle_triplet(bond_mat):
        n_atoms = bond_mat.size(0)
        for j in range(n_atoms):
            for k in range(n_atoms):
                for l in range(n_atoms):
                    if bond_mat[j, k].item() == 0 or bond_mat[k, l].item() == 0: continue
                    if (j == k) or (k == l) or (j >= l): continue
                    yield(j, k, l)

    @staticmethod
    def iter_dihedral_quartet(bond_mat):
        n_atoms = bond_mat.size(0)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i >= j: continue
                if bond_mat[i,j].item() == 0:continue
                for k in range(n_atoms):
                    for l in range(n_atoms):
                        if (k in (i,j)) or (l in (i,j)): continue
                        if bond_mat[k,i].item() == 0 or bond_mat[l,j].item() == 0: continue
                        yield(k, i, j, l)

    def __call__(self, data:Data):
        N = data.num_nodes
        if 'is_bond' in data:
            bond_mat = to_dense_adj(data.edge_index, edge_attr=data.is_bond).long().squeeze(0) > 0
        else:
            bond_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).long().squeeze(0) > 0

        # Note: if the name of attribute contains `index`, it will automatically
        #       increases during batching.
        data.angle_index = torch.LongTensor(list(self.iter_angle_triplet(bond_mat))).t()
        data.dihedral_index = torch.LongTensor(list(self.iter_dihedral_quartet(bond_mat))).t()

        return data


class CountNodesPerGraph(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.num_nodes_per_graph = torch.LongTensor([data.num_nodes])
        return data

from torch import nn

class SubgraphNoiseTransform(object):
    """ expectation state + k-step same-subgraph diffusion"""
    def __init__(self, config, tag='', ddpm=False, boltzmann_weight=False):

        self.config = config
        self.ddpm=ddpm # typical DDPM, each atom views as independent point 
        self.tag= tag

        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        self.one_minu_beta_sqrt = (1-self.betas).sqrt()
        ## variances
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        check_alphas(self.alphas)
        self.num_timesteps = self.betas.size(0)
        self.same_mask_steps =self.config.get("same_mask_steps", 250) # same subgraph step k



    def __call__(self, data):
        # select conformer
        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        # Sample noise levels
        atom_type=data.atom_type
        pos=data.pos
        bond_index=data.edge_index
        bond_type=data.edge_type
        num_nodes_per_graph=data.num_nodes_per_graph
        mask_subgraph = data.mask_subgraph
        time_step = torch.randint(
            1, self.num_timesteps, size=(1,), device=pos.device)
        # time_step =0 is the beta_1 in Eq

        data.time_step = time_step
        beta = self.betas.index_select(-1, time_step)
        data.beta = beta

        if self.ddpm:
            alpha= self.alphas.index_select(-1, time_step)
            alpha = alpha.expand(pos.shape[0], 1)
            data.alpha = alpha
            data.last_select=torch.ones_like(alpha)
            data.noise_scale = 1- alpha
            del data.mask_subgraph
            return data
        


        last_select, alpha, noise_scale = self._get_alpha_nodes_expect_same_mask(mask_subgraph, time_step, k =self.same_mask_steps)

        data.alpha = alpha
        data.last_select=last_select # # for predict last selected subgraph 
        data.noise_scale = noise_scale
        del data.mask_subgraph
        return data
    
    
    def _get_alpha_nodes_expect_same_mask(self, mask_subgraph, time_step, p=0.5, k=250):
        """ expectation stata + k-same subgraph(mask) step diffusion """
        expect_step = time_step.div(k,rounding_mode='floor') # m := time_step//k
        mask_step = time_step % k
        
        if mask_step==0:
            expect_step-=1
            mask_step=k
        selected_index = torch.randint(low=0, high=mask_subgraph.shape[-1], size=(1,))
        selected_node =mask_subgraph.index_select(-1, selected_index) # mask_subgraph[:,[1,2,5]]
        if expect_step==0:
            selected_nodes = selected_node.repeat(1, mask_step+1)
            selected_nodes[:,[i for i in range(0, min(3,time_step+1))]] = True
            bern_beta_mask_t = self.betas[0:time_step+1]*selected_nodes
            
            alpha_t = (1. - bern_beta_mask_t).prod(dim=-1,keepdim=True)

            return selected_node, alpha_t, 1-alpha_t
        
        ## Phase I: compute t step mean state 0: km
        p = mask_subgraph.sum(-1).unsqueeze(-1)/mask_subgraph.size(-1) # node selection probability
        if self.tag.startswith('Recover_GeoDiff'):
            p = torch.ones_like(p)
            selected_node[:][:] = True
        if not hasattr(self, "prod_one_minu_beta_sqrt"):
            self.prod_one_minu_beta_sqrt =torch.zeros(self.num_timesteps//k)
            self.prod_one_minu_beta_sqrt[0] = self.alphas[k-1]
            # For every k step, we calcualte a expectation state  
            for j in range(1, self.num_timesteps//k):
                self.prod_one_minu_beta_sqrt[j] = (1-self.betas)[j*k:(j+1)*k].prod(dim=-1) # \prod_{i=(j-1)k+1}^{kj}(1-\beta_i)
            self.prod_one_minu_beta_sqrt = self.prod_one_minu_beta_sqrt.sqrt()
            
        alpha_exp = (p*self.prod_one_minu_beta_sqrt[:expect_step] + 1-p)**2 # \alpha_j= (p\sqrt{\prod_{i=(j-1)k+1}^{kj}(1-\beta_i)} + 1-p)^2
        alpha_nodes= alpha_exp.cumprod(dim=-1) # \bar\alpha
        noise_scale = (alpha_nodes[:,expect_step-1]/alpha_nodes).mm( (1-self.prod_one_minu_beta_sqrt[:expect_step]**2).unsqueeze(-1) ) * p**2 # [ p\sqrt{\sum_{l=1}^{m} \frac{\bar\alpha_{m}}{\bar\alpha_{l}} (1-\prod_{i=(l-1)k+1}^{kl}(1-\beta_i))} ]^2


        ## Phase II: time step:  km+1 ->  t
        bern_beta_mask_t = self.betas[time_step-mask_step:time_step+1].repeat(selected_node.shape[0], 1)*selected_node
        alpha_t = (1. - bern_beta_mask_t).prod(dim=-1,keepdim=True)
        ## combine
        alpha_node_t = alpha_t * alpha_nodes.index_select(-1, expect_step-1)
        noise_scale_t = alpha_t * noise_scale  + 1-alpha_t
        if (alpha_node_t==0).sum():
            print(alpha_node_t,time_step)

        return selected_node, alpha_node_t, noise_scale_t
    

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(sigma_min={self.sigma_min}, '
                f'sigma_max={self.sigma_max})')


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

def modify_conformer(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v] # convention: positive rotation if pointing inwards. NOTE: DIFFERENT FROM THE PAPER!
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos

def get_transformation_mask(pyg_data):
    ''' get subgraph distribution'''
    G = to_networkx(pyg_data, to_undirected=False)
    to_rotate = []
    edge_set=[]
    edges = pyg_data.edge_index.T.numpy()
    for i in range(0, edges.shape[0]):
        # assert edges[i, 0] == edges[i+1, 1]
        edge = list(edges[i])
        if  edge[::-1] in edge_set: # remove replicated undircted edge
            continue
        edge_set.append(edge)
        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len))
            if len(l[0]) > 1:
                to_rotate.extend(l)

    if len(to_rotate)==0:
        to_rotate.append(list(G.nodes()))

    mask_rotate = np.zeros((len(G.nodes()), len(to_rotate)), dtype=bool)
    for i in range(len(to_rotate)):
        mask_rotate.T[i][list(to_rotate[i])] = True
    pyg_data.mask_subgraph = torch.tensor(mask_rotate)
    return pyg_data

def get_alpah_nodes_schedule(pyg_data, config):
    mask_subgraph=pyg_data.mask_subgraph
    selected_index= torch.randint(low=0,high=mask_subgraph.shape[-1], size=(config.num_diffusion_timesteps,))
    selected_nodes=mask_subgraph.index_select(-1, selected_index) # mask_subgraph[:,[1,2,5]]
    # gurantee the noises have been added into every node
    selected_nodes[:,[i for i in range(0, min(3,config.num_diffusion_timesteps+1))]] = True
    
    betas = get_beta_schedule(
        beta_schedule=config.beta_schedule,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        num_diffusion_timesteps=config.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float()

    betas_nodes = betas * selected_nodes 
    alpha_nodes = (1. - betas_nodes).cumprod(dim=-1) # alpha_t for each node
    pyg_data.selected_nodes, pyg_data.alpha_nodes = selected_nodes, alpha_nodes
    return pyg_data

def check_alphas(alphas):
    for n,a in enumerate(alphas): 
        if a==0:
            print(f"Warning bar alpha become zero at {n}-th time_step"); 
            break
    print("The smallest alpha is ", a.item())

