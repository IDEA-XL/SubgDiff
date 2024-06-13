import copy


import torch
from torch_geometric.data import Data

from torch_scatter import scatter
#from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from collections import defaultdict

# from confgf import utils
# from datasets.chem import BOND_TYPES, BOND_NAMES
from rdkit.Chem.rdchem import BondType as BT

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}

def rdmol_to_data(mol:Mol, pos=None,y=None, idx=None, smiles=None):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()
    # if smiles is None:
    #     smiles = Chem.MolToSmiles(mol)
    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    atom_count = defaultdict(int)
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        atom_count[atom.GetAtomicNum()] += 1
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
        
    try:
        name = mol.GetProp('_Name')
    except:
        name=None

    # data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
    #             rdmol=copy.deepcopy(mol), smiles=smiles,y=y,id=idx, name=name)
    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type)
    #data.nx = to_networkx(data, to_undirected=True)

    return data, atom_count