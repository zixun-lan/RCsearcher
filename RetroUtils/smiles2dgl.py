import torch
from rdkit.Chem import AllChem
from rdkit import Chem
from dgllife.utils import atom_type_one_hot, atom_total_degree_one_hot, atom_explicit_valence_one_hot, \
    atom_implicit_valence_one_hot, atom_hybridization_one_hot, atom_total_num_H_one_hot
from dgllife.utils import atom_formal_charge_one_hot, atom_is_aromatic_one_hot, \
    atom_is_in_ring_one_hot
from dgllife.utils import ConcatFeaturizer, atomic_number, BaseAtomFeaturizer, bond_type_one_hot, \
    bond_is_conjugated_one_hot, bond_is_in_ring_one_hot, bond_stereo_one_hot, bond_direction_one_hot
from dgllife.utils import BaseBondFeaturizer, smiles_to_bigraph, get_mol_3d_coordinates


def smiles_to_dglGraph(smiles, seed=0, add_self_loop=False, Conformer=False):
    # atom_concat_featurizer = ConcatFeaturizer([atom_type_one_hot, atom_total_degree_one_hot,
    #                                            atom_explicit_valence_one_hot,
    #                                            atom_implicit_valence_one_hot,
    #                                            atom_hybridization_one_hot,
    #                                            atom_total_num_H_one_hot,
    #                                            atom_formal_charge_one_hot,
    #                                            atom_num_radical_electrons_one_hot,
    #                                            atom_is_aromatic_one_hot,
    #                                            atom_is_in_ring_one_hot,
    #                                            atom_chiral_tag_one_hot,
    #                                            atom_chirality_type_one_hot,
    #                                            atom_is_chiral_center
    #                                            ])
    atom_concat_featurizer = ConcatFeaturizer([atom_type_one_hot, atom_total_degree_one_hot,
                                               atom_explicit_valence_one_hot,
                                               atom_implicit_valence_one_hot,
                                               atom_hybridization_one_hot,
                                               atom_total_num_H_one_hot,
                                               atom_formal_charge_one_hot,
                                               atom_is_aromatic_one_hot,
                                               atom_is_in_ring_one_hot
                                               ])

    atom_type = ConcatFeaturizer([atomic_number])

    # Construct a featurizer for featurizing all atoms in a molecule
    mol_atom_featurizer = BaseAtomFeaturizer({'x': atom_concat_featurizer, 'node_type': atom_type})

    # bond_concat_featurizer = ConcatFeaturizer(
    #     [bond_type_one_hot, bond_is_conjugated_one_hot, bond_is_in_ring_one_hot, bond_stereo_one_hot,
    #      bond_direction_one_hot])
    bond_concat_featurizer = ConcatFeaturizer(
        [bond_type_one_hot, bond_is_conjugated_one_hot, bond_is_in_ring_one_hot, bond_stereo_one_hot,
         bond_direction_one_hot])
    bond_type = ConcatFeaturizer([bond_type_one_hot])
    # hbj = ConcatFeaturizer([bond_direction_one_hot])

    # Construct a featurizer for featurizing all bonds in a molecule
    mol_bond_featurizer = BaseBondFeaturizer({'e': bond_concat_featurizer, 'edge_type': bond_type})

    g = smiles_to_bigraph(smiles, add_self_loop=add_self_loop,
                          node_featurizer=mol_atom_featurizer,
                          edge_featurizer=mol_bond_featurizer,
                          canonical_atom_order=False,
                          explicit_hydrogens=False,
                          num_virtual_nodes=0)

    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    mapping = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        mapping.append(atom.GetAtomMapNum())

    charge = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        charge.append(atom.GetFormalCharge())
    g.ndata['charge'] = torch.tensor(charge).reshape(-1, 1)

    hydrogen = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        hydrogen.append(atom.GetNumExplicitHs())
    g.ndata['hydrogen'] = torch.tensor(hydrogen).reshape(-1, 1)

    if sum(mapping) == 0:
        g.ndata['mapping'] = torch.arange(num_atoms).reshape(-1, 1)
    else:
        g.ndata['mapping'] = torch.tensor(mapping).reshape(-1, 1)

    if Conformer:
        mol = Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        AllChem.MMFFOptimizeMolecule(mol)
        coords = get_mol_3d_coordinates(mol)
        g.ndata['coords'] = torch.tensor(coords, dtype=torch.float32)
    return g


if __name__ == '__main__':
    smiles = '[CH2:7]1[CH:8]([CH2:19][CH:4]([C:2](=[O:3])[NH2:1])[CH2:5][NH:6]1)[c:9]1[cH:18][cH:17][c:12]([cH:11][' \
             'cH:10]1)[C:13]([F:16])([F:14])[F:15] '
    graph = smiles_to_dglGraph(smiles)
    print(graph)
    print(graph.num_nodes())
    # print(graph.ndata['node_type'])
    # print(graph.ndata['mapping'])
