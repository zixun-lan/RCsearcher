import json
from rdkit import Chem
from smiles2dgl import smiles_to_dglGraph
from utils.other_utils import canonicalize


rule_path = "rules.json"
with open(rule_path, "r", encoding="utf-8") as f:
    rules = json.load(f)


def G2smiles(g, hasmap=True):
    node_labels = g.ndata["node_type"].to(int).reshape(1, -1).squeeze(0).tolist()
    edge_labels = g.adjacency_matrix().indices().permute(1, 0).tolist()
    bond_labels = g.edata["edge_type"].to(int)
    maps = g.ndata["mapping"].squeeze(1).tolist()
    charges = g.ndata["charge"].squeeze(1).tolist()
    hydrogens = g.ndata["hydrogen"].squeeze(1).tolist()
    mol = Chem.RWMol()

    def bond_decoder(x):
        x = x.tolist()
        for i, idx in enumerate(x):
            if idx == 1:
                return eval("Chem.rdchem.BondType." + rules["Bonds"][i])
        print("No Bond Type")

    for node_label in node_labels:
        mol.AddAtom(Chem.Atom(rules["Atoms"][node_label - 1]))
    for index, (start, end) in enumerate(edge_labels):
        if start > end:
            mol.AddBond(
                int(start), int(end), bond_decoder(bond_labels[index]))
    if hasmap:
        for idx, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(maps[idx])

    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetFormalCharge(charges[idx])
        atom.SetNumExplicitHs(hydrogens[idx])

    return Chem.MolToSmiles(mol, canonical=True)


if __name__ == '__main__':
    smiles = '[CH2:7]1[CH:8]([CH2:19][CH:4]([C:2](=[O:3])[NH2:1])[CH2:5][NH:6]1)[c:9]1[cH:18][cH:17][c:12]([cH:11][' \
             'cH:10]1)[C:13]([F:16])([F:14])[F:15] '
    graph = smiles_to_dglGraph(smiles)
    smiles2 = G2smiles(graph, hasmap=True)
    # print(smiles2)
    print('原始的smiles:', canonicalize(smiles))
    print('转换的smiles:', canonicalize(smiles2))
    print(canonicalize(smiles) == canonicalize(smiles2))
