import os
import sys

sys.path.append('..')
from rdkit import Chem
from rdchiral.template_extractor import mols_from_smiles_list, replace_deuterated, extract_from_reaction_plus


def canonicalize(smiles, isomericSmiles=True):
    mol = Chem.MolFromSmiles(smiles)
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomericSmiles)
    return smiles


def remove_smarts_fragment_mapping(smarts):
    mol = Chem.MolFromSmarts(smarts)
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    return Chem.MolToSmarts(mol)
    # return Chem.MolToSmiles(mol)


def products_process(products):
    products = mols_from_smiles_list(replace_deuterated(products).split('.'))

    for i in range(len(products)):
        products[i] = Chem.RemoveHs(products[i])  # *might* not be safe

    [Chem.SanitizeMol(mol) for mol in products]  # redundant w/ RemoveHs
    [mol.UpdatePropertyCache() for mol in products]
    return products


def get_reaction_template(smi, super_general=False, canonical=True):  # 这里输入的是化学字符串
    reactants, products = smi.split('>>')

    reaction_dict = {'reactants': reactants,
                     "products": products,
                     '_id': 000}  # 这里是未来转换格式以适应extract_from_reaction需要的格式，然后可以得到template
    template = extract_from_reaction_plus(reaction_dict, super_general=super_general, canonical=canonical)

    return template
    # print(template['reaction_smarts'])可以拆解开来看看template到底是什么


if __name__ == '__main__':
    smi = 'O[C:37]([c:11]1[c:10](-[c:7]2[cH:6][cH:5][c:4]([F:3])[cH:9][cH:8]2)[o:36][c:13]2[c:12]1[cH:17][c:16](-[' \
          'c:18]1[cH:19][c:20]([C:24]([NH:25][C:26]([CH3:27])([CH3:28])[c:29]3[cH:30][cH:31][cH:32][cH:33][cH:34]3)=[' \
          'O:35])[cH:21][cH:22][cH:23]1)[cH:15][n:14]2)=[O:38].CC(C)[N:42](C(C)C)[CH2:41]C>>[F:3][c:4]1[cH:5][cH:6][' \
          'c:7](-[c:10]2[c:11]([C:37](=[O:38])[NH:42][CH3:41])[c:12]3[c:13]([n:14][cH:15][c:16](-[c:18]4[cH:19][' \
          'c:20]([C:24]([NH:25][C:26]([CH3:27])([CH3:28])[c:29]5[cH:30][cH:31][cH:32][cH:33][cH:34]5)=[O:35])[cH:21][' \
          'cH:22][cH:23]4)[cH:17]3)[o:36]2)[cH:8][cH:9]1 '
    tmp = get_reaction_template(smi, super_general=True, canonical=False)
    print(tmp)
