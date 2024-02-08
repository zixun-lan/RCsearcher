import os
import sys

sys.path.append('..')
from rdkit import Chem
from utils.get_template import get_reaction_template


def remove_smarts_mapping(smarts):
    mol = Chem.MolFromSmarts(smarts)
    # 去除原子的mapping
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # 确保原子的写法唯一
    m = Chem.MolToSmarts(mol)
    return m


def remove_smiles_mapping_and_get_cano_smiles(smiles, isomericSmiles=True):
    # 将 SMILES 转化为分子对象
    mol = Chem.MolFromSmiles(smiles)

    # 去除映射
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')

    # 获取规范 SMILES
    cano_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomericSmiles)

    return cano_smiles


def remove_tmp_mapping(smiles, return_smarts=True):  ## rc>>lg
    rc, lg = smiles.split('>>')

    mol_rc = Chem.MolFromSmarts(rc)
    for atom in mol_rc.GetAtoms():
        if return_smarts:
            atom.SetAtomMapNum(0)
    mol_lg = Chem.MolFromSmarts(lg)
    for atom in mol_lg.GetAtoms():
        if return_smarts:
            atom.SetAtomMapNum(0)

    if return_smarts:
        rst = Chem.MolToSmarts(mol_rc) + '>>' + Chem.MolToSmarts(mol_lg)
    else:
        rst = Chem.MolToSmiles(mol_rc) + '>>' + Chem.MolToSmiles(mol_lg)
    return rst


if __name__ == '__main__':
    rxn = "CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7].[O:8]=[CH:9][c:10]1[cH:11][nH:12][cH:13][" \
          "n:14]1>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:12]1[cH:11][c:10]([CH:9]=[O:8])[n:14][cH:13]1 "
    tmp = get_reaction_template(rxn, super_general=True, canonical=True)
    tmp_no_mapping = remove_tmp_mapping(tmp, return_smarts=True)
    print('tmp: ', tmp)
    print('tmp_no_mapping: ', tmp_no_mapping)
