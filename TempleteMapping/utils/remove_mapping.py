from rdkit import Chem
from get_template import get_reaction_template


def remove_smarts_mapping(smarts):
    mol = Chem.MolFromSmarts(smarts)
    # 去除原子的mapping
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # 确保原子的写法唯一
    m = Chem.MolToSmarts(mol)
    return m


def remove_smiles_mapping_and_get_cano_smiles(smiles):
    # 将 SMILES 转化为分子对象
    mol = Chem.MolFromSmiles(smiles)

    # 去除映射
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')

    # 获取规范 SMILES
    cano_smiles = Chem.MolToSmiles(mol, canonical=True)

    return cano_smiles


def remove_tmp_mapping(tmp_smarts, return_smarts=True):  ## rc>>lg
    """
    :param tmp_smarts: Tmp_WithMapping
    :param return_smarts:
    if return_smarts == True:
        :return TmpSmarts_NoMapping
    else:
        :return TmpSmiles_WithMapping
    """

    rc, lg = tmp_smarts.split('>>')

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
          "n:14]1>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:12]1[cH:11][c:10]([CH:9]=[O:8])[n:14][cH:13]1"
    print('rxn: ', rxn)
    Tmp_WithMapping = get_reaction_template(reaction=rxn, super_general=True, canonical=False)
    print('super_general=True, canonical=False ---------------')
    print(Tmp_WithMapping)
    TmpSmarts_NoMapping = remove_tmp_mapping(tmp_smarts=Tmp_WithMapping, return_smarts=True)
    print('TmpSmarts_NoMapping (return_smarts=True): -------------', TmpSmarts_NoMapping)
    TmpSmiles_WithMapping = remove_tmp_mapping(tmp_smarts=Tmp_WithMapping, return_smarts=False)
    print('TmpSmiles_NoMapping (return_smarts=False): -------------', TmpSmiles_WithMapping)

    print(remove_smarts_mapping('O=[C;H0;D3;+0:1]1-[C@H;D3;+0:2]-[C@;H0;D4;+0:3]23-[CH2;D2;+0:4]-[CH2;D2;+0:5]-[N;H0;D3;+0:6]-[C@H;D3;+0:7](-[CH2;D2;+0:8]-[c;H0;D3;+0:9]:[c;H0;D3;+0:10]-2)-[C@;H0;D4;+0:11]-3(-[OH;D1;+0:12])-[CH2;D2;+0:13]-[CH2;D2;+0:14]-1'))
