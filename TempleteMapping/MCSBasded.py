import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdqueries
from utils.get_template import get_reaction_template
from utils.remove_mapping import remove_tmp_mapping
from utils.run_template import check_run


def exclude_mapped_atoms(query):
    # 创建一个新的查询分子，其中包含一个额外的原子类型查询，用于排除带有映射编号的原子
    new_query = Chem.RWMol()
    for atom in query.GetAtoms():
        new_atom = rdqueries.HasPropQueryAtom('molAtomMapNumber', True)
        new_atom.ExpandQuery(atom, Chem.rdchem.CompositeQueryType.COMPOSITE_AND)
        new_query.AddAtom(new_atom)
    for bond in query.GetBonds():
        new_query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return new_query.GetMol()


class CustomAtomCompare(rdFMCS.MCSAtomCompare):
    def compare(self, p, mol1, atom1, mol2, atom2):
        if mol1.GetAtomWithIdx(atom1).GetAtomMapNum() == 0 and mol2.GetAtomWithIdx(atom2).GetAtomMapNum() == 0:
            return mol1.GetAtomWithIdx(atom1).GetAtomicNum() == mol2.GetAtomWithIdx(atom2).GetAtomicNum()
        else:
            return False


def get_max_map_num(mol):
    # 初始化一个变量来记录最高的映射编号，初始值为0
    max_map_num = 0
    # 遍历分子对象的原子
    for atom in mol.GetAtoms():
        # 获取原子的映射编号
        map_num = atom.GetAtomMapNum()

        # 如果原子的映射编号大于当前的最高的映射编号，就更新最高的映射编号
        if map_num > max_map_num:
            max_map_num = map_num
    # 返回最高的映射编号
    return max_map_num


def mapping_by_sub(rc_smart, lg_smart):
    rc_mol = Chem.MolFromSmarts(rc_smart)
    lg_mol = Chem.MolFromSmarts(lg_smart)

    params = rdFMCS.MCSParameters()
    params.AtomTyper = CustomAtomCompare()
    # params = rdFMCS.MCSParameters()
    params.BondTyper = rdFMCS.BondCompare.CompareAny

    mcs = rdFMCS.FindMCS([rc_mol, lg_mol], params)
    # 将最大公共子图转换成分子对象
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    mcs_mol = exclude_mapped_atoms(mcs_mol)
    # print(111, mcs.smartsString)
    max_map_num = get_max_map_num(rc_mol)
    params = Chem.SubstructMatchParameters()
    params.useChirality = True
    params.uniquify = True
    # params.useQueryQueryMatches = True
    params.aromaticMatchesConjugated = True
    # 遍历两个分子对象的原子
    for mol in [rc_mol, lg_mol]:
        # 找到与最大公共子图匹配的原子索引
        # print(Chem.MolToSmarts(mol), Chem.MolToSmarts(mcs_mol))

        match = mol.GetSubstructMatch(mcs_mol, params)

        for index, i in enumerate(match):
            map_num = max_map_num + index + 1  # 根据索引值设置映射编号
            # print(map_num)
            # 获取原子对象
            atom = mol.GetAtomWithIdx(i)
            # 设置原子的映射编号为当前的编号
            atom.SetAtomMapNum(map_num)

    rc_smart_mapped = Chem.MolToSmarts(rc_mol)
    lg_smart_mapped = Chem.MolToSmarts(lg_mol)
    # 打印两个分子对象的SMILES字符串，带有记号
    # print('rc_mol:', rc_smart_mapped)
    # print('lg_mol:', lg_smart_mapped)
    return rc_smart_mapped, lg_smart_mapped


def MCS_maked(tmpNoMap):
    rc_smart, lgs_smart = tmpNoMap.split('>>')
    if '.' in rc_smart:
        raise Exception("rc不连通！")
    lgList = lgs_smart.split('.')

    dict_lgList = {}
    for lgsmart in lgList:
        dict_lgList[lgsmart] = Chem.MolFromSmarts(lgsmart).GetNumAtoms()

    # 1、对lglist 里面元素从小到大排序。
    lgList = [k for k, v in sorted(dict_lgList.items(), key=lambda item: item[1])]

    mapped_lgList = []

    for lg_smart in lgList:
        rc_smart, lg_smart = mapping_by_sub(rc_smart, lg_smart)
        mapped_lgList.append(lg_smart)
        # 2、计算ignore_atoms，计算当前rc_smart带mapping的atom的index
    react = ".".join(mapped_lgList)
    tep = rc_smart + ">>" + react
    return tep


if __name__ == '__main__':
    rxn = "[CH3:1][CH2:2][n:3]1[n:4][n:5][n:6][c:7]1[C:8]([CH:9]=[CH:10][CH:11]=[O:12])=[C:13]([c:14]1[cH:15][cH:16][" \
          "c:17]([F:18])[cH:19][cH:20]1)[c:21]1[cH:22][cH:23][c:24]([F:25])[cH:26][cH:27]1.[CH3:28][CH2:29][O:30][" \
          "C:31](=[O:32])[CH2:33][C:34]([CH3:35])=[O:36]>>[CH3:1][CH2:2][n:3]1[n:4][n:5][n:6][c:7]1[C:8]([CH:9]=[" \
          "CH:10][CH:11]([OH:12])[CH2:35][C:34]([CH2:33][C:31]([O:30][CH2:29][CH3:28])=[O:32])=[O:36])=[C:13]([" \
          "c:14]1[cH:15][cH:16][c:17]([F:18])[cH:19][cH:20]1)[c:21]1[cH:22][cH:23][c:24]([F:25])[cH:26][cH:27]1 "
    print('rxn: ', rxn)
    raw_tmp = get_reaction_template(reaction=rxn, super_general=True, canonical=False)
    print('raw tmp:', raw_tmp)
    tmp = remove_tmp_mapping(tmp_smarts=raw_tmp, return_smarts=True)
    print('tmpNomapping: ', tmp)
    tmp = MCS_maked(tmp)
    print('composed tmp:', tmp)
    print('check raw_tmp: ', check_run(raw_tmp, rxn))
    print('check composed_tmp:', check_run(tmp, rxn))
