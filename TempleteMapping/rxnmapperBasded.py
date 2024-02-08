import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

from rxnmapper import RXNMapper
from rdkit import Chem
from rdkit.Chem import rdFMCS
from utils.get_template import get_reaction_template
from utils.remove_mapping import remove_tmp_mapping
from utils.run_template import check_run

rxn_mapper = RXNMapper()


def map_transfor(mol1, mol2):  # mol2 带mapping的smile,mol1 不带mapping的smart
    mol1 = Chem.MolFromSmarts(mol1)
    mol2 = Chem.MolFromSmarts(mol2)
    num_1 = mol1.GetNumAtoms()
    num_2 = mol2.GetNumAtoms()
    mcs = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    # print(mcs.smartsString)
    pub_num = mcs_mol.GetNumAtoms()  # 公共子图smart形式的字母数目，
    # print(num_2, num_1, pub_num)
    if num_2 == num_1 == pub_num:

        mol2_dic = {}
        for idx, atom in enumerate(mol2.GetAtoms()):
            mol2_dic[idx] = atom.GetAtomMapNum()
        key_mol2 = mol2.GetSubstructMatch(mcs_mol)
        mapping = [mol2_dic[i] for i in key_mol2]
        values = mol1.GetSubstructMatch(mcs_mol)
        mol2_mol1_dic = dict(zip(key_mol2, values))
        key_mol1 = [mol2_mol1_dic[i] for i in key_mol2]

        mol1_mapped = dict(zip(key_mol1, mapping))

        # 读取smart字符串为mol对象
        mol = mol1
        for key, value in mol1_mapped.items():
            atom = mol.GetAtomWithIdx(key)
            atom.SetAtomMapNum(value)

        return Chem.MolToSmarts(mol)
    else:
        return None


def main_function(input_string1, input_string2):
    # 检查字符串中是否有 "."
    react_list = []
    if "." in input_string1 and "." in input_string2:
        # 使用 "." 为分隔符将字符串分割
        reactants1 = input_string1.split(".")
        reactants2 = input_string2.split(".")
        # 迭代每个分割后的字符串并调用函数
        for reactant1 in reactants1:
            for reactant2 in reactants2:
                try:
                    a = map_transfor(reactant1, reactant2)
                    if a is not None:
                        react_list.append(a)

                except:
                    print("Error: failed to process reactant pair:", reactant1, reactant2)
    else:
        # 如果字符串中没有 "."，则直接处理字符串
        try:
            a = map_transfor(input_string1, input_string2)
            if a is not None:
                react_list.append(a)
        except:
            print("Error: failed to process reactant pair:", input_string1, input_string2)
    return react_list


def RXN_compose_template(smarts_template):
    smarts_rc, smarts_lg = smarts_template.split('>>')
    smarts_template_forward = smarts_lg + '>>' + smarts_rc

    rxns = [smarts_template_forward]

    # test whether change rxn module in env
    # results = rxn_mapper.get_attention_guided_atom_maps(rxns)
    try:
        results = rxn_mapper.get_attention_guided_atom_maps(rxns)
    except:
        return None
    smiles_template_mapping_forward = results[0]['mapped_rxn']
    # print(1111111111111111111, smiles_template_mapping_forward)

    smart_product, smart_reactant = smarts_template.split(">>")
    smile_reactant, smile_product = smiles_template_mapping_forward.split(">>")
    a = main_function(smart_reactant, smile_reactant)
    b = main_function(smart_product, smile_product)
    separator = "."
    react = separator.join(a)
    product = separator.join(b)
    # print("react", react)
    # print("product", product)
    maked_tem = product + ">>" + react
    # print(maked_tem)
    return maked_tem


if __name__ == '__main__':
    rxn = "[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([F:11])([F:12])[F:13])[cH:14][c:15]2[Cl:16])[n:17][c:18]1[CH2:19][OH:20]>>[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([F:11])([F:12])[F:13])[cH:14][c:15]2[Cl:16])[n:17][c:18]1[CH:19]=[O:20]"
    rxn = 'CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7].[CH3:8][C:9](=[O:10])[c:11]1[cH:12][cH:13][c:14]2[nH:15][cH:16][cH:17][c:18]2[cH:19]1>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:15]1[c:14]2[cH:13][cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16]1'
    rxn = "[CH3:1][CH2:2][n:3]1[n:4][n:5][n:6][c:7]1[C:8]([CH:9]=[CH:10][CH:11]=[O:12])=[C:13]([c:14]1[cH:15][cH:16][" \
          "c:17]([F:18])[cH:19][cH:20]1)[c:21]1[cH:22][cH:23][c:24]([F:25])[cH:26][cH:27]1.[CH3:28][CH2:29][O:30][" \
          "C:31](=[O:32])[CH2:33][C:34]([CH3:35])=[O:36]>>[CH3:1][CH2:2][n:3]1[n:4][n:5][n:6][c:7]1[C:8]([CH:9]=[" \
          "CH:10][CH:11]([OH:12])[CH2:35][C:34]([CH2:33][C:31]([O:30][CH2:29][CH3:28])=[O:32])=[O:36])=[C:13]([" \
          "c:14]1[cH:15][cH:16][c:17]([F:18])[cH:19][cH:20]1)[c:21]1[cH:22][cH:23][c:24]([F:25])[cH:26][cH:27]1 "
    print('rxn: ', rxn)
    raw_tmp = get_reaction_template(reaction=rxn, super_general=True, canonical=False)
    print('raw_tmp:', raw_tmp)
    tmp = remove_tmp_mapping(tmp_smarts=raw_tmp, return_smarts=True)
    print('tmpNomapping: ', tmp)
    tmp = RXN_compose_template(tmp)
    print('composed_tmp:', tmp)
    print('check raw_tmp: ', check_run(raw_tmp, rxn))
    print('check composed_tmp:', check_run(tmp, rxn))
