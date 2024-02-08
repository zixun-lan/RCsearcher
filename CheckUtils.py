from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral.main import rdchiralRunText
from rdchiral.template_extractor import extract_from_reaction_plus


def get_reaction_template(rxn, super_general=False, canonical=True):  # 这里输入的是化学字符串
    reactants, products = rxn.split('>>')

    reaction_dict = {'reactants': reactants,
                     "products": products,
                     '_id': 000}  # 这里是未来转换格式以适应extract_from_reaction需要的格式，然后可以得到template
    template = extract_from_reaction_plus(reaction=reaction_dict, super_general=super_general, canonical=canonical)

    return template


def remove_smarts_mapping(smarts):
    mol = Chem.MolFromSmarts(smarts)
    # 去除原子的mapping
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # 确保原子的写法唯一
    m = Chem.MolToSmarts(mol)
    return m


def remove_smiles_mapping_and_get_cano_smiles(smiles, normalization=True):
    # 将 SMILES 转化为分子对象
    mol = Chem.MolFromSmiles(smiles)

    # 去除映射
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')

    cano_smiles = Chem.MolToSmiles(mol, canonical=True)

    return cano_smiles


def rdchiralRunText_our(tmp, product, return_mapped=False, keep_mapnums=False):
    product = remove_smiles_mapping_and_get_cano_smiles(product)
    reaction_center, leaving_groups = tmp.split('>>')
    reaction_center = '(' + reaction_center + ')'
    leaving_groups = '(' + leaving_groups + ')'
    tmp = reaction_center + '>>' + leaving_groups
    outcomes = rdchiralRunText(tmp, product, return_mapped=return_mapped, keep_mapnums=keep_mapnums)
    return outcomes


def check_run(tmp, rxn):
    reactants, product = rxn.split(">>")
    product = remove_smiles_mapping_and_get_cano_smiles(product)
    reactants = remove_smiles_mapping_and_get_cano_smiles(reactants)
    rst = False
    try:
        outcome = rdchiralRunText_our(tmp, product)
    except:
        # print('In function (check_run): run的时候报错！！！！！！！！！')
        return False

    if len(outcome) == 0:
        # print('In function (check_run): outcomes为空！！！！！！！！！')
        return False
    else:
        for string in outcome:
            string = remove_smiles_mapping_and_get_cano_smiles(string)
            if string == reactants:
                rst = True
                break
    return rst


def calculate_morgan_fingerprint(smarts, nBits=2048):
    mol = Chem.MolFromSmarts(smarts)

    Chem.SanitizeMol(mol)

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    return fingerprint


def set_bonds_to_single(smarts, atom_index):
    mol = Chem.MolFromSmarts(smarts)

    # 获取指定索引的原子
    atom = mol.GetAtomWithIdx(atom_index)

    # 获取与原子相连的化学键
    bonds = atom.GetBonds()

    # 将与原子相连的化学键设置为单键
    for bond in bonds:
        bond.SetBondType(Chem.BondType.SINGLE)

    # 将修正后的分子转换为 SMARTS 表达式
    modified_smarts = Chem.MolToSmiles(mol)

    return modified_smarts


def set_atoms_to_NoAromatic(smarts, atom_index):
    mol = Chem.MolFromSmarts(smarts)

    # 获取指定索引的原子
    atom = mol.GetAtomWithIdx(atom_index)

    # 获取与原子相连的化学键
    atom.SetIsAromatic(False)

    return mol


def calculate_morgan_fingerprint_for_LG(smarts, nBits=2048):
    done = True
    mol = Chem.MolFromSmarts(smarts)
    while done:
        try:
            Chem.SanitizeMol(mol)
            done = False
        except Chem.MolSanitizeException as e:
            # Explicit valence for atom # 8 N, 4, is greater than permitted
            # non-ring atom 0 marked aromatic
            error_str = str(e)
            if error_str.split()[0] == 'Explicit':
                atom_idx = error_str.split()[error_str.split().index('#') + 1]
                smarts = set_bonds_to_single(smarts=smarts, atom_index=int(atom_idx))
                mol = Chem.MolFromSmarts(smarts)
            elif error_str.split()[0] == 'non-ring':
                atom_idx = error_str.split()[error_str.split().index('atom') + 1]
                mol = set_atoms_to_NoAromatic(smarts=smarts, atom_index=int(atom_idx))
                # mol = Chem.MolFromSmarts(smarts)
            # print(smarts)

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    return fingerprint


