from rdkit import Chem
from MCSBasded import MCS_maked
from rxnmapperBasded import RXN_compose_template
from utils.get_template import get_reaction_template
from utils.remove_mapping import remove_tmp_mapping
from utils.run_template import check_run


def check_atom_zeromapping(smi):
    mol = Chem.MolFromSmarts(smi)
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            return False
    return True


def mix_mcs_and_rxn(tmpNoMap):
    try:
        tmp = MCS_maked(tmpNoMap)
    except:
        return RXN_compose_template(tmpNoMap)

    if not check_atom_zeromapping(tmp.split('>>')[0]):
        tmp = RXN_compose_template(tmpNoMap)

    return tmp


def mix_rxn_and_mcs(tmpNoMap):
    try:
        tmp = RXN_compose_template(tmpNoMap)
    except:
        return MCS_maked(tmpNoMap)

    if tmp is None:
        return MCS_maked(tmpNoMap)

    if not check_atom_zeromapping(tmp.split('>>')[0]):
        tmp = MCS_maked(tmpNoMap)

    return tmp


if __name__ == '__main__':
    rxn = "[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([F:11])([F:12])[F:13])[cH:14][c:15]2[Cl:16])[n:17][c:18]1[CH2:19][OH:20]>>[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([F:11])([F:12])[F:13])[cH:14][c:15]2[Cl:16])[n:17][c:18]1[CH:19]=[O:20]"
    # rxn = 'CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7].[CH3:8][C:9](=[O:10])[c:11]1[cH:12][cH:13][c:14]2[nH:15][cH:16][cH:17][c:18]2[cH:19]1>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:15]1[c:14]2[cH:13][cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16]1'
    raw_tmp = get_reaction_template(rxn, super_general=True, canonical=True)
    print('raw tmp:', raw_tmp)
    tmp = remove_tmp_mapping(raw_tmp)
    print('tmpNomapping: ', tmp)
    tmp = mix_rxn_and_mcs(tmp)
    print('composed tmp:', tmp)
    print(check_run(raw_tmp, rxn))
    print(check_run(tmp, rxn))
