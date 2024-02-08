import os
import sys

sys.path.append('..')
import rdkit.Chem as Chem
from rdchiral.main import rdchiralRunText
from utils.remove_mapping import remove_smiles_mapping_and_get_cano_smiles
from utils.remove_mapping import get_reaction_template


def rdchiralRunText_our(tmp, product, return_mapped=False, keep_mapnums=False):
    p = remove_smiles_mapping_and_get_cano_smiles(product)
    rc, lgs = tmp.split('>>')
    rc = '(' + rc + ')'
    lgs = '(' + lgs + ')'
    tmp = rc + '>>' + lgs
    outcomes = rdchiralRunText(tmp, p, return_mapped=return_mapped, keep_mapnums=keep_mapnums)
    return outcomes


def addisootope(edatom, p):
    mol_p = Chem.MolFromSmiles(p)
    gt = []
    for atom_idx in edatom:
        atom = mol_p.GetAtomWithIdx(atom_idx)
        atom.SetIsotope(atom.GetAtomMapNum())
        gt.append(atom.GetAtomMapNum())
    return Chem.MolToSmiles(mol_p, canonical=False), gt


def rmpiso(sim):
    mol = Chem.MolFromSmiles(sim)
    for atom in mol.GetAtoms():
        if atom.GetIsotope() != 0:
            atom.SetIsotope(0)
    return Chem.MolToSmiles(mol)


def rdchiralRunText_our_final(product, tmp, action_label):
    P_iso, gt = addisootope(action_label, product)
    outcomes = rdchiralRunText_our(tmp, P_iso, return_mapped=True, keep_mapnums=True)
    res = []
    for item in outcomes[0]:
        # print(item)
        rec_smiles, pred = outcomes[1][item]
        rec_mol = Chem.MolFromSmiles(rec_smiles)
        rec_map = set()

        for atom in rec_mol.GetAtoms():
            if atom.GetAtomMapNum() in pred:
                if atom.GetIsotope() != 0:
                    # print(atom.GetAtomMapNum(), atom.GetIsotope())
                    rec_map.add(atom.GetIsotope())

        # print(pred, rec_map, gt, '\n')
        if rec_map == set(gt):
            res.append(remove_smiles_mapping_and_get_cano_smiles(rmpiso(rec_smiles), isomericSmiles=False))
            # return remove_smiles_mapping_and_get_cano_smiles(rmpiso(rec_smiles), isomericSmiles=False)
    return res


def check_run(tmp, rxn):
    r, p = rxn.split(">>")
    p = remove_smiles_mapping_and_get_cano_smiles(p)
    r = remove_smiles_mapping_and_get_cano_smiles(r)
    rst = False
    try:
        outcome = rdchiralRunText_our(tmp, p)
    except:
        print('run的时候报错！！！！！！！！！')
        return False

    if len(outcome) == 0:
        print('outcomes为空！！！！！！！！！')
        return False
    else:
        for string in outcome:
            string = remove_smiles_mapping_and_get_cano_smiles(string)
            if string == r:
                rst = True
                break
    return rst


if __name__ == '__main__':
    rxn = "[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([F:11])([F:12])[F:13])[cH:14][c:15]2[" \
          "Cl:16])[n:17][c:18]1[CH2:19][OH:20]>>[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([" \
          "F:11])([F:12])[F:13])[cH:14][c:15]2[Cl:16])[n:17][c:18]1[CH:19]=[O:20] "
    rxn = 'CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7].[CH3:8][C:9](=[O:10])[c:11]1[cH:12][cH:13][c:14]2[nH:15][cH:16][cH:17][c:18]2[cH:19]1>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:15]1[c:14]2[cH:13][cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16]1'
    r, p = rxn.split(">>")
    tem = get_reaction_template(rxn, super_general=True, canonical=False)
    print('template:', tem)
    action_label = [5, 7]
    outcomes = rdchiralRunText_our_final(p, tem, action_label)
    gt = remove_smiles_mapping_and_get_cano_smiles(r)

    print('ground truth reactants:', gt)
    print('outcomes: ', outcomes)
    for o in outcomes:
        pred = remove_smiles_mapping_and_get_cano_smiles(o)
        print('correct run reactants:', pred, gt == pred)
    print('original run: ', rdchiralRunText_our(tmp=tem, product=p))
