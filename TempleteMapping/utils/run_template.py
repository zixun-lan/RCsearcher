import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rdchiral.main import rdchiralRunText
from remove_mapping import remove_smiles_mapping_and_get_cano_smiles
from remove_mapping import get_reaction_template


def rdchiralRunText_our(tmp, product):
    product = remove_smiles_mapping_and_get_cano_smiles(product)
    reaction_center, leaving_groups = tmp.split('>>')
    reaction_center = '(' + reaction_center + ')'
    leaving_groups = '(' + leaving_groups + ')'
    tmp = reaction_center + '>>' + leaving_groups
    outcomes = rdchiralRunText(tmp, product)
    return outcomes


def check_run(tmp, rxn):
    reactants, product = rxn.split(">>")
    product = remove_smiles_mapping_and_get_cano_smiles(product)
    reactants = remove_smiles_mapping_and_get_cano_smiles(reactants)
    rst = False
    try:
        outcome = rdchiralRunText_our(tmp, product)
    except:
        print('In function (check_run): run的时候报错！！！！！！！！！')
        return False

    if len(outcome) == 0:
        print('In function (check_run): outcomes为空！！！！！！！！！')
        return False
    else:
        for string in outcome:
            string = remove_smiles_mapping_and_get_cano_smiles(string)
            if string == reactants:
                rst = True
                break
    return rst


if __name__ == '__main__':
    rxn = "[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([F:11])([F:12])[F:13])[cH:14][c:15]2[" \
          "Cl:16])[n:17][c:18]1[CH2:19][OH:20]>>[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([" \
          "F:11])([F:12])[F:13])[cH:14][c:15]2[Cl:16])[n:17][c:18]1[CH:19]=[O:20] "
    reactants, product = rxn.split(">>")
    tmp = get_reaction_template(reaction=rxn, super_general=True, canonical=True)
    outcomes = rdchiralRunText_our(tmp, product)
    print('rxn: ', rxn)
    print('tmp: ', tmp)
    print('outcomes: ', outcomes)
    print('canonical_reactants: ', remove_smiles_mapping_and_get_cano_smiles(reactants))
    print('check_run: ', check_run(tmp, rxn))
