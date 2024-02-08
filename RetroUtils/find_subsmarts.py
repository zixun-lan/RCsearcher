import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

from rdkit import Chem
from utils.other_utils import products_process, remove_smarts_fragment_mapping
from utils.remove_mapping import remove_smarts_mapping
from utils.get_template import get_reaction_template
from rdchiral.template_extractor import get_fragments_for_changed_atoms


def find_subsmarts(production, action_label):
    production_mol = Chem.MolFromSmiles(production)

    edits_label = []
    for i in action_label:
        edits_label.append(str(production_mol.GetAtomWithIdx(i).GetAtomMapNum()))

    products = products_process(production)
    product_fragments, _, _ = get_fragments_for_changed_atoms(products, edits_label, radius=0,
                                                              expansion=[], category='products')
    pred = remove_smarts_fragment_mapping(product_fragments[1:-1])
    return pred


if __name__ == '__main__':
    rxn = "[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([F:11])([F:12])[F:13])[cH:14][c:15]2[" \
          "Cl:16])[n:17][c:18]1[CH2:19][OH:20]>>[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([" \
          "F:11])([F:12])[F:13])[cH:14][c:15]2[Cl:16])[n:17][c:18]1[CH:19]=[O:20] "
    r, p = rxn.split(">>")
    tem = get_reaction_template(rxn, super_general=True, canonical=False)
    print('template:', tem)  # mapping: 19, 20
    action_label = [18, 19]
    sub_smarts = find_subsmarts(p, action_label)
    rc = tem.split('>>')[0]
    print('sub_smarts: ', sub_smarts)
    print('rc: ', remove_smarts_mapping(rc))