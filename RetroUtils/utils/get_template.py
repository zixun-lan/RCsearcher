import os
import sys

sys.path.append('..')
from rdchiral.template_extractor import extract_from_reaction_plus


def get_reaction_template(smi, super_general=False, canonical=True):  # 这里输入的是化学字符串
    reactants, products = smi.split('>>')

    reaction_dict = {'reactants': reactants,
                     "products": products,
                     '_id': 000}  # 这里是未来转换格式以适应extract_from_reaction需要的格式，然后可以得到template
    template = extract_from_reaction_plus(reaction_dict, super_general=super_general, canonical=canonical)

    return template
    # print(template['reaction_smarts'])可以拆解开来看看template到底是什么


if __name__ == '__main__':
    rxn = "[Br:1][c:2]1[cH:3][c:4]([C:9]([F:10])([F:11])[F:12])[c:5]([OH:8])[n:6][cH:7]1.I[CH2:20][CH3:21]>>[Br:1][" \
          "c:2]1[cH:3][c:4]([C:9]([F:10])([F:11])[F:12])[c:5](=[O:8])[n:6]([CH2:20][CH3:21])[cH:7]1 "
    tmp = get_reaction_template(rxn, super_general=True, canonical=False)
    print(tmp)
