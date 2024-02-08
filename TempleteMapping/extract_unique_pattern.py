import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

import json
import pandas as pd
from tqdm import tqdm
# from utils.get_template import get_reaction_template
from utils.remove_mapping import remove_smarts_mapping
from TempleteMapping.rdchiral.template_extractor import extract_from_reaction, extract_from_reaction_plus


def get_reaction_template(reaction, super_general=False, canonical=True):  # 这里输入的是化学字符串
    reactants, products = reaction.split('>>')

    reaction_dict = {'reactants': reactants,
                     "products": products,
                     '_id': 000}  # 这里是未来转换格式以适应extract_from_reaction需要的格式，然后可以得到template
    template = extract_from_reaction_plus(reaction=reaction_dict, super_general=super_general, canonical=canonical)

    return template


def extract_unique_RC_smartsDict_And_LG_smartsDict(RawDataFile_path, ProcessedDataFile_path):
    df = pd.read_csv(os.path.join(RawDataFile_path, 'raw_train.csv'))
    rc_index_list = []
    lg_index_list = []
    for index, row in tqdm(df.iterrows(), leave=False):
        super_tmp = get_reaction_template(reaction=row[2], super_general=True, canonical=True)
        rc, lgs = super_tmp.split('>>')

        # 处理unique_RC_smartsDict
        rc_without_mapping = remove_smarts_mapping(rc)
        rc_index_list.append(rc_without_mapping)

        # 处理unique_LG_smartsDict
        for lg in lgs.split("."):
            lg_without_mapping = remove_smarts_mapping(lg)
            lg_index_list.append(lg_without_mapping)

    # 处理unique_RC_smartsDict
    rc_unique_list = list(set(rc_index_list))
    len_rc = len(rc_unique_list)
    RC2idx = {}  # {smarts: 0, smarts: 2, ............................}
    idx2RC = {}
    for i, smarts in enumerate(rc_unique_list):
        RC2idx[smarts] = i
        idx2RC[i] = smarts
    with open(os.path.join(ProcessedDataFile_path, 'RC2idx.json'), 'w') as f:
        json.dump(RC2idx, f, indent=4)
    with open(os.path.join(ProcessedDataFile_path, 'idx2RC.json'), 'w') as f:
        json.dump(idx2RC, f, indent=4)

    # 处理unique_LG_smartsDict
    lg_unique_list = list(set(lg_index_list))
    len_lg = len(lg_unique_list)
    LG2idx = {}  # {smarts: 0, smarts: 2, ............................}
    idx2LG = {}
    for i, smarts in enumerate(lg_unique_list):
        LG2idx[smarts] = i
        idx2LG[i] = smarts
    with open(os.path.join(ProcessedDataFile_path, 'LG2idx.json'), 'w') as f:
        json.dump(LG2idx, f, indent=4)
    with open(os.path.join(ProcessedDataFile_path, 'idx2LG.json'), 'w') as f:
        json.dump(idx2LG, f, indent=4)

    print('extract/save rc and lg smarts finish!')
    print('len_rc, len_lg: ', len_rc, len_lg)
    return RC2idx, LG2idx


if __name__ == '__main__':
    RC2idx, LG2idx = extract_unique_RC_smartsDict_And_LG_smartsDict(RawDataFile_path='../data/RawData/USPTO-50k',
                                                                    ProcessedDataFile_path='../data/ProcessedData/USPTO-50k')
    print(RC2idx)
    print(LG2idx)
