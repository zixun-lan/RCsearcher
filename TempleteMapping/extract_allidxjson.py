import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

import json
import pandas as pd
from tqdm import tqdm
from TempleteMapping.utils.get_template import get_reaction_template
from utils.remove_mapping import remove_smarts_mapping
from extract_unique_pattern import extract_unique_RC_smartsDict_And_LG_smartsDict


def extract_key_listOFlist(tmp, RC_json_dict, LG_json_dict):
    RC_idx_list = []
    LG_idx_list = []
    rc, lgs = tmp.split('>>')

    # 构建RC_idx_list
    rc_without_mapping = remove_smarts_mapping(rc)
    RC_idx_list.append(RC_json_dict[rc_without_mapping])
    RC_idx_list.sort()

    # 构建LG_idx_list
    for lg in lgs.split("."):
        lg_without_mapping = remove_smarts_mapping(lg)
        LG_idx_list.append(lg_without_mapping)
    LG_idx_list = [LG_json_dict[key] for key in LG_idx_list]
    LG_idx_list.sort()

    final_idx_list = [RC_idx_list, LG_idx_list]
    return final_idx_list


def extract_all(RawDataFile_path, ProcessedDataFile_path):
    df = pd.read_csv(os.path.join(RawDataFile_path, 'raw_train.csv'))
    idx2tmp = {}
    tmp2idx = {}
    tmp2freq = {}

    # 打开RC和LG独一无二字典
    RC2idx, LG2idx = extract_unique_RC_smartsDict_And_LG_smartsDict(RawDataFile_path, ProcessedDataFile_path)

    # 获取list-of-list作为key
    for index, row in tqdm(df.iterrows(), leave=False):
        template = get_reaction_template(reaction=row[2], super_general=True, canonical=True)
        # frequency template
        if template in tmp2freq:
            tmp2freq[template] += 1
        else:
            tmp2freq[template] = 1

        try:
            key_listOFlist = extract_key_listOFlist(tmp=template, RC_json_dict=RC2idx, LG_json_dict=LG2idx)
        except:
            print('pattern索取不到！')
            continue
        idx2tmp[str(key_listOFlist)] = template
        tmp2idx[template] = key_listOFlist
    with open(os.path.join(ProcessedDataFile_path, 'idx2tmp.json'), 'w') as f:
        json.dump(idx2tmp, f, indent=4)
    with open(os.path.join(ProcessedDataFile_path, 'tmp2idx.json'), 'w') as f:
        json.dump(tmp2idx, f, indent=4)

    with open(os.path.join(ProcessedDataFile_path, 'tmp2freq.json'), 'w') as f:
        json.dump(tmp2freq, f, indent=4)

    print('extract/save all idx finish!')
    print('len: ', len(idx2tmp))


if __name__ == '__main__':
    extract_all(RawDataFile_path='../data/RawData/USPTO-50k',
                ProcessedDataFile_path='../data/ProcessedData/USPTO-50k')
