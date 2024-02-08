import json
import os
import tqdm
import numpy as np
from CheckUtils import remove_smarts_mapping, calculate_morgan_fingerprint_for_LG


def create_hypergraph(ProcessedDataFile_path):
    with open(os.path.join(ProcessedDataFile_path, 'tmp2freq.json'), 'r') as f:
        tmp2freq = json.load(f)
    with open(os.path.join(ProcessedDataFile_path, 'LG2idx.json'), 'r') as f:
        LG2idx = json.load(f)

    # create hyper edge
    num_lg = []
    e_list = []
    count_error = 0
    for tmp in tqdm.tqdm(tmp2freq):
        reaction_center, leaving_groups = tmp.split('>>')
        # create one hyper edge
        try:
            one_hyper_edge = []
            for lg in leaving_groups.split('.'):
                lg = remove_smarts_mapping(lg)
                one_hyper_edge.append(LG2idx[lg])
            one_hyper_edge.sort()
            one_hyper_edge = tuple(one_hyper_edge)
            if one_hyper_edge not in e_list:
                e_list.append(one_hyper_edge)
            num_lg.append(len(one_hyper_edge))
        except:
            count_error += 1
            continue
    e_list = list(set(e_list))  # list of tuple

    # num_v
    num_v = len(LG2idx)  # int

    # v_fp
    v_fp = []
    for lg in LG2idx:
        # print(lg)
        fp = calculate_morgan_fingerprint_for_LG(lg)
        fp = fp.ToList()
        v_fp.append(fp)

    assert len(v_fp) == num_v, 'num_fp is less num_v !'

    data = {'num_v': num_v, 'e_list': e_list, 'v_fp': v_fp}
    with open(os.path.join(ProcessedDataFile_path, 'hypergraph.json'), 'w') as f:
        json.dump(data, f, indent=4)

    print('create_hypergraph finished!', flush=True)
    print('num_v: ', num_v, '| num_edge: ', len(e_list), '| mean_num_lg_per_tmp: ', np.mean(num_lg), '| count_error: ',
          count_error)

    return data


