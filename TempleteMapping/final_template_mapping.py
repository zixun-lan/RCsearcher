import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

import json
# from mix_two_nobased_corpus import mix_rxn_and_mcs, mix_mcs_and_rxn
from MCSBasded import MCS_maked
from mapping_based_corpus import compose_tmp_by_corpus
from utils.remove_mapping import remove_tmp_mapping
from utils.get_template import get_reaction_template
from utils.run_template import check_run


# def final_tmp_mapping1(tmpNoMapping, RC_json_dict, LG_json_dict, idx2tmp_dict):
#     try:
#         tmp = mapping_template_based_corpus(tmpNoMapping, RC_json_dict=RC_json_dict,
#                                             LG_json_dict=LG_json_dict,
#                                             idx2tmp_dict=idx2tmp_dict)
#     except:
#         return mix_rxn_and_mcs(tmpNoMapping)
#
#     if tmp is None:
#         return mix_rxn_and_mcs(tmpNoMapping)
#
#     return tmp
#
#
# def final_tmp_mapping2(tmpNoMapping):
#     try:
#         tmp = mapping_template_based_corpus(tmpNoMapping)
#     except:
#         return mix_mcs_and_rxn(tmpNoMapping)
#
#     if tmp is None:
#         return mix_mcs_and_rxn(tmpNoMapping)
#
#     return tmp


class final_compose_tmp:
    def __init__(self, ProcessedDataFile_path):
        self.compose_tmp_by_corpus = compose_tmp_by_corpus(ProcessedDataFile_path=ProcessedDataFile_path)

    def compose(self, tmp_NoMapping):
        try:
            tmp = self.compose_tmp_by_corpus.compose_tmp(tmp_NoMapping)
        except:
            tmp = MCS_maked(tmp_NoMapping)

        if tmp is None:
            return MCS_maked(tmp_NoMapping)

        return tmp


if __name__ == '__main__':
    tmp_composer = final_compose_tmp(ProcessedDataFile_path='../data/ProcessedData/USPTO-50k')
    # rxn = "CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7].[O:8]=[CH:9][c:10]1[cH:11][nH:12][cH:13][n:14]1>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:12]1[cH:11][c:10]([CH:9]=[O:8])[n:14][cH:13]1"
    # rxn = "[CH3:1][CH2:2][CH2:3][C@:4]([CH2:5][CH2:6][OH:7])([CH2:8][CH2:9][c:10]1[cH:11][cH:12][cH:13][cH:14][cH:15]1)[O:16][CH2:17][O:18][c:19]1[cH:20][cH:21][c:22](-[c:23]2[cH:24][cH:25][cH:26][cH:27][cH:28]2)[cH:29][cH:30]1>>[CH3:1][CH2:2][CH2:3][C@:4]([CH2:5][CH:6]=[O:7])([CH2:8][CH2:9][c:10]1[cH:11][cH:12][cH:13][cH:14][cH:15]1)[O:16][CH2:17][O:18][c:19]1[cH:20][cH:21][c:22](-[c:23]2[cH:24][cH:25][cH:26][cH:27][cH:28]2)[cH:29][cH:30]1"
    # rxn = "O[C:1](=[O:2])[c:3]1[n:4][c:5](-[c:6]2[cH:7][cH:8][cH:9][cH:10][cH:11]2)[cH:12][s:13]1.[CH3:14][O:15][C:16](=[O:17])[C@@H:18]([NH2:19])[CH2:20][c:21]1[cH:22][cH:23][cH:24][cH:25][cH:26]1>>[C:1](=[O:2])([c:3]1[n:4][c:5](-[c:6]2[cH:7][cH:8][cH:9][cH:10][cH:11]2)[cH:12][s:13]1)[NH:19][CH:18]([C:16]([O:15][CH3:14])=[O:17])[CH2:20][c:21]1[cH:22][cH:23][cH:24][cH:25][cH:26]1"
    # rxn = "[CH3:1][O:2][c:3]1[cH:4][c:5]([CH2:6][OH:7])[cH:8][c:9]([O:10][CH3:11])[c:12]1[CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][CH2:18][CH2:19][CH2:20][CH2:21][CH2:22][CH2:23][CH2:24][CH2:25][CH2:26][O:27][Si:28]([CH3:29])([CH3:30])[C:31]([CH3:32])([CH3:33])[CH3:34]>>[CH3:1][O:2][c:3]1[cH:4][c:5]([CH:6]=[O:7])[cH:8][c:9]([O:10][CH3:11])[c:12]1[CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][CH2:18][CH2:19][CH2:20][CH2:21][CH2:22][CH2:23][CH2:24][CH2:25][CH2:26][O:27][Si:28]([CH3:29])([CH3:30])[C:31]([CH3:32])([CH3:33])[CH3:34]"
    # rxn  = "[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([F:11])([F:12])[F:13])[cH:14][c:15]2[Cl:16])[n:17][c:18]1[CH2:19][OH:20]>>[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([F:11])([F:12])[F:13])[cH:14][c:15]2[Cl:16])[n:17][c:18]1[CH:19]=[O:20]"
    # rxn = "[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([F:11])([F:12])[F:13])[cH:14][c:15]2[Cl:16])[n:17][c:18]1[CH2:19][OH:20]>>[CH3:1][c:2]1[n:3][n:4](-[c:5]2[c:6]([Cl:7])[cH:8][c:9]([C:10]([F:11])([F:12])[F:13])[cH:14][c:15]2[Cl:16])[n:17][c:18]1[CH:19]=[O:20]"
    # rxn = "[CH3:1][N:2]([CH2:3][c:4]1[cH:5][cH:6][c:7]([CH2:8][OH:9])[c:10]([F:11])[cH:12]1)[C:13](=[O:14])[O:15][C:16]([CH3:17])([CH3:18])[CH3:19]>>[CH3:1][N:2]([CH2:3][c:4]1[cH:5][cH:6][c:7]([CH:8]=[O:9])[c:10]([F:11])[cH:12]1)[C:13](=[O:14])[O:15][C:16]([CH3:17])([CH3:18])[CH3:19]"
    # rxn = "[OH:1][CH2:2]/[CH:3]=[CH:4]/[c:5]1[cH:6][c:7]([F:8])[cH:9][c:10]([F:11])[cH:12]1>>[O:1]=[CH:2]/[CH:3]=[CH:4]/[c:5]1[cH:6][c:7]([F:8])[cH:9][c:10]([F:11])[cH:12]1"
    # rxn = "O=[C:1]([CH2:2]Cl)[c:3]1[cH:4][cH:5][cH:6][c:7]([C:8]([F:9])([F:10])[F:11])[c:12]1[F:13].[NH2:14][C:15]([NH2:16])=[S:17]>>[c:1]1(-[c:3]2[cH:4][cH:5][cH:6][c:7]([C:8]([F:9])([F:10])[F:11])[c:12]2[F:13])[cH:2][s:17][c:15]([NH2:16])[n:14]1"
    rxn = 'O[C:1]([CH2:2][C:3]#[N:4])=[O:5].[CH3:6][CH:7]([CH3:8])[n:9]1[cH:10][cH:11][c:12]([NH:13][c:14]2[cH:15][c:16]([NH:17][C@H:18]3[CH2:19][CH2:20][C@H:21]([NH2:22])[CH2:23][CH2:24]3)[n:25][n:26]3[c:27]([C:28](=[O:29])[NH:30][c:31]4[cH:32][cH:33][n:34][cH:35][c:36]4[F:37])[cH:38][n:39][c:40]23)[n:41]1>>[C:1]([CH2:2][C:3]#[N:4])(=[O:5])[NH:22][C@H:21]1[CH2:20][CH2:19][C@H:18]([NH:17][c:16]2[cH:15][c:14]([NH:13][c:12]3[cH:11][cH:10][n:9]([CH:7]([CH3:6])[CH3:8])[n:41]3)[c:40]3[n:26]([n:25]2)[c:27]([C:28](=[O:29])[NH:30][c:31]2[cH:32][cH:33][n:34][cH:35][c:36]2[F:37])[cH:38][n:39]3)[CH2:24][CH2:23]1'
    # rxn = 'c1ccc(C[O:1][C:2]([CH:3]2[N:4]([CH2:5][CH3:6])[CH2:7][CH2:8][CH2:9]2)=[O:10])cc1>>[OH:1][C:2]([C@H:3]1[N:4]([CH2:5][CH3:6])[CH2:7][CH2:8][CH2:9]1)=[O:10]'
    # rxn = "[CH3:1][Si:2]([CH3:3])([CH3:4])[CH2:5][CH2:6][S:7](=[O:8])(=[O:9])[N:10]1[CH2:11][CH2:12][CH2:13][CH:14]([CH2:15][OH:16])[CH2:17]1>>[CH3:1][Si:2]([CH3:3])([CH3:4])[CH2:5][CH2:6][S:7](=[O:8])(=[O:9])[N:10]1[CH2:11][CH2:12][CH2:13][CH:14]([CH:15]=[O:16])[CH2:17]1"
    # rxn = 'CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7].[CH3:8][C:9](=[O:10])[c:11]1[cH:12][cH:13][c:14]2[nH:15][cH:16][cH:17][c:18]2[cH:19]1>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:15]1[c:14]2[cH:13][cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16]1'
    raw_tmp = get_reaction_template(rxn, super_general=True, canonical=True)
    print('rxn: ', rxn)
    print('raw tmp:', raw_tmp)
    tmp = remove_tmp_mapping(raw_tmp)
    print('tmpNomapping: ', tmp)
    tmp = tmp_composer.compose(tmp)
    print('composed tmp:', tmp)
    print('check raw_tmp: ', check_run(raw_tmp, rxn))
    print('check composed_tmp:', check_run(tmp, rxn))

























