import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../RetroUtils'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../TempleteMapping'))

import json
import tqdm
from rdkit import Chem
import pandas as pd
from RetroUtils.find_subsmarts import find_subsmarts
from CheckUtils import get_reaction_template, remove_smarts_mapping, check_run, calculate_morgan_fingerprint, calculate_morgan_fingerprint_for_LG
from TempleteMapping.final_template_mapping import final_compose_tmp


def extract_atom_mapping(smarts):
    mol = Chem.MolFromSmarts(smarts)
    atom_mapping = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    assert len(atom_mapping) == mol.GetNumAtoms(), 'rc mapping error in function (extract_atom_mapping)'
    atom_mapping.sort()
    return atom_mapping


def find_atoms_by_mapping(smarts, mapping_list):
    mol = Chem.MolFromSmarts(smarts)
    atom_indices = []
    for atom in mol.GetAtoms():
        atom_mapping = atom.GetAtomMapNum()
        if atom_mapping in mapping_list:
            atom_indices.append(atom.GetIdx())
    atom_indices.sort()
    return atom_indices


def check_single_edit(rc_smarts):
    if '.' in rc_smarts:
        return False
    mol = Chem.MolFromSmarts(rc_smarts)
    num_atom = mol.GetNumAtoms()
    if num_atom <= 2:
        return True
    else:
        return False


class process_data:
    def __init__(self, RawDataFile_path, ProcessedDataFile_path):
        self.RawDataFile_path = RawDataFile_path
        self.ProcessedDataFile_path = ProcessedDataFile_path
        self.final_compose_tmp = final_compose_tmp(ProcessedDataFile_path=ProcessedDataFile_path)

        # tmp2freq, RC2idx, LG2idx, idx2LG
        with open(os.path.join(ProcessedDataFile_path, 'tmp2freq.json'), 'r') as f:
            self.tmp2freq = json.load(f)
        with open(os.path.join(ProcessedDataFile_path, 'RC2idx.json'), 'r') as f:
            self.RC2idx = json.load(f)
        with open(os.path.join(ProcessedDataFile_path, 'LG2idx.json'), 'r') as f:
            self.LG2idx = json.load(f)
        with open(os.path.join(ProcessedDataFile_path, 'idx2LG.json'), 'r') as f:
            self.idx2LG = json.load(f)

    def find_compose_template(self, product, rc_node_idx, lg_idx, idx2LG):  # smarts, [atom_idx, atom_idx, ...]
        rc_smarts = find_subsmarts(production=product, action_label=rc_node_idx)
        lg_smarts = []
        for idx in lg_idx:
            lg_smarts.append(idx2LG[str(idx)])
        lg_smarts = '.'.join(lg_smarts)
        tmp = rc_smarts + '>>' + lg_smarts
        tmp = self.final_compose_tmp.compose(tmp)
        return tmp

    def process_one_rxn(self, rxn, tmp2freq, RC2idx, LG2idx, idx2LG):
        reactants, product = rxn.split('>>')  # !
        canonical_template = get_reaction_template(rxn=rxn, super_general=True, canonical=True)  # !
        freq_tmp = tmp2freq[canonical_template]  # !

        # rc_node_idx
        nocanonical_tmp = get_reaction_template(rxn=rxn, super_general=True, canonical=False)
        nocanonical_rc = nocanonical_tmp.split('>>')[0]
        rc_atom_mapping = extract_atom_mapping(nocanonical_rc)  # [rc_mapping, rc_mapping]
        rc_node_idx = find_atoms_by_mapping(smarts=product,
                                            mapping_list=rc_atom_mapping)  # [rc_node_idx, rc_node_idx]  # !

        rc, lgs = canonical_template.split('>>')
        # rc_idx
        rc_idx = [RC2idx[remove_smarts_mapping(rc)]]  # !
        rc_idx.sort()

        # lg_idx
        lg_idx = []
        for lg in lgs.split('.'):
            lg_idx.append(LG2idx[remove_smarts_mapping(lg)])
        lg_idx.sort()

        # raw_check
        raw_check = check_run(tmp=canonical_template, rxn=rxn, relaxation=True)

        # compose_check
        compose_tmp = self.find_compose_template(product=product, rc_node_idx=rc_node_idx, lg_idx=lg_idx, idx2LG=idx2LG)
        compose_check = check_run(tmp=compose_tmp, rxn=rxn, relaxation=True)

        # single_edit
        single_edit = check_single_edit(rc)

        # product_fp
        product_fp = calculate_morgan_fingerprint_for_LG(smarts=product)
        product_fp = product_fp.ToList()

        data = {'rxn': rxn, 'product': product, 'reactants': reactants,
                'template': canonical_template, 'freq_tmp': freq_tmp, 'rc_node_idx': rc_node_idx,
                'rc_idx': rc_idx, 'lg_idx': lg_idx, 'raw_check': raw_check,
                'compose_check': compose_check, 'single_edit': single_edit, 'product_fp': product_fp}

        return data

    def process(self):
        name_list = ['raw_train.csv', 'raw_val.csv', 'raw_test.csv']
        for name in name_list:
            print(f'start process {name} !')
            raw_data_path = os.path.join(self.RawDataFile_path, name)
            df = pd.read_csv(raw_data_path)
            rst = []
            count_raw_check = 0
            count_compose_check = 0
            count_multi_edits = 0
            for index, row in tqdm.tqdm(df.iterrows(), leave=False):
                rxn = row[2]
                canonical_template = get_reaction_template(rxn=rxn, super_general=True, canonical=True)
                rc = canonical_template.split('>>')[0]
                if '.' in rc:
                    continue
                try:
                    one_data = self.process_one_rxn(rxn, self.tmp2freq, self.RC2idx, self.LG2idx, self.idx2LG)
                except:
                    continue

                if one_data['raw_check']:
                    count_raw_check += 1
                if one_data['compose_check']:
                    count_compose_check += 1
                if not one_data['single_edit']:
                    count_multi_edits += 1

                rst.append(one_data)
            with open(os.path.join(self.ProcessedDataFile_path, name.replace('.csv', '.json')), 'w') as f:
                json.dump(rst, f, indent=4)
            print(f'end/save {name} !')
            print(
                f'{name} info:  num_rst: {len(rst)}, count_raw_check: {count_raw_check}, count_compose_check: {count_compose_check}, ration: {count_compose_check / count_raw_check}, count_multi_edits: {count_multi_edits}')


