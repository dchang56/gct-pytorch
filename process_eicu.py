import pickle
import csv
import os
import sys
import numpy as np
import sklearn.model_selection as ms
import torch
import time
from torch.utils.data import TensorDataset
from tqdm import tqdm


class EncounterInfo:
    def __init__(self, patient_id, encounter_id, 
                 encounter_timestamp, expired, readmission):
        self.patient_id = patient_id
        self.encounter_id = encounter_id
        self.encounter_timestamp = encounter_timestamp
        self.expired = expired
        self.readmission = readmission
        self.dx_ids = []
        self.rx_ids = []
        self.labs = {}
        # self.physicals = []
        self.treatments = []

class EncounterFeatures:
    def __init__(self, patient_id, label_expired, label_readmission, dx_ids, dx_ints, proc_ids, proc_ints):
        self.patient_id = patient_id
        self.label_expired = label_expired
        self.label_readmission = label_readmission
        self.dx_ids = dx_ids
        self.dx_ints = dx_ints
        self.proc_ids = proc_ids
        self.proc_ints = proc_ints
        self.prior_indices = None
        self.prior_values = None
        self.dx_mask = None
        self.proc_mask = None
        

def process_patient(infile, encounter_dict, hour_threshold=24):
    with open(infile, 'r') as f:
        patient_dict = {}
        for line in csv.DictReader(f):
            patient_id = line['patienthealthsystemstayid']
            encounter_id = line['patientunitstayid']
            encounter_timestamp = -int(line['hospitaladmitoffset'])
            if patient_id not in patient_dict:
                patient_dict[patient_id] = []
            patient_dict[patient_id].append((encounter_timestamp, encounter_id))
    
    patient_dict_sorted = {}
    for patient_id, time_enc_tuples in patient_dict.items():
        patient_dict_sorted[patient_id] = sorted(time_enc_tuples)
    
    enc_readmission_dict = {}
    for patient_id, time_enc_tuples in patient_dict_sorted.items():
        for time_enc_tuple in time_enc_tuples[:-1]:
            enc_id = time_enc_tuple[1]
            enc_readmission_dict[enc_id] = True
        last_enc_id = time_enc_tuples[-1][1]
        enc_readmission_dict[last_enc_id] = False
    
    with open(infile, 'r') as f:
        count = 0
        for line in tqdm(csv.DictReader(f)):

            patient_id = line['patienthealthsystemstayid']
            encounter_id = line['patientunitstayid']
            encounter_timestamp = line['hospitaladmitoffset']
            discharge_status = line['unitdischargestatus']
            duration_minute = float(line['unitdischargeoffset'])
            expired = True if discharge_status=='Expired' else False
            readmission = enc_readmission_dict[encounter_id]
            if duration_minute > 60. * hour_threshold:
                continue
            ei = EncounterInfo(patient_id, encounter_id, encounter_timestamp, expired, readmission)
            if encounter_id in encounter_dict:
                print('duplicate encounter id! skip')
                sys.exit(0)
            encounter_dict[encounter_id] = ei
            count += 1
    return encounter_dict

def process_admission_dx(infile, encounter_dict):
    with open(infile, 'r') as f:
        count = 0
        missing_eid = 0
        for line in tqdm(csv.DictReader(f)):
            encounter_id = line['patientunitstayid']
            dx_id = line['admitdxpath'].lower()
            
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue
            encounter_dict[encounter_id].dx_ids.append(dx_id)
            count += 1
    print('')
    print('Admission Diagnosis without encounter id: {}'.format(missing_eid))
    return encounter_dict

def process_diagnosis(infile, encounter_dict):
    with open(infile, 'r') as f:
        count = 0
        missing_eid = 0
        for line in tqdm(csv.DictReader(f)):
            encounter_id = line['patientunitstayid']
            dx_id = line['diagnosisstring'].lower()
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue
            encounter_dict[encounter_id].dx_ids.append(dx_id)
            count += 1
    print('Diagnosis without encounter id: {}'.format(missing_eid))
    return encounter_dict

def process_treatment(infile, encounter_dict):
    with open(infile, 'r') as f:
        count = 0
        missing_eid = 0
        for line in tqdm(csv.DictReader(f)):
            encounter_id = line['patientunitstayid']
            treatment_id = line['treatmentstring'].lower()
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue
            encounter_dict[encounter_id].treatments.append(treatment_id)
            count += 1
    print('Treatment without encounter id: {}'.format(missing_eid))
    print('accepted treatment: {}'.format(count))
    return encounter_dict


def get_encounter_features(encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=50):
    """
    In the original tf implementation, dx_ints and proc_ints are serialized as variable length sequences,
    which are converted to SparseTensors when retrieved and converted to dense tensors when the lookup method
    is called to retrieve the embeddings, where max_num_codes and vocab_sizes are used to shape the tensors.
    
    Instead, here I explicitly store them with the proper shape, and skip the reshaping step in embedding lookup.
    
    """
    key_list = []
    enc_features_list = []
    dx_str2int = {}
    treat_str2int = {}
    num_cut = 0
    num_duplicate = 0
    count = 0
    num_dx_ids = 0
    num_treatments = 0
    num_unique_dx_ids = 0
    num_unique_treatments = 0
    min_dx_cut = 0
    min_treatment_cut = 0
    max_dx_cut = 0
    max_treatment_cut = 0
    num_expired = 0
    num_readmission = 0
    

    
    for _, enc in encounter_dict.items():
        if skip_duplicate:
            if (len(enc.dx_ids) > len(set(enc.dx_ids)) or len(enc.treatments) > len(set(enc.treatments))):
                num_duplicate += 1
                continue
        if len(set(enc.dx_ids)) < min_num_codes:
            min_dx_cut += 1
            continue
        if len(set(enc.treatments)) < min_num_codes:
            min_treatment_cut += 1
            continue
        if len(set(enc.dx_ids)) > max_num_codes:
            max_dx_cut += 1
            continue
        if len(set(enc.treatments)) > max_num_codes:
            max_treatment_cut += 1
            continue
        
        count += 1
        num_dx_ids += len(enc.dx_ids)
        num_treatments += len(enc.treatments)
        num_unique_dx_ids += len(set(enc.dx_ids))
        num_unique_treatments += len(set(enc.treatments))

        for dx_id in enc.dx_ids:
            if dx_id not in dx_str2int:
                dx_str2int[dx_id] = len(dx_str2int)
        for treat_id in enc.treatments:
            if treat_id not in treat_str2int:
                treat_str2int[treat_id] = len(treat_str2int)
        
        patient_id = enc.patient_id + ':' + enc.encounter_id
        if enc.expired:
            label_expired = 1
            num_expired += 1
        else:
            label_expired = 0
        if enc.readmission:
            label_readmission = 1
            num_readmission += 1
        else:
            label_readmission = 0
        
        dx_ids = sorted(list(set(enc.dx_ids)))
        dx_ints = [dx_str2int[item] for item in dx_ids]
        proc_ids = sorted(list(set(enc.treatments)))
        proc_ints = [treat_str2int[item] for item in proc_ids]
        
        
        enc_features = EncounterFeatures(patient_id, label_expired, label_readmission, dx_ids, dx_ints, proc_ids, proc_ints)
        
        key_list.append(patient_id)
        enc_features_list.append(enc_features)
    
    
    for ef in enc_features_list:
        dx_padding_idx = len(dx_str2int)
        proc_padding_idx = len(treat_str2int)
        if len(ef.dx_ints) < max_num_codes:
            ef.dx_ints.extend([dx_padding_idx]*(max_num_codes-len(ef.dx_ints)))
        if len(ef.proc_ints) < max_num_codes:
            ef.proc_ints.extend([proc_padding_idx]*(max_num_codes-len(ef.proc_ints)))
        ef.dx_mask = [0 if i==dx_padding_idx else 1 for i in ef.dx_ints]
        ef.proc_mask = [0 if i==proc_padding_idx else 1 for i in ef.proc_ints]

        

    print('Filtered encounters due to duplicate codes: %d' % num_duplicate)
    print('Filtered encounters due to thresholding: %d' % num_cut)
    print('Average num_dx_ids: %f' % (num_dx_ids / count))
    print('Average num_treatments: %f' % (num_treatments / count))
    print('Average num_unique_dx_ids: %f' % (num_unique_dx_ids / count))
    print('Average num_unique_treatments: %f' % (num_unique_treatments / count))
    print('Min dx cut: %d' % min_dx_cut)
    print('Min treatment cut: %d' % min_treatment_cut)
    print('Max dx cut: %d' % max_dx_cut)
    print('Max treatment cut: %d' % max_treatment_cut)
    print('Number of expired: %d' % num_expired)
    print('Number of readmission: %d' % num_readmission)

    return key_list, enc_features_list, dx_str2int, treat_str2int



def select_train_valid_test(key_list, random_seed=1234):
    key_train, key_temp = ms.train_test_split(key_list, test_size=0.2, random_state=random_seed)
    key_valid, key_test = ms.train_test_split(key_temp, test_size=0.5, random_state=random_seed)
    # print('@@@@@@@@@@@@@@@@@@@@')
    # print('keys: {}'.format(key_valid[:5]))
    return key_train, key_valid, key_test


def count_conditional_prob_dp(enc_features_list, output_path, train_key_set=None):
    dx_freqs = {}
    proc_freqs = {}
    dp_freqs = {}
    total_visit = 0
    for enc_feature in enc_features_list:
        key = enc_feature.patient_id
        if (train_key_set is not None and key not in train_key_set):
            total_visit += 1
            continue
        dx_ids = enc_feature.dx_ids
        proc_ids = enc_feature.proc_ids
        for dx in dx_ids:
            if dx not in dx_freqs:
                dx_freqs[dx] = 0
            dx_freqs[dx] += 1
        for proc in proc_ids:
            if proc not in proc_freqs:
                proc_freqs[proc] = 0
            proc_freqs[proc] += 1
        for dx in dx_ids:
            for proc in proc_ids:
                dp = dx + ',' + proc
                if dp not in dp_freqs:
                    dp_freqs[dp] = 0
                dp_freqs[dp] += 1
        total_visit += 1
    
    dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.items()
                ])
    proc_probs = dict([
    (k, v / float(total_visit)) for k, v in proc_freqs.items()
    ])
    dp_probs = dict([(k, v / float(total_visit)) for k, v in dp_freqs.items()
                ])
    
    dp_cond_probs = {}
    pd_cond_probs = {}
    for dx, dx_prob in dx_probs.items():
        for proc, proc_prob in proc_probs.items():
            dp = dx + ',' + proc
            pd = proc + ',' + dx
            if dp in dp_probs:
                dp_cond_probs[dp] = dp_probs[dp] / dx_prob
                pd_cond_probs[pd] = dp_probs[dp] / proc_prob
            else:
                dp_cond_probs[dp] = 0.0
                pd_cond_probs[pd] = 0.0
    #originall supposed to pickle. but for now just return the 2 cond prob dicts that are used
    # return dp_cond_probs, pd_cond_probs
    pickle.dump(dp_cond_probs, open(os.path.join(output_path, 'dp_cond_probs.empirical.p'), 'wb'))
    pickle.dump(pd_cond_probs, open(os.path.join(output_path, 'pd_cond_probs.empirical.p'), 'wb'))


def add_sparse_prior_guide_dp(enc_features_list, stats_path, key_set=None, max_num_codes=50):
    dp_cond_probs = pickle.load(open(os.path.join(stats_path, 'dp_cond_probs.empirical.p'), 'rb'))
    pd_cond_probs = pickle.load(open(os.path.join(stats_path, 'pd_cond_probs.empirical.p'), 'rb'))

    print('Adding prior guide')
    total_visit = 0
    new_enc_features_list = []
    # prior_guide_list = []
    for enc_features in enc_features_list:
        key = enc_features.patient_id
        if (key_set is not None and key not in key_set):
            total_visit += 1
            continue
        dx_ids = enc_features.dx_ids
        proc_ids = enc_features.proc_ids
        indices = []
        values = []
        for i, dx in enumerate(dx_ids):
            for j, proc in enumerate(proc_ids):
                dp = dx + ',' + proc
                indices.append((i, max_num_codes+j))
                prob = 0.0 if dp not in dp_cond_probs else dp_cond_probs[dp]
                values.append(prob)
        for i, proc in enumerate(proc_ids):
            for j, dx in enumerate(dx_ids):
                pd = proc + ',' + dx
                indices.append((max_num_codes+i, j))
                prob = 0.0 if pd not in pd_cond_probs else pd_cond_probs[pd]
                values.append(prob)
        # indices = list(np.array(indices).reshape([-1]))
        
        enc_features.prior_indices = indices
        enc_features.prior_values = values
        new_enc_features_list.append(enc_features)
    

        total_visit += 1
    return new_enc_features_list
        
def convert_features_to_tensors(enc_features):
    # all_patient_ids = torch.tensor([f.patient_id for f in enc_features], dtype=torch.long)
    all_readmission_labels = torch.tensor([f.label_readmission for f in enc_features], dtype=torch.long)
    all_expired_labels = torch.tensor([f.label_expired for f in enc_features], dtype=torch.long)
    all_dx_ints = torch.tensor([f.dx_ints for f in enc_features], dtype=torch.long)
    all_proc_ints = torch.tensor([f.proc_ints for f in enc_features], dtype=torch.long)
    # all_prior_indices = torch.tensor([f.prior_indices for f in enc_features], dtype=torch.long)
    # all_prior_values = torch.tensor([f.prior_values for f in enc_features], dtype=torch.float)
    all_dx_masks = torch.tensor([f.dx_mask for f in enc_features], dtype=torch.float)
    all_proc_masks = torch.tensor([f.proc_mask for f in enc_features], dtype=torch.float)
    dataset = TensorDataset(all_dx_ints, all_proc_ints, all_dx_masks, all_proc_masks, all_readmission_labels, all_expired_labels)
    
    
    return dataset


def get_prior_guide(enc_features):

    prior_guide_list = []
    for feats in enc_features:
        indices = torch.tensor(list(zip(*feats.prior_indices))).reshape(2, -1)
        values = torch.tensor(feats.prior_values)
        prior_guide_list.append((indices, values))
    return prior_guide_list
        
    

def get_datasets(data_dir, fold=0):
    #instead of generating 5 folds manually prior to training using 2 separate scripts, let's generate 1 fold in same script
    patient_file = os.path.join(data_dir, 'patient.csv')
    admission_dx_file = os.path.join(data_dir, 'admissionDx.csv')
    diagnosis_file = os.path.join(data_dir, 'diagnosis.csv')
    treatment_file = os.path.join(data_dir, 'treatment.csv')

    fold_path = os.path.join(data_dir, 'fold_{}'.format(fold))
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    stats_path = os.path.join(fold_path, 'train_stats')
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
    cached_path = os.path.join(fold_path, 'cached')
    if os.path.exists(cached_path):
        start = time.time()
        train_dataset = torch.load(os.path.join(cached_path, 'train_dataset.pt'))
        validation_dataset = torch.load(os.path.join(cached_path, 'valid_dataset.pt'))
        test_dataset = torch.load(os.path.join(cached_path, 'test_dataset.pt'))

        train_prior_guide = torch.load(os.path.join(cached_path, 'train_priors.pt'))
        validation_prior_guide = torch.load(os.path.join(cached_path, 'valid_priors.pt'))
        test_prior_guide = torch.load(os.path.join(cached_path, 'test_priors.pt'))
        
    else:
        os.makedirs(cached_path)
        encounter_dict = {}
        print('Processing patient.csv')
        encounter_dict = process_patient(patient_file, encounter_dict, hour_threshold=24)
        print('Processing admission diagnosis.csv')
        encounter_dict = process_admission_dx(admission_dx_file, encounter_dict)
        print('Processing diagnosis.csv')
        encounter_dict = process_diagnosis(diagnosis_file, encounter_dict)
        print('Processing treatment.csv')
        encounter_dict = process_treatment(treatment_file, encounter_dict)
        
        key_list, enc_features_list, dx_map, proc_map = get_encounter_features(encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=50)
        pickle.dump(dx_map, open(os.path.join(fold_path, 'dx_map.p'), 'wb'))
        pickle.dump(proc_map, open(os.path.join(fold_path, 'proc_map.p'), 'wb'))
        
        key_train, key_valid, key_test = select_train_valid_test(key_list, random_seed=fold)
        count_conditional_prob_dp(enc_features_list, stats_path, set(key_train))
        train_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path, set(key_train), max_num_codes=50)
        validation_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path, set(key_valid), max_num_codes=50)
        test_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path, set(key_test), max_num_codes=50)
        
        train_dataset = convert_features_to_tensors(train_enc_features)
        validation_dataset = convert_features_to_tensors(validation_enc_features)
        test_dataset = convert_features_to_tensors(test_enc_features)
        
        torch.save(train_dataset, os.path.join(cached_path, 'train_dataset.pt'))
        torch.save(validation_dataset, os.path.join(cached_path, 'valid_dataset.pt'))
        torch.save(test_dataset, os.path.join(cached_path, 'test_dataset.pt'))
        
        ## get prior_indices and prior_values for each split and save as list of tensors
        train_prior_guide = get_prior_guide(train_enc_features)
        validation_prior_guide = get_prior_guide(validation_enc_features)
        test_prior_guide = get_prior_guide(test_enc_features)
        
        #save the prior_indices and prior_values
        torch.save(train_prior_guide, os.path.join(cached_path, 'train_priors.pt'))
        torch.save(validation_prior_guide, os.path.join(cached_path, 'valid_priors.pt'))
        torch.save(test_prior_guide, os.path.join(cached_path, 'test_priors.pt'))
        
    
    
    return ([train_dataset, validation_dataset, test_dataset], [train_prior_guide, validation_prior_guide, test_prior_guide])
        


        
        
        
        
        
        
        
    
        
        
        
    
    

    
    