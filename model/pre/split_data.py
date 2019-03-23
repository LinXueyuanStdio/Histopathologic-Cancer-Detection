import pandas as pd
import numpy as np
import os


def return_tumor_or_not(dic, one_id):
    return dic[one_id]


def create_dict(train_label_path):
    df = pd.read_csv(train_label_path)
    result_dict = {}
    for index in range(df.shape[0]):
        one_id = df.iloc[index, 0]
        tumor_or_not = df.iloc[index, 1]
        result_dict[one_id] = int(tumor_or_not)
    return result_dict


def find_missing(train_ids, cv_ids, train_label_path):
    all_ids = set(pd.read_csv(train_label_path)['id'].values)
    wsi_ids = set(train_ids + cv_ids)

    missing_ids = list(all_ids-wsi_ids)
    return missing_ids


def generate_split(train_label_path, wsi_path):
    ids = pd.read_csv(wsi_path)
    wsi_dict = {}
    for i in range(ids.shape[0]):
        wsi = ids.iloc[i, 1]
        train_id = ids.iloc[i, 0]
        wsi_array = wsi.split('_')
        number = int(wsi_array[3])
        if wsi_dict.get(number) is None:
            wsi_dict[number] = [train_id]
        else:
            wsi_dict[number].append(train_id)

    wsi_keys = list(wsi_dict.keys())
    np.random.seed()
    np.random.shuffle(wsi_keys)
    amount_of_keys = len(wsi_keys)

    keys_for_train = wsi_keys[0:int(amount_of_keys*0.8)]
    keys_for_cv = wsi_keys[int(amount_of_keys*0.8):]
    train_ids = []
    cv_ids = []

    for key in keys_for_train:
        train_ids += wsi_dict[key]

    for key in keys_for_cv:
        cv_ids += wsi_dict[key]

    dic = create_dict(train_label_path)

    missing_ids = find_missing(train_ids, cv_ids, train_label_path)
    missing_ids_total = len(missing_ids)
    np.random.seed()
    np.random.shuffle(missing_ids)

    train_missing_ids = missing_ids[0:int(missing_ids_total*0.8)]
    cv_missing_ids = missing_ids[int(missing_ids_total*0.8):]

    train_ids += train_missing_ids
    cv_ids += cv_missing_ids

    train_labels = []
    cv_labels = []

    train_tumor = 0
    for one_id in train_ids:
        temp = return_tumor_or_not(dic, one_id)
        train_tumor += temp
        train_labels.append(temp)

    cv_tumor = 0
    for one_id in cv_ids:
        temp = return_tumor_or_not(dic, one_id)
        cv_tumor += temp
        cv_labels.append(temp)
    total = len(train_ids) + len(cv_ids)

    print("Amount of train labels: {}, {}/{}".format(len(train_ids), train_tumor, len(train_ids)-train_tumor))
    print("Amount of cv labels: {}, {}/{}".format(len(cv_ids), cv_tumor, len(cv_ids) - cv_tumor))
    print("Percentage of cv labels: {}".format(len(cv_ids)/total))

    return train_ids, cv_ids, train_labels, cv_labels


if __name__ == "__main__":
    data_dir = '/mnt/d/project/AI.Health.kaggle/'
    label_dir = '../../data/'
    train_path = data_dir+'train/'
    test_path = data_dir+'test/'
    train_label_path = label_dir+'full_train_labels.csv'
    wsi_path = label_dir+'patch_id_wsi.csv'
    train_ids, cv_ids, train_labels, cv_labels = generate_split(train_label_path, wsi_path)

    print(len(train_ids), len(cv_ids), len(train_labels), len(cv_labels))
    print(train_ids[:5], cv_ids[:5], train_labels[:5], cv_labels[:5])
'''
Amount of train labels: 175144, 72019/103125
Amount of cv labels: 44881, 17098/27783
Percentage of cv labels: 0.2039813657538916
175144 44881 175144 44881
['4faf2b15c22cc437fb16fb11b2f59f2a44afb88c',
 '67b241cf6dfb3e486bea7ca652500e59783eb124',
 '0e0a49f00652b915dd6afc661c9bee4eb7f75723',
 'a912e404d73397e65f7bfd4630f2a13ad800fe47',
 'ba5ba4d9622b91f44a3f1bca709c7a7344b5bf78']
['a0e35f55df27bfe6a43894780333f202108da95c',
 '781a505340288d069b646d63eb0897528cecf752',
 'efcf1f070efa0100968773c522af474f561cee40',
 '381e7a49d67485b2cb084b6e5134a71443d44b5d',
 'a08962087ec1477dbec7cdf48a2d13c5d0070cbd']
[0, 1, 0, 1, 0]
[0, 0, 0, 0, 0]
'''
