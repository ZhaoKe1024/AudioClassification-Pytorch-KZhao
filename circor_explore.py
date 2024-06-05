# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-06-05 21:35
import numpy as np
import pandas as pd


def read_txt():
    pass


def read_tsv(filepath):
    mat = []
    with open(filepath, 'r') as fin:
        line = fin.readline()
        while line:
            parts = line.split('\t')
            mat.append([float(parts[0]), float(parts[1]), int(parts[2])])
            line = fin.readline()
    return mat


def read_hea(filepath):
    # mat = []
    with open(filepath, 'r') as fin:
        line = fin.readline()
        while line:
            print(line)
            # parts = line.split('\t')
            # mat.append([float(parts[0]), float(parts[1]), int(parts[2])])
            line = fin.readline()
    # return mat


def read_segments():
    pass


# tem_files("2530", "AV+PV+TV+MV"))
# print(get_item_files("29378", "AV+MV"))
# print(get_im_files("46532", "AV"))
def get_item_files(p_id: str, locs: str):
    exts = ["hea", "tsv", "wav"]
    flist = []
    if '+' in locs:
        parts = locs.split('+')
        for part in parts:
            flist.append(p_id + '_' + part)
    res_list = []

    # for ext in exts:
    #     if len(flist) == 0:
    #         res_list.append(p_id + '_' + locs  + '.' + ext)
    #     else:
    #         for fname in flist:
    #             res_list.append(p_id + '_' + fname + '.' + ext)

    if len(flist) == 0:
        for ext in exts:
            res_list.append(p_id + '_' + locs + '.' + ext)
    else:
        for fname in flist:
            for ext in exts:
                res_list.append(fname + '.' + ext)
    return res_list


def read_data():
    label2num = {"Normal": 0, "Abnormal": 1}
    data_root = "D:/DATAS/Medical/circor/"
    data_dir = "training_data/"
    metafile = "training_data.csv"
    metadata = pd.read_csv(data_root + metafile, header=0, index_col=None, usecols=[0, 1, 20])
    metadata["Outcome"] = metadata["Outcome"].apply(lambda x: label2num[x])
    # print(metadata)
    for j, row in metadata.iterrows():
        # print(row[0], row[1])
        label = row[2]
        cur_files = get_item_files(str(row[0]), row[1])
        print(cur_files[0], label)


if __name__ == '__main__':
    # read_data()
    data_root = "D:/DATAS/Medical/circor/"
    data_dir = "training_data/"
    print(read_hea(data_root + data_dir + "9979_AV.hea"))
