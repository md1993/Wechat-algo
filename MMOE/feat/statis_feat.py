# -*- codeing = utf-8 -*-
# @Time : 2021/6/20 18:30
# @Author : Evan_wyl
# @File : statis_feat.py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from params.Constant import ACTION_LIST, ACTION_SAMPLE_RATE, ACTION_DAY_NUM, STAGE_END_DAY, FEAT_COLUMN_LIST


ROOT_PATH = "../data"
DATASET_PATH = os.path.join(ROOT_PATH, "wechat_algo_data1")
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
END_DAY = 15


def mkdir():
    needs_dir = ["feature", "model", "submit"]
    for ned_dir in needs_dir:
        path = os.path.join(ROOT_PATH, ned_dir)
        if not os.path.exists(path):
            os.mkdir(path)

def statis_feat(start_day=1, before_day=5, agg=None):
    if agg is None:
        agg = ["mean", "sum", "count"]
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEAT_COLUMN_LIST]
    history_data.sort_values(by="date_", inplace=True)
    feature_dir = os.path.join(ROOT_PATH, "feature")
    for dim in ["userid", "feedid"]:
        print(dim)
        user_data = history_data[[dim, "date_"] + FEAT_COLUMN_LIST]
        res_arr = []
        tmp_name = "_" + dim + "_"
        for start in range(2, before_day + 1):
            temp = user_data[user_data["date_"] <= start]
            temp = temp.drop(columns=["date_"])
            temp = temp.groupby([dim]).agg(agg).reset_index()
            temp.columns = [dim] + list(map(tmp_name.join, temp.columns.values[1:]))
            temp["date_"] = start
            res_arr.append(temp)

        for start in range(start_day, END_DAY-before_day+1):
            temp = user_data[(user_data["date_"] >= start) & (user_data["date_"] < (start + before_day))]
            temp = temp.drop("date_", axis=1)
            temp = temp.groupby([dim]).agg(agg).reset_index()
            temp.columns = [dim] + list(map(tmp_name.join, temp.columns.values[1:]))
            temp["date_"] = start + before_day
            res_arr.append(temp)

        dim_feature = pd.concat(res_arr)
        feature_path = os.path.join(feature_dir, dim + "_feature.csv")
        print("Save to: %s " % feature_path)
        dim_feature.to_csv(feature_path, index=False)

if __name__ == '__main__':
    mkdir()
    statis_feat(start_day=1, before_day=5, agg=["mean", "sum", "count"])
