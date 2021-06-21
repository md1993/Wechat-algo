# -*- codeing = utf-8 -*-
# @Time : 2021/6/20 18:43
# @Author : Evan_wyl
# @File : crossed_feat.py

import pandas as pd
import numpy as np
import os
from params.Constant import FEAT_COLUMN_LIST, ACTION_LIST, ACTION_SAMPLE_RATE, ACTION_DAY_NUM, STAGE_END_DAY

before_day = 5
start_day = 1
end_day = 15

def get_crossed_feat(agg=None):
    if agg is None:
        agg = ["mean", "sum", "count"]
    user_action = pd.read_csv("../data/wechat_algo_data1/user_action.csv")
    feed_info = pd.read_csv("../data/wechat_algo_data1/feed_info.csv")

    user_data = user_action[["userid", "date_", "feedid"] + FEAT_COLUMN_LIST]
    feed_data = feed_info[["feedid", "authorid", "bgm_song_id", "bgm_singer_id"]]
    user_data = user_data.merge(feed_data, on=["feedid"], how="left")

    user_data["bgm_song_id"] += 1
    user_data["bgm_singer_id"] += 1
    user_data["authorid"] += 1
    user_data.fillna(0, inplace=True)

    for dim in ["authorid", "bgm_song_id", "bgm_singer_id"]:
        tmp_name = "_" + "userid_" + dim + "_"
        res_arr = []
        for start in range(2, before_day+1):
            temp = user_data[[dim] + ["userid", "date_"] + FEAT_COLUMN_LIST]
            temp = temp[(temp["date_"] <= start)]
            temp = temp.drop(columns=["date_"])
            temp = temp.groupby(["userid", dim]).agg(agg).reset_index(drop=True)
            temp.columns = [dim] + list(map(tmp_name.join, temp.columns.values[1:]))
            temp["date_"] = start
            res_arr.append(temp)

        for start in range(start_day, end_day - before_day + 1):
            temp = user_data[[dim] + ["userid", "date_"] + FEAT_COLUMN_LIST]
            temp = temp[(user_data["date_"] >= start) & (user_data["date_"] < start + before_day)]
            temp = temp.drop(["date_"], axis=1)
            temp = temp.groupby(["userid", dim]).agg(agg).reset_index(drop=True)
            temp.columns = ["userid"] + list(map(tmp_name.join, temp.columns.values[1:]))
            temp["date_"] = start + before_day
            res_arr.append(temp)

        dim_feature = pd.concat(res_arr)
        feature_path = os.path.join("../data", "feature", "userid_" + dim + "_feature.csv")
        print("Save to: %s " % feature_path)
        dim_feature.to_csv(feature_path, index=False)



if __name__ == '__main__':
    print("Geting.........")
    get_crossed_feat()
    print("Ending.........")

