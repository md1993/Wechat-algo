# -*- codeing = utf-8 -*-
# @Time : 2021/6/20 19:24
# @Author : Evan_wyl
# @File : user_play_statis.py

import pandas as pd
import os

before_day = 5
start_day = 1
END_DAY = 15

feature_dir = os.path.join("../data", "feature")
user_data = pd.read_csv("../data/wechat_algo_data1/user_action.csv")
user_data = user_data[["userid", "date_", "play"]]
def user_play_statis(agg=None):
    if agg is None:
        agg = ["mean", "max", "std"]
    res_arr = []
    tmp_name = "_" + "userid" + "_"
    for start in range(2, before_day + 1):
        temp = user_data[user_data["date_"] <= start]
        temp = temp.drop(columns=["date_"])
        temp = temp.groupby(["userid"]).agg(agg).reset_index()
        temp.columns = ["userid"] + list(map(tmp_name.join, temp.columns.values[1:]))
        temp["date_"] = start
        res_arr.append(temp)

    for start in range(start_day, END_DAY - before_day + 1):
        temp = user_data[(user_data["date_"] >= start) & (user_data["date_"] < (start + before_day))]
        temp = temp.drop("date_", axis=1)
        temp = temp.groupby(["userid"]).agg(agg).reset_index()
        temp.columns = ["userid"] + list(map(tmp_name.join, temp.columns.values[1:]))
        temp["date_"] = start + before_day
        res_arr.append(temp)

    dim_feature = pd.concat(res_arr)
    feature_path = os.path.join(feature_dir, "userid_play" + "_feature.csv")
    print("Save to: %s " % feature_path)
    dim_feature.to_csv(feature_path, index=False)

if __name__ == '__main__':
    user_play_statis()

