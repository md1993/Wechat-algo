# -*- codeing = utf-8 -*-
# @Time : 2021/6/20 18:44
# @Author : Evan_wyl
# @File : concat_feat.py

import numpy as np
import pandas as pd
import os
import gc
import sys
sys.path.append('..')

from params.Constant import Feed_emb, Feedid_feat, Userid_feat, Dense_features, Sparse_features, Crossed_feat
from params.Constant import FEAT_COLUMN_LIST



def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


def train_merge_feat(stage="train"):
    if stage == "train":
        user_info = pd.read_csv("../data/wechat_algo_data1/user_action.csv")
    else:
        user_info = pd.read_csv("../data/wechat_algo_data1/test_a.csv")
        user_info["date_"] = [15 for i in range(user_info.shape[0])]

    feed_info = pd.read_csv("../data/wechat_algo_data1/feed_info.csv")
    userid_play_feature=  pd.read_csv("../data/feature/userid_play_feature.csv")
    userid_feature = pd.read_csv("../data/feature/userid_feature.csv")
    feeid_feature = pd.read_csv("../data/feature/feedid_feature.csv")
    feed_embeddings_PCA = pd.read_csv("../data/feed_embeddings_PCA.csv")

    userid_authorid_feature = pd.read_csv("../data/feature/userid_authorid_feature.csv")
    userid_bgm_singer_id_feature = pd.read_csv("../data/feature/userid_bgm_singer_id_feature.csv")
    userid_bgm_song_id_feature = pd.read_csv("../data/feature/userid_bgm_song_id_feature.csv")

    user_data = user_info[["userid", "date_", "feedid"]]
    feed_data = feed_info[["feedid", "authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]]
    all_data = user_data.merge(feed_data, on=["feedid"], how="left")

    all_data["bgm_song_id"] += 1
    all_data["bgm_singer_id"] += 1
    all_data.fillna(0, inplace=True)

    all_data["userid"] = all_data["userid"].astype(int)
    all_data["feedid"] = all_data["feedid"].astype(int)
    all_data["authorid"] = all_data["authorid"].astype(int)
    all_data["bgm_song_id"] = all_data["bgm_song_id"].astype(int)
    all_data["bgm_singer_id"] = all_data["bgm_singer_id"].astype(int)

    all_data = all_data.merge(userid_play_feature, on=["userid", "date_"], how="left")
    all_data = all_data.merge(userid_feature, on=["userid", "date_"], how="left")
    all_data = all_data.merge(feeid_feature, on=["feedid", "date_"], how="left")
    all_data = all_data.merge(feed_embeddings_PCA, on=["feedid"], how="left")
    all_data = all_data.merge(userid_authorid_feature, on=["userid", "authorid", "date_"], how="left")
    all_data = all_data.merge(userid_bgm_singer_id_feature, on=["userid", "bgm_singer_id", "date_"], how="left")
    all_data = all_data.merge(userid_bgm_song_id_feature, on=["userid", "bgm_song_id", "date_"], how="left")

    if stage == "train":
        all_data = all_data.merge(user_info[["userid", "feedid", "date_"] + FEAT_COLUMN_LIST], on=["userid", "feedid", "date_"], how="left")

    all_data.fillna(0, inplace=True)
    return all_data



def main():
    print("Train Merge_Feat........")
    df = train_merge_feat()
    print(df.shape)
    reduce_mem(df)
    df.to_csv("../data/training_data.csv", index=False)
    print("Train df Saved.....")
    print("Test Merge......")
    test_df = train_merge_feat(stage="test")
    print(test_df.shape)
    reduce_mem(test_df)
    test_df.to_csv("../data/testing_data.csv", index=False)
    print("Test df Saved.....")


if __name__ == '__main__':
    main()

