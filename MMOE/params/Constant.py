# -*- codeing = utf-8 -*-
# @Time : 2021/6/20 9:08
# @Author : Evan_wyl
# @File : Constant.py

PCA_DIM = 32

# 初赛带预测行为特征
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 用于构造特征的字段列表
FEAT_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}
# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"online_train": 14, "offline_train": 12, "evaluate": 13, "submit": 15}
# 各种行为统计天数
ACTION_DAY_NUM = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 5, "comment": 5, "follow": 5,
                  "favorite": 5}




Sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']

Dense_features = ['videoplayseconds', "play_userid_mean", "play_userid_max", "play_userid_std"]

Feedid_feat = [
    "read_comment_feedid_mean", "like_feedid_mean", "click_avatar_feedid_mean", "forward_feedid_mean",
    "comment_feedid_mean", "follow_feedid_mean", "favorite_feedid_mean",

    "read_comment_feedid_sum", "like_feedid_sum", "click_avatar_feedid_sum", "forward_feedid_sum",
    "comment_feedid_sum", "follow_feedid_sum", "favorite_feedid_sum",

    "read_comment_feedid_count", "like_feedid_count", "click_avatar_feedid_count", "forward_feedid_count",
    "comment_feedid_count", "follow_feedid_count", "favorite_feedid_count", ]

Userid_feat = [
    "read_comment_userid_mean", "like_userid_mean", "click_avatar_userid_mean", "forward_userid_mean",
    "comment_userid_mean", "follow_userid_mean", "favorite_userid_mean",

    "read_comment_userid_sum", "like_userid_sum", "click_avatar_userid_sum", "forward_userid_sum",
    "comment_userid_sum", "follow_userid_sum", "favorite_userid_sum",

    "read_comment_userid_count", "like_userid_count", "click_avatar_userid_count", "forward_userid_count",
    "comment_userid_count", "follow_userid_count", "favorite_userid_count"
]


Feed_emb = ["feed_embed_" + str(i) for i in range(0, PCA_DIM)]


Userid_authorid_feature = [
     "like_userid_authorid_mean", "click_avatar_userid_authorid_mean", "forward_userid_authorid_mean",
    "comment_userid_authorid_mean", "follow_userid_authorid_mean", "favorite_userid_authorid_mean",

    "read_comment_userid_authorid_sum", "like_userid_authorid_sum", "click_avatar_userid_authorid_sum", "forward_userid_authorid_sum",
    "comment_userid_authorid_sum", "follow_userid_authorid_sum", "favorite_userid_authorid_sum",

    "read_comment_userid_authorid_count", "like_userid_authorid_count", "click_avatar_userid_authorid_count", "forward_userid_authorid_count",
    "comment_userid_authorid_count", "follow_userid_authorid_count", "favorite_userid_authorid_count",
]

Userid_bgm_singer_id_feature = [
    "like_userid_bgm_singer_id_mean", "click_avatar_userid_bgm_singer_id_mean", "forward_userid_bgm_singer_id_mean",
    "comment_userid_bgm_singer_id_mean", "follow_userid_bgm_singer_id_mean", "favorite_userid_bgm_singer_id_mean",

    "read_comment_userid_bgm_singer_id_sum", "like_userid_bgm_singer_id_sum", "click_avatar_userid_bgm_singer_id_sum", "forward_userid_bgm_singer_id_sum",
    "comment_userid_bgm_singer_id_sum", "follow_userid_bgm_singer_id_sum",  "favorite_userid_bgm_singer_id_sum",

    "read_comment_userid_bgm_singer_id_count", "like_userid_bgm_singer_id_count", "click_avatar_userid_bgm_singer_id_count", "forward_userid_bgm_singer_id_count",
    "comment_userid_bgm_singer_id_count", "follow_userid_bgm_singer_id_count",  "favorite_userid_bgm_singer_id_count",
]

Userid_bgm_song_id_feature = [
    "like_userid_bgm_song_id_mean", "click_avatar_userid_bgm_song_id_mean", "forward_userid_bgm_song_id_mean",
    "comment_userid_bgm_song_id_mean", "follow_userid_bgm_song_id_mean", "favorite_userid_bgm_song_id_mean",

    "read_comment_userid_bgm_song_id_sum", "like_userid_bgm_song_id_sum", "click_avatar_userid_bgm_song_id_sum", "forward_userid_bgm_song_id_sum",
    "comment_userid_bgm_song_id_sum", "follow_userid_bgm_song_id_sum", "favorite_userid_bgm_song_id_sum",

    "read_comment_userid_bgm_song_id_count", "like_userid_bgm_song_id_count", "click_avatar_userid_bgm_song_id_count", "forward_userid_bgm_song_id_count",
    "comment_userid_bgm_song_id_count", "follow_userid_bgm_song_id_count", "favorite_userid_bgm_song_id_count",
]

Crossed_feat = Userid_authorid_feature + Userid_bgm_singer_id_feature + Userid_bgm_song_id_feature


