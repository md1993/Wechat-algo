import os
import pandas as pd
import numpy as np
import tensorflow as tf

from time import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from mmoe import MMOE
from evaluation import evaluate_deepctr
from params.Constant import ACTION_LIST, ACTION_DAY_NUM, ACTION_SAMPLE_RATE, FEAT_COLUMN_LIST
from params.Constant import *

# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


if __name__ == '__main__':
    epochs = 2
    batch_size = 512
    embedding_dim = 16
    target = ACTION_LIST
    sparse_features = Sparse_features
    dense_features = Dense_features + Feedid_feat + Userid_feat + Feed_emb + Userid_authorid_feature + \
                     Userid_bgm_singer_id_feature + Userid_bgm_song_id_feature

    lens = len(dense_features)
    print("dense_features lens:", lens)

    feat = sparse_features + dense_features + target + ["date_"]
    data = pd.read_csv("../data/training_data.csv")
    data = data[feat]

    test = pd.read_csv("../data/testing_data.csv")

    data[dense_features] = data[dense_features].fillna(0, )
    test[dense_features] = test[dense_features].fillna(0, )

    # data["mean_subtract_feed"] = np.abs(data["mean_subtract_feed"])
    # data["max_subtract_feed"] = np.abs(data["max_subtract_feed"])

    data[dense_features] = np.log(data[dense_features] + 1.0)
    test[dense_features] = np.log(test[dense_features] + 1.0)
    print('data.shape', data.shape)
    print('data.columns', data.columns.tolist())
    print('unique date_: ', data['date_'].unique())

    train = data[(data['date_'] < 14) & (data['date_'] > 1)]
    val = data[data['date_'] == 14]  # 第14天样本作为验证集
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
                              for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

    # 3.generate input data for model
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    userid_list = val['userid'].astype(str).tolist()
    test_model_input = {name: test[name] for name in feature_names}

    train_labels = [train[y].values for y in target]
    val_labels = [val[y].values for y in target]

    # 4.Define Model,train,predict and evaluate
    train_model = MMOE(dnn_feature_columns, num_tasks=4, expert_dim=8, dnn_hidden_units=(128, 128),
                       tasks=['binary', 'binary', 'binary', 'binary'])
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    train_model.compile(optimizer, loss='binary_crossentropy', metrics=["acc"])

    # print(train_model.summary())
    for epoch in range(epochs):
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1)

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
        evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)

    t1 = time()
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
    t2 = time()
    print('4个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
    ts = (t2 - t1) * 1000.0 / len(test) * 2000.0
    print('4个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)

    # 5.生成提交文件
    for i, action in enumerate(target):
        test[action] = pred_ans[i]
    test[['userid', 'feedid'] + target].to_csv('result.csv', index=None, float_format='%.6f')
    print('to_csv ok')


