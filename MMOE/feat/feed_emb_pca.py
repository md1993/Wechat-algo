# -*- codeing = utf-8 -*-
# @Time : 2021/6/20 15:18
# @Author : Evan_wyl
# @File : feed_emb_pca.py

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from params.Constant import PCA_DIM

feed_emb = pd.read_csv("../data/wechat_algo_data1/feed_embeddings.csv")
feed_embedd = []
cnt = 0
for i in feed_emb["feed_embedding"].values:
    cnt += 1
    feed_embedd.append([float(ii) for ii in i.split(' ') if ii!=''])

pca = PCA(n_components=PCA_DIM)
feed_embedd=pca.fit_transform(np.array(feed_embedd))
feed_embedding=pd.concat((feed_emb,pd.DataFrame(feed_embedd)),axis=1)
feed_embedding.drop(['feed_embedding'],axis=1,inplace=True)
feed_embedding.columns=['feedid']+['feed_embed_'+str(i) for i in range(PCA_DIM)]
feed_embedding.to_csv('../data/feed_embeddings_PCA.csv',index=False)

