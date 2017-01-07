# -*- coding: utf-8 -*-

import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram

PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), '../../preprocess/preprocessed_data/blog-articles')
SAMPLE_NUM = 5

if __name__ == '__main__':

    # データの読み込み
    df = pd.read_csv(os.path.join(PREPROCESSED_DIR, 'wakati-tokens.csv'))

    yui_data = df[df['classlabel'] == 'yui']
    kaori_data = df[df['classlabel'] == 'kaori']

    # 2人のデータからそれぞれSAMPLE_NUM分だけ抽出して新しいDataFrameを作成
    yui_data = yui_data.loc[random.sample(list(yui_data.index), SAMPLE_NUM)]
    kaori_data = kaori_data.loc[random.sample(list(kaori_data.index), SAMPLE_NUM)]
    df = pd.concat([yui_data, kaori_data])

    # 分かち書きされた文書からTF-IDFを計算
    X = df['article'].values
    vectorizer = TfidfVectorizer()
    dtm = vectorizer.fit_transform(X)
    vocab = vectorizer.get_feature_names()

    # scipy matrix型からnumpy型に変換
    dtm = dtm.toarray()
    vocab = np.array(vocab)

    # 図に表示するときの表示名を設定
    blog_classlabel = df['classlabel'].values
    blog_title = df['title'].values
    names = ['{}:{}'.format(label, title) for label, title in zip(blog_classlabel, blog_title)]

    # 文書間のコサイン類似度を計算する
    dist = 1 - cosine_similarity(dtm)

    # 計算したコサイン類似度を用いて文書間の距離を可視化する
    # 2次元で可視化してみる
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
    pos = mds.fit_transform(dist)

    xs, ys = pos[:, 0], pos[:, 1]
    for x, y, name in zip(xs, ys, names):
        plt.scatter(x, y)
        plt.text(x, y, name)
    plt.savefig('visualizing_distances2D.png')

    # 3次元で可視化してみる
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=1)
    pos = mds.fit_transform(dist)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    for x, y, z, s in zip(pos[:, 0], pos[:, 1], pos[:, 2], names):
        ax.text(x, y, z, s)
    plt.savefig('visualizing_distances3D.png')

    # デンドログラムを描画する
    plt.figure(figsize=(8, 5))
    linkage_matrix = ward(dist)
    dendrogram(linkage_matrix, orientation='left', labels=names)
    plt.tight_layout()
    plt.savefig('dendrogram.png')
