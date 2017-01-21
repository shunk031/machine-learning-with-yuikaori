# -*- coding: utf-8 -*-

import os
import random
import pickle
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), '../../preprocess/preprocessed_data/blog-articles/')

if __name__ == '__main__':

    # ブログデータの読み込み
    df = pd.read_csv(os.path.join(PREPROCESSED_DIR, 'wakati-tokens.csv'))
    kaori_data = df[df['classlabel'] == 'kaori']
    num_of_kaori_data = len(kaori_data)
    yui_data = df[df['classlabel'] == 'yui']
    yui_data = yui_data.iloc[random.sample(list(yui_data.index), num_of_kaori_data)]

    df = pd.concat([yui_data, kaori_data])

    # クラスラベル'yui'と'kaori'を整数値に変更
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)

    # 学習データを読み込む
    X = df['article'].values

    # # 学習データを用いてTF-IDFを計算する
    tfidf_vec = TfidfVectorizer()

    # パラメータチューニングをしたSVMモデルを読み込む
    svm_est = joblib.load("gridsearch_svm.pkl")

    # 5 fold cross validation を実行
    skf = StratifiedKFold(y, 5)
    accuracy = 0
    for i, (train_idx, test_idx) in enumerate(skf):

        X_train_tfidf = tfidf_vec.fit_transform(X[train_idx])
        X_test_tfidf = tfidf_vec.transform(X[test_idx])

        # # 読み込んだモデルでトレーニング
        svm_est.fit(X_train_tfidf, y[train_idx])

        # # 予測を行う
        y_pred = svm_est.predict(X_test_tfidf)

        # 結果を表示
        print("{} th Classification report".format(i))
        print("{}".format(classification_report(y[test_idx], y_pred)))

        # Confusion Matrixを表示する
        print("{} th Confusion Matrix".format(i))
        print("{}\n".format(confusion_matrix(y[test_idx], y_pred)))

        accuracy += (y_pred == y[test_idx]).sum() / y_pred.size

    # 平均精度を出力する
    print("Average Accuracy: {:.2f} [%]".format((accuracy / 5) * 100))
