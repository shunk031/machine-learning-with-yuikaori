# Classify YUIKAORI blog

ゆいかおりの小倉唯さん・石原夏織さんのブログを分類する。

## 各スクリプトについて

### `src/make_single_file.py`

[Ameblo Crawler](https://github.com/shunk031/ameblo-crawler)によって収集した2人のブログデータを`/data/blog-article`に格納しておく。スクリプトを実行すると、格納した2人のデータがそれぞれ`ogurayui-blog.pkl`・`ishiharakaori-blog.pkl`として、`/data/blog-articles`に1つのpickleファイルとしてシリアライズされる。
  
### `src/wakati.py`
  
シリアライズされたpickleファイルから2人のブログ本文データを読み込み、分かち書きしたものを`「"クラスラベル","分かち書きしたブログ本文"」`としてCSVファイルに出力される。

### `src/grid_search.py`

クラスラベルと分かち書きしたブログ本文のデータを読み込み、TF-IDFの値を計算する。学習器として用いるSVMのパラメータグリッドサーチを用いてチューニングを行う。TF-IDFのモデルとSVMモデルをシリアライズされる。
	
### `src/predict.py`

grid_search.pyで学習したパラメータを用いて未知のデータに対して予測を行う。
