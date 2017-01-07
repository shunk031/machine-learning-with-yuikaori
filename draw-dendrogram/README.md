# Draw dendrogram using YUIKAORI blog

ゆいかおりの小倉唯さん・石原夏織さんのブログデータを用いてデンドログラムを描画する。

### `src/dendrogram.py`

`project root/preprocess/preprocessed_data/blog-articles`から前処理済みのデータ`wakati-tokens.csv`を読み込み、分かち書きされたブログ本文についてTF-IDFを計算したものを使い、文書間のコサイン類似度を計算する。計算結果を用いて文書間の距離を可視化(2D/3D)したものと、デンドログラムを出力する。
