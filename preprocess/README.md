# Preprocess of data

## 各スクリプトについて

### `make_blog_data_to_single_file.py`

[Ameblo Crawler](https://github.com/shunk031/ameblo-crawler)によって収集した2人のブログデータを`project root/row_data/blog-articles`に格納しておく。スクリプトを実行すると、格納した2人のデータがそれぞれ`ogurayui-blog.pkl`・`ishiharakaori-blog.pkl`として、`project root/preprocess/preprocessed_data/blog-articles`に1つのpickleファイルとしてシリアライズされる。

### `wakati.py`
  
シリアライズされたpickleファイルから2人のブログ本文データを読み込み、分かち書きしたものを`「"ブログ記事投稿日", "ブログ記事タイトル", ["分かち書きしたブログ本文"], "クラスラベル"」`としてCSVファイルに出力される。
