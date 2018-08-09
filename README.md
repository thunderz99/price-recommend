# 値段を予測するAIを作ってみた

## 背景

メリカリが値段をAIで査定するニュースがあった。

実際Kaggle上にデータと参考のソースコードがあるため、自分でも作ってみた。

こんな感じ

![画面イメージ](docs/images/price01.png)

デモ

平均の誤差(RMSLE)は0.69で、1400位くらい。
平均の値段の誤差は16ドルくらい。


一位は0.37。


[github ソース]()


## 実現の仕方

前回の[テキスト分類](https://stainless.dreamarts.co.jp/l-zhang/keras_text)を応用した


テキスト分類は、文章はどのカテゴリに該当するかを判断する。

FAQは、ユーザの質問はどの意図かに該当するかを判断して、そして事前用意した標準回答を返す。

### データ収集

SDBのrest apiを利用し、バインダ内の文書を取得


### データ準備

Mecabで文章を単語に分かち書きする

```
採用活動に関する工数はどこにつければいいですか？
↓
採用,活動,工数,どこ,つけれ,いい
```

* 単語のリストをベクトルに変換(word2vec [1],[2])

![word2vecイメージ](./docs/images/keras-faq03.jpg)

### モデル

* 単語ベクトルを、以下のCNN1D(畳み込みネットワーク1D)のモデルにインプット[3]
  * アウトプットはN個の質問

* 訓練済みのモデルを利用しrest apiを作成
* ボットで呼び出す

## 今後の発展

* 社内のFAQを全部ボットに集約？
* SalesWeaponとして売る？

## 参考

[1] <https://www.tensorflow.org/tutorials/word2vec>

[2] <https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/>

[3] <https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py>




