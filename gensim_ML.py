# -*- coding: utf-8 -*-
import MeCab # 形態素解析を行うため、Mecabのライブラリ
import gensim # 機械学習のライブラリ
from gensim import corpora,models,similarities
import codecs # 文字コードに関するライブラリ
import glob # ファイル名の取得するのに必要なライブラリ

# Mecabで文章を分かち書きし配列で返す関数
# 引数：text(解析する文章)
# 返り値：なし
def mecabAnalysis(text):
    res_list = []
    tagger = MeCab.Tagger('-Ochasen')
    tagger.parse('') # 空文字列をparseしておかないとエラーになる場合がある
    result = tagger.parseToNode(text)
    while result:
        #type = result.feature.split(",")[0] #品詞を取得
        word = result.surface
        if word != "": # 空でなければリストに追加
            res_list.append(word)
        result = result.next
    return res_list

# StationTweetディレクトリのツイートデータを分かち書きし、辞書とコーパスを作成する
# 引数：なし
# 返り値：なし
def createDictCorpus():
    word_list = [] #全てのツイートに対する単語リスト

    #指定ディレクトリのCSVデータを全て読み込み分かち書きを行う
    dir_name  = "StationTweet/"
    for filename in glob.glob(dir_name + '*.csv'):
        print("ファイル："+filename+"を解析します。")

        # 指定ファイルを読み込む
        file = codecs.open(filename,'r','utf-8')
        lines = file.readlines()
        file.close()

        # 読み混んだファイルを一行づつ処理する
        tweet_word = []
        for line in lines:
            tweet = line.split(",")[0] # ツイートの部分を取得
            res = mecabAnalysis(tweet)
            tweet_word.extend(res)

        #分かち書きした配列をまとめる
        word_list += [tweet_word]

    # 辞書作成
    dictionary = corpora.Dictionary(word_list)
    #dictionary.filter_extremes(no_below=2, no_above=0.3) # no_belowは出現回数が指定以下は削除、no_aboveは全体の指定%以上なら削除
    dictionary.save_as_text('dictionaryData.txt')

    # コーパスを作成
    corpus = [dictionary.doc2bow(text) for text in word_list]
    corpora.MmCorpus.serialize('corpusData.mm', corpus)

# TFIDFを算出する相似辞書を作成する
# 引数：なし
# 返り値：なし
def createTfidfDic():
    # 作成されたコーパスを読み込む
    tfidf_corpus = gensim.corpora.MmCorpus('corpusData.mm')
    # 辞書を作成し保存する
    tfidf_index = gensim.similarities.SparseMatrixSimilarity(tfidf_corpus)
    tfidf_index.save('tweet_tfidf_similarity.index')

# 作成したTFIDF辞書を使って類似度が高い文書を上位30件表示する
# 引数：クエリ
# 返り値：なし
def testTfidf(query):
    # 文書IDと名称を取得
    stationName_list = {}
    dir_name  = "StationTweet/"
    id_count = 0
    for filename in glob.glob(dir_name + '*.csv'):
        #id = filename.split("_")[0].split("/")[1]
        name = filename.split("_")[1].split(".")[0]
        #print(id+":"+name)
        stationName_list[id_count] = name
        id_count += 1

    #辞書とTFIDFデータを読み込む
    dictionary = gensim.corpora.Dictionary.load_from_text('dictionaryData.txt')
    tfidf_index = gensim.similarities.SparseMatrixSimilarity.load('tweet_tfidf_similarity.index')

    #クエリをベクトル空間にマッピングする
    query_vector = dictionary.doc2bow(query.split()) # queryの特徴ベクトル
    #print(query_vector)

    sims = tfidf_index[query_vector]
    # 類似度が高い順に並べ替えて上位10件を表示((文書ID,類似度)で表示)
    result = (sorted(enumerate(sims), key=lambda item: -item[1])[:30])
    #print(sorted(enumerate(sims), key=lambda item: -item[1])[:10])

    print('クエリ:{0}'.format(query))
    for res in result:
        id = res[0]
        name = stationName_list[res[0]]
        value = res[1]
        print('id:{0}, name:{1}, value:{2}'.format(id,name,value))


#ここからメイン部分

createDictCorpus()
createTfidfDic()
testTfidf("遅延")
