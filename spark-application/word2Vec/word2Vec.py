from pyspark.mllib.feature import Word2Vec
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName('word2Vec')
sc = SparkContext(conf=conf)
inp = sc.textFile("/data/mllib/sample_lda_data.txt").map(lambda row: row.split(" "))

word2vec = Word2Vec()
model = word2vec.fit(inp)

synonyms = model.findSynonyms('1', 5)

for word, cosine_distance in synonyms:
    print("{}: {}".format(word, cosine_distance))

