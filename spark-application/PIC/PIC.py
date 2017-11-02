from pyspark.mllib.clustering import PowerIterationClustering, PowerIterationClusteringModel
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName('PIC')
sc = SparkContext(conf=conf)

# Load and parse the data
data = sc.textFile("/data/mllib/pic_data.txt")
similarities = data.map(lambda line: tuple([float(x) for x in line.split(' ')]))

# Cluster the data into two classes using PowerIterationClustering
model = PowerIterationClustering.train(similarities, 2, 10)

#model.assignments().foreach(lambda x: print(str(x.id) + " -> " + str(x.cluster)))

