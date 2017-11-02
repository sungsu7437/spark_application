from numpy import array
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName('bisectingKmeans')
sc = SparkContext(conf=conf)

from pyspark.mllib.clustering import BisectingKMeans, BisectingKMeansModel

# Load and parse the data
data = sc.textFile("/data/mllib/kmeans_data.txt")
parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

# Build the model (cluster the data)
model = BisectingKMeans.train(parsedData, 2, maxIterations=5)

# Evaluate clustering
cost = model.computeCost(parsedData)
print("Bisecting K-means Cost = " + str(cost))

