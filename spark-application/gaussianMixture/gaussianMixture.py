from numpy import array

from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName('gaussianMixture')
sc = SparkContext(conf=conf)

# Load and parse the data
data = sc.textFile("/data/mllib/gmm_data.txt")
parsedData = data.map(lambda line: array([float(x) for x in line.strip().split(' ')]))

# Build the model (cluster the data)
gmm = GaussianMixture.train(parsedData, 2)

# Save and load model
gmm.save(sc, "target/org/apache/spark/PythonGaussianMixtureExample/GaussianMixtureModel")
sameModel = GaussianMixtureModel\
    .load(sc, "target/org/apache/spark/PythonGaussianMixtureExample/GaussianMixtureModel")


