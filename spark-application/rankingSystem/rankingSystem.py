from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName('rankingSystems')
sc = SparkContext(conf=conf)
# Read in the ratings data
lines = sc.textFile("/data/mllib/sample_movielens_data.txt")

def parseLine(line):
    fields = line.split("::")
    return Rating(int(fields[0]), int(fields[1]), float(fields[2]) - 2.5)
ratings = lines.map(lambda r: parseLine(r))

# Train a model on to predict user-product ratings
model = ALS.train(ratings, 10, 10, 0.01)

# Get predicted ratings on all existing user-product pairs
testData = ratings.map(lambda p: (p.user, p.product))
predictions = model.predictAll(testData).map(lambda r: ((r.user, r.product), r.rating))

ratingsTuple = ratings.map(lambda r: ((r.user, r.product), r.rating))
scoreAndLabels = predictions.join(ratingsTuple).map(lambda tup: tup[1])

# Instantiate regression metrics to compare predicted and actual ratings
metrics = RegressionMetrics(scoreAndLabels)

# Root mean squared error
print("RMSE = %s" % metrics.rootMeanSquaredError)

# R-squared
print("R-squared = %s" % metrics.r2)

