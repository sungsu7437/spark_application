from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import DenseVector
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName('regressionModel')
sc = SparkContext(conf=conf)
# Load and parse the data
def parsePoint(line):
    values = line.split()
    return LabeledPoint(float(values[0]),
                        DenseVector([float(x.split(':')[1]) for x in values[1:]]))

data = sc.textFile("/data/mllib/sample_linear_regression_data.txt")
parsedData = data.map(parsePoint)

# Build the model
model = LinearRegressionWithSGD.train(parsedData)

# Get predictions
valuesAndPreds = parsedData.map(lambda p: (float(model.predict(p.features)), p.label))

# Instantiate metrics object
metrics = RegressionMetrics(valuesAndPreds)

# Squared Error
print("MSE = %s" % metrics.meanSquaredError)
print("RMSE = %s" % metrics.rootMeanSquaredError)

# R-squared
print("R-squared = %s" % metrics.r2)

# Mean absolute error
print("MAE = %s" % metrics.meanAbsoluteError)

# Explained variance
print("Explained variance = %s" % metrics.explainedVariance)

