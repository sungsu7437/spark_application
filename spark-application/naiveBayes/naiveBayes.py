from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName('naiveBayes')
sc = SparkContext(conf=conf)


# Load and parse the data file.
data = MLUtils.loadLibSVMFile(sc, "/data/mllib/sample_libsvm_data.txt")

# Split data approximately into training (60%) and test (40%)
training, test = data.randomSplit([0.6, 0.4])

# Train a naive Bayes model.
model = NaiveBayes.train(training, 1.0)

# Make prediction and test accuracy.
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print('model accuracy {}'.format(accuracy))


