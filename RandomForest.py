# Accuracy: 0.7487437185929648

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('DiabetesPrediction').getOrCreate()

data = spark.read.csv('diabetes.csv', header=True, inferSchema=True)

assembler = VectorAssembler(inputCols=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], outputCol='features')
data = assembler.transform(data)

(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=42)

rf = RandomForestClassifier(labelCol='Outcome', featuresCol='features', numTrees=10)
model = rf.fit(trainingData)

predictions = model.transform(testData)

evaluator = MulticlassClassificationEvaluator(labelCol='Outcome', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')

new_instance = [[2,124,90,34,26,20.4,0.543,25]]
new_instance_df = spark.createDataFrame(new_instance, ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
new_instance_transformed = assembler.transform(new_instance_df)
prediction = model.transform(new_instance_transformed)
print("Prediction for new instance:", prediction.select("prediction").collect()[0][0])