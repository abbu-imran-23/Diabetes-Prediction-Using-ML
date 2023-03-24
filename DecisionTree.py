# Accuracy: 0.7738693467336684

# Import necessary libraries
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName('DiabetesPrediction').getOrCreate()

# Load the data
data = spark.read.csv('diabetes.csv', header=True, inferSchema=True)

# Create a vector assembler to combine the feature columns
assembler = VectorAssembler(inputCols=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], outputCol='features')
data = assembler.transform(data)

# Split the data into training and testing sets
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=42)

# Train the Decision Tree model
dt = DecisionTreeClassifier(labelCol='Outcome', featuresCol='features', maxDepth=5)
model = dt.fit(trainingData)

# Make predictions on the testing data
predictions = model.transform(testData)

# Evaluate the model's performance
evaluator = MulticlassClassificationEvaluator(labelCol='Outcome', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')

# Predict new instance
new_instance = [[2,124,90,34,26,20.4,0.543,25]]
new_instance_df = spark.createDataFrame(new_instance, ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
new_instance_transformed = assembler.transform(new_instance_df)
prediction = model.transform(new_instance_transformed)
print("Prediction for new instance:", prediction.select("prediction").collect()[0][0])
