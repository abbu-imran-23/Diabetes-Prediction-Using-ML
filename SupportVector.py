# Accuracy : 0.7839195979899497 


# Import necessary libraries
from pyspark.ml.classification import LinearSVC
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

# Train the SVM model
svm = LinearSVC(maxIter=10, regParam=0.1, labelCol='Outcome', featuresCol='features')
model = svm.fit(trainingData)

# Make predictions on the testing data
predictions = model.transform(testData)

# Evaluate the model's performance
evaluator = MulticlassClassificationEvaluator(labelCol='Outcome', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')

# Predict new instance
new_instance = [[2,197,70,45,543,30.5,0.158,53]]
new_instance_df = spark.createDataFrame(new_instance, ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
new_instance_transformed = assembler.transform(new_instance_df)
prediction = model.transform(new_instance_transformed)
print("Prediction for new instance:", prediction.select("prediction").collect()[0][0])
