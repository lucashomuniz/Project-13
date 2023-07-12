"""
========
IMPORTS
========
"""
import pyspark
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

# Formatting the outputs
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

"""
==================
SPARK ENVIRONMENT
==================
"""

# Setting random seed (seed) for notebook reproducibility
rnd_seed = 23
np.random.seed = rnd_seed
np.random.set_state = rnd_seed

# Creating the Spark Context
sc = SparkContext(appName = "Project-13")

# Creating the Spark session
spark_session = SparkSession.Builder().getOrCreate()

# Create a SparkSession
spark_session = SparkSession.builder \
    .appName("Project-13") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Display the spark_session object
spark_session

"""
=============
LOADING DATA
=============
"""

# Load the data from the Spark session
df_spark = spark_session.read.csv('dados/dataset.csv', header = 'true', inferSchema = 'true')

# Object type
type(df_spark)

# Visualize the data
df_spark.show()

# Visualize the metadata (schema)
df_spark.printSchema()

# Check the number of lines
(df_spark.count())

"""
=============================
DATA WRANGLING WITH SPARKSQL
=============================
"""

# Create a temp table from the dataframe
# Temporary tables are useful when you want the result set to be visible
# for all other Spark sessions
df_spark.createOrReplaceTempView('dados_bitcoin')

# Run an SQL query
df_bitcoin = spark_session.sql("select *, from_unixtime(Timestamp) as `dateTime` from dados_bitcoin")
type(df_bitcoin)

# Visualize the data
df_bitcoin.show()

# Remove NA values (it doesn't make any sense here, as all quotes columns are NA)
df_bitcoin = df_bitcoin.dropna('any')

# Visualize the data
df_bitcoin.show()

# Number of records
df_bitcoin.count()

# Let's rename some columns to facilitate data manipulation
df_bitcoin = df_bitcoin.withColumnRenamed("Volume_(BTC)", "VolBTC").withColumnRenamed("Volume_(Currency)", "VolCurrency")

# Visualize
df_bitcoin.show()

# The dateTime column provides the quote data details. Let's separate the date elements into different colors.
# Let's split the dataframe by extracting data
df_data = df_bitcoin.withColumn("date", split(col("dateTime")," ").getItem(0))
type(df_data)

# Let's split the dataframe by extracting time
df_data = df_data.withColumn("time", split(col("dateTime")," ").getItem(1))

# Schema
df_data.printSchema()

# Visualize the data
df_data.show()

# Let's split the dataframe by extracting the time
df_data_hora = df_data.withColumn("hour", split(col("time"),":").getItem(0))
df_data_hora.printSchema()
df_data_hora.show()

# Let's adjust the date format to extract the day of the week
df_data_hora = df_data_hora.withColumn("date", df_data_hora["date"].cast(DateType())).withColumn("hour",
df_data_hora["hour"].cast(DoubleType())).withColumn("dateTime", df_data_hora["dateTime"].cast(DateType()))
df_data_hora.show()

# Let's extract the day of the week
df_data_hora = df_data_hora.withColumn('day_of_week', dayofweek(df_data_hora.date))
df_data_hora.printSchema()

# Let's extract the year from the quotation
df_data_hora_ano = df_data_hora.withColumn("year", split(col("date"),"-").getItem(0))
df_data_hora_ano.show()

# Convert Spark dataframe to Pandas for easy exploratory data analysis.
df_pandas = df_data_hora_ano.toPandas()
type(df_pandas)

# Getting individual values to use in graphs
hour = df_pandas["hour"].values.tolist()
weighted_price = df_pandas["Weighted_Price"].values.tolist()
volume_BTC = df_pandas["VolBTC"].values.tolist()
date_of_week = df_pandas["day_of_week"].values.tolist()
year = df_pandas["year"].values.tolist()

"""
=====================
EXPLORATORY ANALYSIS
=====================
"""

# Heatmap to visualize the correlation
df_pandas = df_pandas.select_dtypes(include='number')
corr = df_pandas.corr()
f,ax = plt.subplots(figsize = (10, 10))
sns.heatmap(corr, annot = True, linewidths = .5, fmt = '.1f', ax = ax)
plt.show()

# Ideally we want high correlation between the input variables and the output variable and low
# correlation between input variables!

# Scatter Plot Bitcoin Volume x Currency Volume
plt.figure(figsize = (12,5))
sns.set(style = 'whitegrid')
df_pandas.plot(kind = 'scatter', x = 'VolBTC', y = 'VolCurrency')
plt.xlabel('Bitcoin volume')
plt.ylabel('Currency Volume')
plt.title('Scatter Plot Bitcoin Volume x Currency Volume')
plt.show()

# Line Plot Cotação Open x High 
plt.figure(figsize = (16,5))
df_pandas.Open.plot(kind = 'line', color = 'r', label = 'Open', alpha = 0.5, linewidth = 5, grid = True, linestyle = ':')
df_pandas.High.plot(color = 'g', label = 'High', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.')
plt.legend(loc = 'upper left') 
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Open x High Quote Line Plot')
plt.show()

# Opening quote histogram
df_pandas.Open.plot(kind = 'hist', bins = 50)

# Plot weighted quote value (target variable) per hour
plt.plot(hour, weighted_price , 'g*')
plt.xlabel('Hour')
plt.ylabel('Weighted Quotation Value')
plt.title('Weighted Value of Quotation Per Hour')
plt.show()

# Plot weighted quote value by day of week
plt.plot(date_of_week, weighted_price, 'b*')
plt.xlabel('Day of the Week')
plt.ylabel('Weighted Quotation Value')
plt.title('Weighted Quotation Value by Day of the Week')
plt.show()

# VolBTC hourly plot
plt.plot(hour, volume_BTC, 'r*')
plt.xlabel('Hour')
plt.ylabel('VolBTC')
plt.title('Bitcoin Trading Volume Per Hour')
plt.show()

# VolBTC plot by day of the week
plt.plot(date_of_week, volume_BTC, 'yo')
plt.xlabel('Day of the week')
plt.ylabel('VolBTC')
plt.title('Bitcoin Traded Volume by Day of the Week')
plt.show()

# Plot weighted quote value by year
plt.plot(year, weighted_price , 'm^')
plt.xlabel('Year')
plt.ylabel('Bitcoin Traded Volume by Day of the Week')
plt.title('Weighted Quotation Value Per Year')
plt.show()

# Volume plot by year
plt.plot(year, volume_BTC , 'kD')
plt.xlabel('Year')
plt.ylabel('BTC Volume')
plt.title('BTC Volume Traded Per Year')
plt.show()

"""
==============================
ATTRIBUTE ENGINEERING PYSPARK
==============================
"""
# Attribute Engineering with PySpark
df_bitcoin.printSchema()

# Prepare the attribute vector
assembler = VectorAssembler(inputCols = ['Open', 'VolBTC', 'VolCurrency'], outputCol = "features")

# Create the attribute vector dataframe
df_assembled = assembler.transform(df_bitcoin)

# Visualize the data
df_assembled.show(10, truncate = False)

# Normalization
# Split into training and testing data
dados_treino, dados_teste = df_assembled.randomSplit([.7,.3], seed = rnd_seed)
type(dados_treino)

# Create the scaler
scaler = MinMaxScaler(inputCol = "features", outputCol = "scaled_features")

# Fit on training data
scalerModel = scaler.fit(dados_treino)

# Fit and transform training data
dados_treino_scaled = scalerModel.transform(dados_treino)

# Transform in the test data
dados_teste_scaled = scalerModel.transform(dados_teste)
dados_treino_scaled.select("features", "scaled_features").show(10, truncate = False)

"""
====================================
MACHINE LEARNING MODEL 1: BENCHMARK
====================================
"""

# Create the regression model
modelo_lr_v1 = (LinearRegression(featuresCol = 'scaled_features', labelCol = "Weighted_Price", predictionCol = 'Predicted_price',
maxIter = 100, regParam = 0.3, elasticNetParam = 0.8, standardization = False))

# Train the model
modelo_v1 = modelo_lr_v1.fit(dados_treino_scaled)

# If it has a WARN message, it indicates that Spark did not find the WARN library
# linear algebra optimization (which needs to be installed but is not required for this project).

# Save the model to disk
modelo_v1.write().overwrite().save("modelos/modelo_v1")

# Model Evaluation
# Predictions with test data
previsoes_v1 = modelo_v1.transform(dados_teste_scaled)

# Select the columns
pred_data_v1 = previsoes_v1.select("Predicted_price", "Weighted_Price").show(10)

# Mean Absolute Error
print("Mean Absolute Error (MAE) in the test data: {0}".format(modelo_v1.summary.meanAbsoluteError))

# Create an evaluator for the regression model
evaluator = RegressionEvaluator(labelCol = "Weighted_Price", predictionCol = "Predicted_price", metricName = "rmse")

# Apply the evaluator
rmse_v1 = evaluator.evaluate(previsoes_v1)
print("Root Mean Squared Error (RMSE) in the test data = %g" % rmse_v1)

# Extract the predictions
pred_results_v1 = modelo_v1.evaluate(dados_teste_scaled)

# Actual Y values being converted to Pandas format
Y = pred_results_v1.predictions.select('Weighted_Price').toPandas()

# Predicted Y values being converted to Pandas format
_Y = pred_results_v1.predictions.select("Predicted_price").toPandas()

# Distribution of actual values x predicted values
sns.set_style("dark")
ax1 = sns.displot(Y, color = "r", label = "Actual Values")
sns.displot(_Y, color = "b", label = "Expected Values")

# Plot of actual values x predicted values
plt.figure(figsize = (12,7))
plt.plot(Y, color = 'green', marker = '*', linestyle = 'dashed', label = 'Predicted Price')
plt.plot(_Y, color = 'red', label = 'Weighted Price')
plt.title('Model Result')
plt.xlabel('Real Value')
plt.ylabel('Expected Value')
plt.legend()

"""
============================================================
MACHINE LEARNING MODEL 2: HYPERPARAMETER OPTIMIZATION MODEL
============================================================
"""

# Create the model
modelo_lr_v2 = (LinearRegression(featuresCol = 'scaled_features', labelCol = "Weighted_Price", predictionCol = 'Predicted_price'))

# Create a grid for hyperparameter optimization
grid = ParamGridBuilder().addGrid(modelo_lr_v2.maxIter, [50, 100]).build()

# Create the evaluator (will be used in cross validation)
evaluator = RegressionEvaluator(labelCol = "Weighted_Price", predictionCol = "Predicted_price", metricName = "rmse")

# Create the CrossValidator
cv = CrossValidator(estimator = modelo_lr_v2, estimatorParamMaps = grid, evaluator = evaluator, parallelism = 2)

# Train the CrossValidator
cvModel = cv.fit(dados_treino_scaled)

# Extract the best model from CrossValidator
modelo_v2 = cvModel.bestModel

# Save the model to disk
modelo_v2.write().overwrite().save("modelos/modelo_v2")

# Model Evaluation
# Predictions with test data
previsoes_v2 = modelo_v2.transform(dados_teste_scaled)

# Select the columns
pred_data_v2 = previsoes_v2.select("Predicted_price", "Weighted_Price").show(10)

# Mean Absolute Error
print("MAE: {0}".format(modelo_v2.summary.meanAbsoluteError))
evaluator = RegressionEvaluator(labelCol = "Weighted_Price", predictionCol = "Predicted_price", metricName = "rmse")

# Apply the evaluator
rmse_v2 = evaluator.evaluate(previsoes_v2)
print("Root Mean Squared Error (RMSE) in the test data = %g" % rmse_v2)

# Plot of actual values x predicted values
# Extract the predictions
pred_results_v2 = modelo_v2.evaluate(dados_teste_scaled)

# Actual Y values being converted to Pandas format
Y = pred_results_v2.predictions.select('Weighted_Price').toPandas()

# Predicted Y values being converted to Pandas format
_Y = pred_results_v2.predictions.select("Predicted_price").toPandas()

# Plot
sns.set_style("dark")
ax1 = sns.displot(Y, color = "r", label = "Actual Values")
sns.displot(_Y, color = "b", label = "Expected Values")

# Plot of actual values x predicted values
plt.figure(figsize = (12,7))
plt.plot(Y, color = 'green', marker = '*', linestyle = 'dashed', label = 'Predicted Price')
plt.plot(_Y, color = 'red', label = 'Weighted Price')
plt.title('Model Result')
plt.xlabel('Real Value')
plt.ylabel('Expected Value')
plt.legend()

# The WARN messages in v2_model training indicate that the model
# seems unstable and maybe overfitting. We will use template_v1.

"""
====================
REAL TIME FORECASTS
====================
"""

# New data
novos_dados = [[20546.29, 3422.57, 72403082.02], [21620.85, 3271.14, 71319207.5]]

# Prepare the Pandas dataframe
df_novos_dados = pd.DataFrame(novos_dados, columns = ['Open', 'VolBTC', 'VolCurrency'])

# View
df_novos_dados

# Convert Pandas dataframe to Spark dataframe
df_novos_dados_spark = spark_session.createDataFrame(df_novos_dados) 

# Schema
df_novos_dados_spark.printSchema()

# Visualize
df_novos_dados_spark.show()

# Create the attribute vector dataframe
df_assembled = assembler.transform(df_novos_dados_spark)

# Visualize the data
df_assembled.show()

# Normalize the data
df_assembled_scaled = scalerModel.transform(df_assembled)

# Predictions with the new data
previsoes = modelo_v1.transform(df_assembled_scaled)

# Print the predictions
pred_data = previsoes.select("Predicted_price").show()

# Terminate the Spark session
spark_session.stop()