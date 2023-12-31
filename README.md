# ✅ PROJECT 13

This project is not focused on providing financial advice, but on exploring a relevant business context (finance area) to study tools commonly used by data scientists in data analysis. The main objective is to build a machine learning model capable of predicting cryptocurrency prices. This project will be developed from the identification of the business problem to the delivery of the predictive model, based on publicly available real data.

Initially, we will use data on Bitcoin, but the model can be extended to other cryptocurrencies. Bitcoin is the first and oldest known cryptocurrency, launched in 2009 by anonymous Satoshi Nakamoto as open source. Bitcoin acts as a decentralized medium of digital exchange, with transactions verified and recorded on a distributed public ledger known as the Blockchain, eliminating the need for a central authority to keep records or broker transactions. Transaction blocks are "chained" together via SHA-256 cryptographic hashes of previous blocks, forming an immutable record of all transactions ever performed.

We will use historical Bitcoin quote data from 2011 to 2021. Due to the atypical nature of Bitcoin in 2022, we chose not to include this data in our analysis. Based on this historical data, our model will be trained to predict the real-time Bitcoin price based on new input data. It is noteworthy that this project can be expanded to any other financial instrument that has daily quote data available.

Keywords: Python Language, Apache Spark, PySpark, Data Analysis, Analytics, Cryptcurrencies, Bitcoin, Machine Learning, Real Time Analytics

# ✅ PROCESS

To use Apache Spark, it is necessary to connect to the Spark cluster, either a pseudo-cluster on a single machine or a cluster with thousands of machines. This connection is established by creating a SparkContext. The SparkContext represents the connection with the Spark cluster and allows the creation of RDDs (Distributed Datasets), accumulators and broadcast variables in this cluster. It is important to note that only one SparkContext can be active per JVM instance, and it is necessary to terminate the active SparkContext before creating a new one.

Once the context is established, it is possible to create a Spark session. SparkSession is the entry point to SparkSQL. You can create a SparkSession using the Builder() method, which provides access to the Builder API to configure the session. With the session created, SparkSession allows creating DataFrames, creating Datasets, accessing SparkSQL services (such as ExperimentalMethods, ExecutionListenerManager and UDFRegistration), executing SQL queries, loading tables and last but not least importantly, access to the DataFrameReader interface to load datasets in the desired format. You can have as many SparkSessions as you need in a single Spark application. The common use case is to keep relational entities logically separated in catalogs by SparkSession. When finishing using a SparkSession, it is important to close it using the stop() method.

<img width="933" alt="image" src="https://github.com/lucashomuniz/Project-13/assets/123151332/29304ebe-3cad-40ac-b273-af0b007a7e29">

PySpark is an excellent choice for data processing and machine learning in large-scale environments. However, for exploratory analysis and graphing, PySpark may not offer the best tools due to its distributed nature. In that case, you can switch between PySpark and Pandas data structures. Pandas is especially suited for creating graphs using libraries such as Seaborn and Matplotlib. When you create a data structure in Pandas, it becomes an object in Python, and each object has its own methods and attributes. Pandas objects are ideal for sequential tasks such as creating graphs or manipulating smaller datasets. On the other hand, for processing large datasets, PySpark objects are better suited. They are designed to handle data scalability and distribution, enabling efficient processing in distributed environments.

During the exploratory analysis, it was possible to obtain a good understanding of how the data are organized. It was evident that there is a pattern between the variables, which allows using the dataframe without problems in a machine learning model. However, it was also observed that some variables present a high level of correlation, reaching the point of creating multicollinearity, that is, several input variables represent the same information.

<img width="891" alt="image" src="https://github.com/lucashomuniz/Project-13/assets/123151332/48199e2e-632b-421a-850d-33e89c0ce8b4">


# ✅ CONCLUSION

After using new data from the "bitcoincharts" website, more specifically the two most recent data, we performed a check to assess the accuracy of the prediction model. This evaluation was based on the variables "open", "VolBTC" and "VolCurrency". The resulting predicted values were 20538.57 and 21612.56 for the "Predicted_Price" field. When analyzing the data mentioned above, referring to the last two rows, we observe that the difference between the predicted values and the actual data is approximately $200 to $600. This difference is not considered significant, especially considering that this is the first machine learning model and the first version developed. Therefore, the result is considered acceptable and indicates qualified performance for the current model. This finding is encouraging as it suggests that the model is providing reasonably accurate predictions based on the variables used. However, it is important to emphasize that continuous improvement and rigorous validation are necessary to further improve the model and ensure more reliable results in the future.

<img width="837" alt="image" src="https://github.com/lucashomuniz/Project-13/assets/123151332/c35910ee-ae0a-4e4c-8c6a-dbc5cef7a569">

There are several measures that can be taken to optimize and boost the results of the Machine Learning model. One approach is to create different versions of the model, exploring different algorithms or configurations. This allows you to assess which approach offers the best performance and most accurate results. In addition, it is important to dedicate efforts to optimizing the model's hyperparameters. Hyperparameters are parameters that are not learned by the model, but that influence its performance. Through techniques such as grid search or Bayesian optimization, it is possible to find the ideal combination of hyperparameters that maximize the performance of the model.

Another strategy is to change the way data is normalized or even create new attributes that can improve data representation and the model's predictive capacity. Experimentation plays a crucial role in this process, allowing testing different approaches and evaluating the results obtained. It is important to emphasize that the validation of the Machine Learning model is an ongoing process. It is necessary to conduct rigorous tests and compare the results with real data to ensure the effectiveness of the model in different scenarios. Through constant experimentation and adjustment of adopted strategies, it is possible to develop a validated and reliable model.

# ✅ DATA SOURCES

The dataset is being provided to you and the data has been extracted from the website below:

https://bitcoincharts.com/charts/bitstampUSD#rg60ztgSzm1g10zm2g25zv

https://drive.google.com/file/d/1zov5-RWVpyQk8vRvZXb9u3XUrWzasPvw/view

When accessing the website, you may receive a message that it is an unsafe website. This is due to the expired digital certificate. Always be careful when accessing sites with these types of messages, but this specific site is not dangerous. Anyway, the dataset is being provided. The CSV file contains data from 2011 to 2021, with OHLC (Open, High, Low, Close) records of the Bitcoin quote, Volume in BTC and Volume in the currency (in this case the dollar). The last column indicates the weighted Bitcoin price. Timestamps are in Unix time. Timestamps without any trades or activity have their data fields filled with NaNs. If a timestamp is missing or there are jumps, it could be because Exchange was down, Exchange did not exist, or some other technical error occurred while collecting the data.
