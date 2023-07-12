# ✅ PROJECT 13

This project is not focused on providing financial advice, but on exploring a relevant business context (finance area) to study tools commonly used by data scientists in data analysis. The main objective is to build a machine learning model capable of predicting cryptocurrency prices. This project will be developed from the identification of the business problem to the delivery of the predictive model, based on publicly available real data.

Initially, we will use data on Bitcoin, but the model can be extended to other cryptocurrencies. Bitcoin is the first and oldest known cryptocurrency, launched in 2009 by anonymous Satoshi Nakamoto as open source. Bitcoin acts as a decentralized medium of digital exchange, with transactions verified and recorded on a distributed public ledger known as the Blockchain, eliminating the need for a central authority to keep records or broker transactions. Transaction blocks are "chained" together via SHA-256 cryptographic hashes of previous blocks, forming an immutable record of all transactions ever performed.

We will use historical Bitcoin quote data from 2011 to 2021. Due to the atypical nature of Bitcoin in 2022, we chose not to include this data in our analysis. Based on this historical data, our model will be trained to predict the real-time Bitcoin price based on new input data. It is noteworthy that this project can be expanded to any other financial instrument that has daily quote data available.

Keywords: Python Language, Apache Spark, PySpark, Data Analysis, Analytics, Criptmoedas, Bitcoin, Machine Learning, Real Time Analytics

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

Existem várias medidas que podem ser adotadas para otimizar e impulsionar os resultados do modelo de Machine Learning. Uma abordagem é criar diferentes versões do modelo, explorando diferentes algoritmos ou configurações. Isso permite avaliar qual abordagem oferece o melhor desempenho e resultados mais precisos. Além disso, é importante dedicar esforços à otimização dos hiperparâmetros do modelo. Os hiperparâmetros são parâmetros que não são aprendidos pelo modelo, mas que influenciam seu desempenho. Através de técnicas como busca em grade ou otimização bayesiana, é possível encontrar a combinação ideal de hiperparâmetros que maximize o desempenho do modelo.

Outra estratégia é modificar a forma de normalização dos dados ou até mesmo criar novos atributos que possam melhorar a representação dos dados e a capacidade de previsão do modelo. A experimentação desempenha um papel crucial nesse processo, permitindo testar diferentes abordagens e avaliar os resultados obtidos. É importante ressaltar que a validação do modelo de Machine Learning é um processo contínuo. É necessário realizar testes rigorosos e comparar os resultados com dados reais para garantir a eficácia do modelo em diferentes cenários. Através da experimentação constante e do ajuste das estratégias adotadas, é possível desenvolver um modelo validado e confiável.

# ✅ DATA SOURCES

O dataset está sendo fornecido a você e os dados foram extraídos do site abaixo: 

https://bitcoincharts.com/charts/bitstampUSD#rg60ztgSzm1g10zm2g25zv / https://drive.google.com/file/d/1zov5-RWVpyQk8vRvZXb9u3XUrWzasPvw/view

Ao acessar o web site você pode receber mensagem de que é um site inseguro. Isso se deve ao certificado digital expirado. Sempre tome cuidado ao acessar sites com esse tipo de mensagem, mas esse site em específico não apresenta perigo. De qualquer forma, o dataset está sendo fornecido. O arquivo CSV contém dadosde 2011a2021, com registrosOHLC (Open, High, Low, Close)da cotação do Bitcoin, Volume em BTC e Volume na moeda (nesse caso o dólar). A última coluna indica opreço ponderado do Bitcoin. Os “carimbos”de data/hora (timestamp) estão em hora Unix. Timestamps sem nenhuma negociação ou atividade têm seus campos de dados preenchidos com NaNs. Se estiver faltando um carimbo de data/hora ou se houver saltos, isso pode ser porque a Exchange estava inativa, a Exchange não existia ou algum outro erro técnico ocorreu na coleta dos dados.
