# ✅ PROJECT 13

Previsão da cotação de Criptomoedas em tempo real com Pyspark e Machine Learning. Este projeto não tem como foco servir como aconselhamento financeiro, mas ism usar um importante contexto de negócio (área de finanças) para estudar ferramentas de análise de dados comuns no dia a dia de um cientista de dados. Vamos abodrdar o projeto desde a concepção do problema de negócio a ser resolvido até a entrega do modelo preditivo. Será utilizado dados reais disponíveis publicamente. 

Nesse projeto o objetivo é construir um modelo de machine learning capaz de prever a cotação de criptmoedas. Usaremos dados sobre o Bitcoin, porém pode-se estender a outras criptomoedas. O Bitcoin é a criptmoeda mais antiga conhecida, lançada pela primeira vez como código aberto em 2009 pelo anônimo Satoshi Nakamoto. O Bitcoin serve como um meio descentralizado de troca digital, com transações verificadas e registradas em um livro público distribuído (Blockchain) sem a necessidade de uma autoridade de manutenção de registros ou intermediário centra. Os blocos de transação contêm um hash criptográfico SHA-256 de blocos de transação anteriores e, portanto, são "encadeados"juntos, sevindo como um registro imutável de todas as transações que já ocorreram. 

Usaremos dados históricos de cotação do Bitcoin de 2011 a 2021. Como o ano de 2022 foi bem atípico par ao Bitcoin, optamos por nào usar dados desse ano. Com base em dados históricos de cotação do Bitcoin, nosso modelo deve ser capaz de prever a cotação do Bitcoin em tempo real a partir de novos dados de entrada. Este projeto pode ser  estendido para qualquer outro instrumento financeiro que tenha dados de cotação diária disponível.

Para desenvolver esse projeto um dos frameworks principais utilizados, será o Apache Spark, mais precisamento com a bibiioteca PySpark.

Falar os pontos positivos em se utilizar ApacheSpark / PySpark

Keywords: Python Language, Apache Spark, PySpark, Data Analysis, Criptmoedas, Bitcoin, Machine Learning

# ✅ PROCESS

# ✅ CONCLUSION

# ✅ DATA SOURCES

O dataset está sendo fornecido a você e os dados foram extraídos do site abaixo: 

https://bitcoincharts.com/charts/bitstampUSD#rg60ztgSzm1g10zm2g25zv / https://drive.google.com/file/d/1zov5-RWVpyQk8vRvZXb9u3XUrWzasPvw/view

Ao acessar o web site você pode receber mensagem de que é um site inseguro. Isso se deve ao certificado digital expirado. Sempre tome cuidado ao acessar sites com esse tipo de mensagem, mas esse site em específico não apresenta perigo. De qualquer forma, o dataset está sendo fornecido. O arquivo CSV contém dadosde 2011a2021, com registrosOHLC (Open, High, Low, Close)da cotação do Bitcoin, Volume em BTC e Volume na moeda (nesse caso o dólar). A última coluna indica opreço ponderado do Bitcoin. Os “carimbos”de data/hora (timestamp) estão em hora Unix. Timestamps sem nenhuma negociação ou atividade têm seus campos de dados preenchidos com NaNs. Se estiver faltando um carimbo de data/hora ou se houver saltos, isso pode ser porque a Exchange estava inativa, a Exchange não existia ou algum outro erro técnico ocorreu na coleta dos dados.
