# ✅ PROJECT 13

Previsão da cotação de Criptomoedas em tempo real com Pyspark e Machine Learning. Este projeto não tem como foco servir como aconselhamento financeiro, mas ism usar um importante contexto de negócio (área de finanças) para estudar ferramentas de análise de dados comuns no dia a dia de um cientista de dados. Vamos abodrdar o projeto desde a concepção do problema de negócio a ser resolvido até a entrega do modelo preditivo. Será utilizado dados reais disponíveis publicamente. Nesse projeto o objetivo é construir um modelo de machine learning capaz de prever a cotação de criptmoedas. Usaremos dados sobre o Bitcoin, porém pode-se estender a outras criptomoedas. O Bitcoin é a criptmoeda mais antiga conhecida, lançada pela primeira vez como código aberto em 2009 pelo anônimo Satoshi Nakamoto. O Bitcoin serve como um meio descentralizado de troca digital, com transações verificadas e registradas em um livro público distribuído (Blockchain) sem a necessidade de uma autoridade de manutenção de registros ou intermediário centra. Os blocos de transação contêm um hash criptográfico SHA-256 de blocos de transação anteriores e, portanto, são "encadeados"juntos, sevindo como um registro imutável de todas as transações que já ocorreram. Usaremos dados históricos de cotação do Bitcoin de 2011 a 2021. Como o ano de 2022 foi bem atípico par ao Bitcoin, optamos por nào usar dados desse ano. Com base em dados históricos de cotação do Bitcoin, nosso modelo deve ser capaz de prever a cotação do Bitcoin em tempo real a partir de novos dados de entrada. Este projeto pode ser  estendido para qualquer outro instrumento financeiro que tenha dados de cotação diária disponível. 

Para desenvolver esse projeto um dos frameworks principais utilizados, será o Apache Spark, mais precisamento com a bibiioteca PySpark. Para executar um software em um ambiente de cluster de computadores, o software deve ser capaz de realizar processamento distribuído, ou seja, dividir uma tarefa de processamento em sub-tarefas, enviá-las para as máquinas do cluster, coletar as respostas, juntar tudo e entregar o resultado.

Keywords: Python Language, Apache Spark, PySpark, Data Analysis, Analytics, Criptmoedas, Bitcoin, Machine Learning, Real Time Analytics

# ✅ PROCESS

Para trabalhar com o Apache Spark, primeiro devemos conectar no cluster Spark (seja um pseudo-cluster de uma única máquina,seja em um cluster de milhares de máquinas). A conexão é feita criando um SparkContext. Um SparkContext representa a conexão com um cluster Spark e pode ser usado para criar RDDs(Datasets Distribuídos), acumuladores e variáveis de broadcastnesse cluster.Apenas um SparkContext deve estar ativo por JVM. Você deve pararo SparkContext ativo antes de criar um novo.Com o contexto criado, podemos então criar uma sessão Spark. SparkSessioné o ponto de entrada para o SparkSQL. Você cria uma SparkSession usando o método Builder(),que dá acesso à API do Builder que você usa para configurar a sessão.Uma vez criada a sessão, SparkSession permite criar um DataFrame, criar um Dataset, acessar serviços SparkSQL (por exemplo, ExperimentalMethods, ExecutionListenerManager, UDFRegistration), executar uma consulta SQL, carregar uma tabela e o último,mas  não menos importante, acessar a interface  DataFrameReader para carregar um conjunto de dados do formato de sua escolha.Você pode ter quantas SparkSessions quiser em um único aplicativoSpark. O caso de uso comum é manter as entidades relacionais separadas logicamente em catálogos por SparkSession. No final, você interrompe uma SparkSession usando o método stop().

<img width="872" alt="image" src="https://github.com/lucashomuniz/Project-13/assets/123151332/19b667de-cc9b-4ffd-abee-119344147e9f">

No Spark, um DataFrame é uma coleção distribuída de dados organizados em colunas nomeadas. É conceitualmente equivalente a uma tabela em um banco de dados relacional ou um dataframe em R e/ou Python, mas com otimizações mais ricase para ambiente distribuído. Os DataFrames podem ser construídos a partir de uma ampla variedade de fontes, como: arquivos de dados estruturados, tabelas no Hive, bancos de dados externos ou RDDs.Uma vez criados, os DataFrames fornecem uma estrutura de dadosespecífica para manipulação de dados distribuídos. Semelhante aos RDDs, os DataFrames são avaliados  lentamente. Ou seja, a computação só acontece quandouma ação (por exemplo, exibir resultado, salvar saída) é necessária. Isso permite que suas execuções sejam otimizadas. Todas as operações do DataFrame também são paralelizadas automaticamente e distribuídas em clusters.

O PySpark é ideal para processamento de dados e Machine Learningem ambiente em larga escala, mas para a análise exploratória e criação de gráficos, o PySpark não oferece as melhores ferramentas, devido seu caráter distribuído.Vocêpode alternar entre asestruturas de dados com PySpark e Pandas. O Pandas será ideal para criação de gráficos com o Seaborn e Matplotlib. Quando vocêcria uma estrutura de dados ela se torna um objeto em Python e cada objeto tem métodos e atributos. Objetos do Pandas serão ideais para tarefas sequenciais como criação de gráficos ou manipulações em volumes de dados menores. Para o processamento de grandes conjuntos de dados objetos do PySpark são os ideais.

Na análise exploratória foi possível ter uma boa idéia de como os dados podem estar sendo organizados, claramente há um padrão entre as variáveis, podendo com isso utilizar sem problemas o dataframe em um modelo de machine learning. Entranto, claramente é possível perceber que temos variáveis que possuem um alto nível de correlação chegando ao ponto de criar uma multicolinearidade (variáveis de entrada que representam a mesma informação). 

<img width="891" alt="image" src="https://github.com/lucashomuniz/Project-13/assets/123151332/48199e2e-632b-421a-850d-33e89c0ce8b4">


# ✅ CONCLUSION

Algoritmos de Machine Learning utilizados foram: Benchmark, 















# ✅ DATA SOURCES

O dataset está sendo fornecido a você e os dados foram extraídos do site abaixo: 

https://bitcoincharts.com/charts/bitstampUSD#rg60ztgSzm1g10zm2g25zv / https://drive.google.com/file/d/1zov5-RWVpyQk8vRvZXb9u3XUrWzasPvw/view

Ao acessar o web site você pode receber mensagem de que é um site inseguro. Isso se deve ao certificado digital expirado. Sempre tome cuidado ao acessar sites com esse tipo de mensagem, mas esse site em específico não apresenta perigo. De qualquer forma, o dataset está sendo fornecido. O arquivo CSV contém dadosde 2011a2021, com registrosOHLC (Open, High, Low, Close)da cotação do Bitcoin, Volume em BTC e Volume na moeda (nesse caso o dólar). A última coluna indica opreço ponderado do Bitcoin. Os “carimbos”de data/hora (timestamp) estão em hora Unix. Timestamps sem nenhuma negociação ou atividade têm seus campos de dados preenchidos com NaNs. Se estiver faltando um carimbo de data/hora ou se houver saltos, isso pode ser porque a Exchange estava inativa, a Exchange não existia ou algum outro erro técnico ocorreu na coleta dos dados.
