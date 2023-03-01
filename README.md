README - Regressão com Rede Neural Artificial (RNA)

Este repositório contém o código-fonte de um modelo de regressão utilizando Rede Neural Artificial (RNA), desenvolvido em Python, utilizando a biblioteca Scikit-Learn.

O modelo foi treinado utilizando o conjunto de dados California Housing disponibilizado pelo Google. Este conjunto de dados contém informações sobre preços de imóveis na Califórnia, incluindo atributos como a longitude, latitude, idade da habitação, número de quartos, renda média da população local, entre outros.

O modelo utiliza a RNA Multi-Layer Perceptron (MLP) para realizar a regressão. A MLP é uma rede neural artificial com camadas ocultas que podem ser ajustadas para diferentes tarefas de aprendizado de máquina. Neste modelo, foram utilizadas apenas 1 camada oculta com 3 neurônios.

O modelo utiliza as métricas de R^2 e RMSE para avaliar sua performance. O R^2 (coeficiente de determinação) mede a proporção da variância na variável dependente que é explicada pelas variáveis independentes do modelo. O RMSE (erro quadrático médio) mede a raiz quadrada da média dos erros ao quadrado entre os valores preditos e os valores reais. O objetivo do modelo é minimizar o RMSE e maximizar o R^2.

O modelo utiliza o conjunto de dados de treino para ajustar os pesos da rede neural e o conjunto de dados de teste para avaliar o desempenho da rede neural após o treinamento.

Dependências
Python 3.7 ou superior
Pandas 1.1.5 ou superior
Numpy 1.19.5 ou superior
Matplotlib 3.3.4 ou superior
Scikit-Learn 0.24.2 ou superior
Executando o modelo
Para executar o modelo, siga os seguintes passos:

Clone o repositório para a sua máquina local:
bash
Copy code
git clone https://github.com/seu-usuario/seu-repositorio.git
Abra o arquivo california_housing.py em um ambiente de desenvolvimento Python, como o Jupyter Notebook ou o PyCharm.

Execute o arquivo california_housing.py. O modelo será treinado e avaliado, e serão gerados dois gráficos de dispersão que mostram a performance do modelo nos conjuntos de treino e teste.

Observe os valores de R^2 e RMSE no título de cada gráfico para avaliar o desempenho do modelo.

Arquivos
california_housing.py: arquivo principal do modelo de regressão com RNA.
california_housing_test.py: arquivo com testes unitários do modelo. Utilize este arquivo para verificar se o modelo está funcionando corretamente antes de executá-lo com seus próprios dados.
california_housing_train.csv: conjunto de dados utilizado para treinar e avaliar o modelo. Este arquivo foi disponibilizado pelo Google.
Licença
Este repositório está licenciado sob a licença MIT.