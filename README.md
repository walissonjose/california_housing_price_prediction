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

# Construção da rede
# mlp = MLPRegressor(hidden_layer_sizes=(3, ), activation='tanh', solver='adam', max_iter=5000, tol=0.000001)
Este trecho de código é responsável por construir a rede neural artificial (RNA) que será utilizada para realizar a regressão dos dados. Abaixo segue a explicação de cada termo:

MLPRegressor: classe utilizada para construir um modelo de regressão utilizando uma RNA do tipo Perceptron de Múltiplas Camadas (Multi-Layer Perceptron, MLP).

hidden_layer_sizes=(3, ): parâmetro que define a arquitetura da rede. Neste caso, temos uma rede com apenas uma camada oculta, que possui 3 neurônios. Caso desejássemos adicionar mais camadas ocultas, poderíamos informar uma tupla com a quantidade de neurônios de cada camada (por exemplo, hidden_layer_sizes=(10, 5)).

activation='tanh': função de ativação utilizada nos neurônios da rede. Neste caso, foi escolhida a função tangente hiperbólica (tanh), que é uma função não-linear que mapeia um número real em um intervalo entre -1 e 1.

solver='adam': algoritmo utilizado para realizar a otimização dos pesos da rede durante o treinamento. Neste caso, foi escolhido o algoritmo Adam, que é um método de otimização baseado em gradiente estocástico.

max_iter=5000: número máximo de iterações permitidas durante o treinamento da rede. Caso o treinamento não tenha convergido após esse número de iterações, o processo é interrompido.

tol=0.000001: tolerância para considerar que o treinamento convergiu. Caso a diferença entre o erro da iteração atual e o erro da iteração anterior seja menor que tol, o treinamento é considerado como tendo convergido.

No geral, os parâmetros utilizados neste trecho de código visam construir uma rede neural com uma arquitetura simples, utilizando uma função de ativação não-linear e um algoritmo de otimização eficiente, que permita obter uma boa aproximação dos dados em um tempo de treinamento razoável. O R^2 e o RMSE são métricas que são calculadas posteriormente, utilizando o modelo treinado, para avaliar a qualidade da aproximação. O R^2 é uma medida de quão bem o modelo se ajusta aos dados (variando de 0 a 1, sendo 1 o melhor valor possível) e o RMSE é uma medida da magnitude do erro de predição (quanto menor, melhor).