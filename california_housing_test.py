import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# carrega o conjunto de dados
housing = pd.read_csv('https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv', sep=",")

# seleciona as colunas relevantes
housing = housing[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']]

# separa as variáveis independentes e a variável dependente
X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

# separa os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# cria um objeto scaler para normalizar os dados de entrada
scalerInput = StandardScaler()
X_train = scalerInput.fit_transform(X_train)
X_test = scalerInput.transform(X_test)

# cria um objeto scaler para normalizar os dados de saída
scalerOutput = StandardScaler()
y_train = np.array(y_train).reshape(-1, 1)
y_train = scalerOutput.fit_transform(y_train)

y_test = np.array(y_test).reshape(-1, 1)
y_test = scalerOutput.transform(y_test)

# imprime as dimensões dos conjuntos de dados
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

# Construção da rede
mlp = MLPRegressor(hidden_layer_sizes=(3, ), activation='tanh', solver='adam',
                   max_iter=5000, tol=0.000001)

# Treinamento da rede
mlp.fit(X_train, y_train.ravel())

# Resultados
# Teste
y_pred = mlp.predict(X_test)
# Métricas
score = mlp.score(X_test, y_test)
RMSE = mean_squared_error(y_test, y_pred, squared=False)

# Gráfico
plt.figure(figsize=(6, 6))
plt.title(f'Teste = Score:{round(score, 4)} RMSE:{round(RMSE, 4)}')
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_pred)], [min(y_test), max(y_pred)], 'k')
plt.plot()
plt.xlabel('Target')
plt.ylabel('Predict')

# Treino
y_pred = mlp.predict(X_train)
# Métricas
score = mlp.score(X_train, y_train)
RMSE = mean_squared_error(y_train, y_pred, squared=False)

# Gráfico
plt.figure(figsize=(6, 6))
plt.title(f'Treino = Score:{round(score, 4)} RMSE:{round(RMSE, 4)}')
plt.scatter(y_train, y_pred)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_pred)], 'k')
plt.plot()
plt.xlabel('Target')
plt.ylabel('Predict')
plt.show()
