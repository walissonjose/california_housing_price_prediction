# Importar bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# Carregar os dados
housing = pd.read_csv('california_housing.csv')

# Separar as colunas de entrada (X) e saída (y)
X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

# Separar em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Selecionar as colunas que são categóricas
categorical_columns = ['ocean_proximity']

# Aplicar o one-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_train[categorical_columns])
X_train_encoded = encoder.transform(X_train[categorical_columns]).toarray()
X_test_encoded = encoder.transform(X_test[categorical_columns]).toarray()

# Remover as colunas categóricas dos conjuntos de treino e teste
X_train = X_train.drop(categorical_columns, axis=1)
X_test = X_test.drop(categorical_columns, axis=1)

# Juntar as colunas numéricas com as codificadas
X_train = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(X_train_encoded)], axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(X_test_encoded)], axis=1)

# Converter todas as colunas para strings
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# reshape y_test para uma matriz unidimensional com uma coluna
y_test = np.reshape(y_test, (-1, 1))

# transformar y_test com scalerOutput
y_test = scalerOutput.transform(y_test)

# Normalização
# Dados de entrada
y_train = y_train.to_frame()
y_train.columns = y_train.columns.astype(str)

scalerInput = StandardScaler()
scalerInput.fit(X_train)

X_train = scalerInput.transform(X_train)
X_test = scalerInput.transform(X_test)

# Dados de saída
scalerOutput = StandardScaler()
y_train.columns = y_train.columns.astype(str)  # converter nomes das colunas para string
scalerOutput.fit(y_train)

y_train = scalerOutput.transform(y_train)
y_test = scalerOutput.transform(y_test)

# Normalizar os dados
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

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
