# Importar bibliotecas
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Carregar os dados
# Entrada
X = pd.read_csv('entrada.csv')

# Saída
y = pd.read_csv('saida.csv')

# Separação dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Normalização
# Dados de entrada
scalerInput = StandardScaler()
scalerInput.fit(X_train)

X_train = scalerInput.transform(X_train)
X_test = scalerInput.transform(X_test)

# Dados de saída
scalerOutput = StandardScaler()
scalerOutput.fit(y_train)

y_train = scalerOutput.transform(y_train)
y_test = scalerOutput.transform(y_test)


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
