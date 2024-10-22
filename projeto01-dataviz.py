# Projeto 01 - Análise e Visualização de Dados de Vendas ao Longo do Tempo com PySpark
# Script de Visualização de Dados

# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define o diretório onde os arquivos CSV gerados estão localizados (diretório previsões deploy gerada) 
diretorio =  r'C:\Users\Alexandre\OneDrive\Documentos\profissional\projetos_publicados\cluster_central_apache_spark_docker\datasets\previsoesdeploy'

# Encontra o primeiro arquivo CSV no diretório
arquivo_csv = next((f for f in os.listdir(diretorio) if f.endswith('.csv')), None)
if arquivo_csv is None:
    raise FileNotFoundError("Nenhum arquivo CSV encontrado no diretório especificado.")

# Carrega os dados do CSV para um DataFrame
caminho_completo = os.path.join(diretorio, arquivo_csv)
dados = pd.read_csv(caminho_completo)

# Converte a coluna 'Date' para datetime para melhor manipulação
dados['Date'] = pd.to_datetime(dados['Date'])

# Cria um gráfico de linhas
plt.figure(figsize=(10, 5))
plt.plot(dados['Date'], dados['prediction'], marker='o', linestyle='-', color='b')
plt.title('Previsões de Vendas ao Longo do Tempo')
plt.xlabel('Data')
plt.ylabel('Previsão de Vendas')
plt.grid(True)

# Salva o gráfico como uma imagem PNG
plt.savefig('projeto01.png')

# Mostra o gráfico (opcional)
plt.show()
