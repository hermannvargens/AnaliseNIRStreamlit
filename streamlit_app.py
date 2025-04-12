import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Função para carregar o modelo PLS salvo
@st.cache_resource
def load_model():
    #model = joblib.load('pls_normalizado.joblib')
    #model = joblib.load('pls_nao_normalizado.joblib') 
    model = joblib.load('knn_normalizado.joblib')
    #model = joblib.load('knn_nao_normalizado.joblib')
    #model = joblib.load('rfr.joblib')
    #model = joblib.load('svr_normalizado.joblib')
    #model = joblib.load('svr_nao_normalizado.joblib')
    return model

# Função para processar o arquivo CSV com o mesmo pré-processamento usado no Jupyter
def process_csv(file):
    df = pd.read_csv(file)
    
    # Corrigir nomes das colunas (virgula para ponto e remover ponto extra)
    df.columns = [col.replace('.', '').replace(',', '.') for col in df.columns]
    
    # Corrigir valores decimais e converter para float
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

#Função para ler o arquivo CSV utilizado para a modelagem
def read_csv():
    # Carregar os dados e excluir colunas desnecessárias
    df = pd.read_csv("espectros_derivada.csv", delimiter=";")
    df = df[:-3]
    
    # Substituir '.' por '' e ',' por '.' nos nomes das colunas
    df.columns = [col.replace('.', '').replace(',', '.') for col in df.columns]
    
    # substituir o ponto por '' e a virgula por ponto em todas as colunas do df
    for col in df.columns:
        if df[col].dtype == 'object':  # Check if the column contains strings
            df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        #For numeric columns with decimal separators as '.' and ','
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    
    # Transformar todas as colunas para float
    for col in df.columns:
      df[col] = df[col].astype(float)

    return df

#Função para plotar o gráfico do arquivo CSV usado para a modelagem
# Título do app
def plotar_graf():
    st.title("Visualização de Dados Espectrais")

    # Carregar o dataframe
    df = read_csv()
    
    # Separar dados espectrais a partir da 4ª coluna
    df_espectros = df.iloc[:, 3:]
    df_espectros = df_espectros.T  # Transpor para ter comprimentos de onda no eixo X
    
    # Gráfico
    st.subheader("Gráfico de Absorbância por Comprimento de Onda")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for column in df_espectros.columns:
        ax.plot(df_espectros.index, df_espectros[column], label=column)
    
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Absorbance")
    ax.grid(True)
    ax.set_xticks(np.arange(0, len(df_espectros.index), 10))
    # ax.legend()  # Descomente se quiser legenda
    st.pyplot(fig)

#função para plotar RMSE
# Título do app
def plotar_rmse():

    results_df = pd.read_csv('resultados.csv')
    # Ordenar pelo RMSE (Predito)
    results_df = results_df.sort_values(by='RMSE (Predito)', ascending=True).reset_index(drop=True)
    
    #st.title("Comparação de RMSE - Modelos e Normalizações")
    
    # Combinar as colunas 'Modelo' e 'Normalização'
    results_df['Modelo_Normalização'] = results_df['Modelo'] + ' (' + results_df['Normalização'] + ')'
    
    # Plot
    st.subheader("Gráfico de Comparação de RMSE (Treino vs Teste)")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    X_axis = np.arange(len(results_df))
    
    ax.bar(X_axis - 0.2, results_df['RMSE (Treino)'], 0.4, label='RMSE (Treino)')
    ax.bar(X_axis + 0.2, results_df['RMSE (Predito)'], 0.4, label='RMSE (Teste)')
    
    ax.set_xticks(X_axis)
    ax.set_xticklabels(results_df['Modelo_Normalização'], rotation=45, ha='right')
    ax.set_ylabel('RMSE')
    #ax.set_title('Comparação de RMSE (Treino) e RMSE (Teste)')
    ax.legend()
    fig.tight_layout()
    
    st.pyplot(fig)

#Plotar scatterplot Real x Predito
def plotar_realpredito():
    # Título do app
    st.title("Real x Predito")
    
    df_svr_nao_normalizado = pd.read_csv('df_pred_real_svr.csv')
    
    # Criar a figura com subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Gráfico 1 - xAgua
    axs[0].scatter(df_svr_nao_normalizado['xAgua_test'], df_svr_nao_normalizado['xAgua_pred'])
    axs[0].plot(
        [df_svr_nao_normalizado['xAgua_test'].min(), df_svr_nao_normalizado['xAgua_test'].max()],
        [df_svr_nao_normalizado['xAgua_test'].min(), df_svr_nao_normalizado['xAgua_test'].max()],
        'k--', lw=2
    )
    axs[0].set_xlabel('xAgua (Test)')
    axs[0].set_ylabel('xAgua (Pred)')
    axs[0].set_title('xAgua')
    
    # Gráfico 2 - xEtanol
    axs[1].scatter(df_svr_nao_normalizado['xEtanol_test'], df_svr_nao_normalizado['xEtanol_pred'])
    axs[1].plot(
        [df_svr_nao_normalizado['xEtanol_test'].min(), df_svr_nao_normalizado['xEtanol_test'].max()],
        [df_svr_nao_normalizado['xEtanol_test'].min(), df_svr_nao_normalizado['xEtanol_test'].max()],
        'k--', lw=2
    )
    axs[1].set_xlabel('xEtanol (Test)')
    axs[1].set_ylabel('xEtanol (Pred)')
    axs[1].set_title('xEtanol')
    
    # Gráfico 3 - xDEC
    axs[2].scatter(df_svr_nao_normalizado['xDEC_test'], df_svr_nao_normalizado['xDEC_pred'])
    axs[2].plot(
        [df_svr_nao_normalizado['xDEC_test'].min(), df_svr_nao_normalizado['xDEC_test'].max()],
        [df_svr_nao_normalizado['xDEC_test'].min(), df_svr_nao_normalizado['xDEC_test'].max()],
        'k--', lw=2
    )
    axs[2].set_xlabel('xDEC (Test)')
    axs[2].set_ylabel('xDEC (Pred)')
    axs[2].set_title('xDEC')
    
    # Ajustar layout e exibir no Streamlit
    plt.tight_layout()
    st.pyplot(fig)

# Sidebar como menu
st.sidebar.title("Menu")
page = st.sidebar.selectbox("Escolha a página:", ["Início", "Preparação dos Dados", "Modelagem", "Resultados", "Novas Predições"])

# Conteúdo principal muda conforme o menu
if page == "Início":
    st.title("Início")
    st.write("Foram analisadas 75 amostras, pré-processadas com Savitzky-Golay, 7 pontos.")

    df = read_csv()
    df

    plotar_graf()
    

elif page == "Preparação dos Dados":
    st.title("Preparação dos Dados")
    st.write("Os dados já estavam pré-processados usando Savitzky-Golay com 7 pontos. O conjunto de dados foi então dividido em dados de treino (80%) e teste (20%).")

elif page == "Modelagem":
    st.title("Modelagem")
    st.write("A Modelagem foi realizada com os algoritmos PLS, KNN, Random Forest e SVR.")
    st.write("Além disso, com exceção do Random Forest, os demais algoritmos foram treinados com e sem Normalização (média em 0 e desvio-padrão 1), para comparação de resultados.")
    st.write("Para o treino dos dados, foi utilizada validação cruzada, com 5 folds.")
    st.write("Foi utilizada Otimização Bayesiana para otimizar os hiperparâmetros dos modelos")
    st.title("Avaliação")
    st.write("Foram utilizadas as métricas RMSE e R², avaliadas no conjunto de treino e teste.")

elif page == "Resultados":
    st.title("Resultados")
    st.subheader("Métricas")
    results_df = pd.read_csv('resultados.csv')
    results_df

    plotar_rmse()

    plotar_realpredito()


    
# Você pode adicionar sliders, selects, gráficos, etc.
elif page == "Novas Predições":
    st.title("Novas Predições")
    st.write("Aplicação de regressão usando Streamlit.")
    # Carregar o modelo PLS
    model = load_model()
    
    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Carregue o arquivo CSV com os dados de entrada", type=["csv"])
    
    if uploaded_file is not None:
        # Processar o arquivo CSV
        df = process_csv(uploaded_file)
        st.write("Dados carregados:")
        st.write(df)
    
        X = df.values
        #y_pred = model.predict(X) # se não for normalizado
    
        #########se for normalizado
        obj = joblib.load('pls_normalizado.joblib')
        model = obj['model']
        scaler_X = obj['scaler_X']
        scaler_y = obj['scaler_y']
        
        X_input = scaler_X.transform(X)
        y_pred = model.predict(X_input)
        y_pred = scaler_y.inverse_transform(y_pred)
        #########
    
        # Plotar espectro no Streamlit
        st.subheader("Gráfico do Espectro ")
        
        step = 5
        x_values = np.arange(0, len(df.columns), step)
        x_labels = [str(i) for i in range(0, len(df.columns), step)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.iloc[0, :])
        
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        
        ax.grid(True)
        ax.set_title("Espectro")
        ax.set_xlabel("Absorvância")
        ax.set_ylabel("Wavelength (nm)")
        
        plt.tight_layout()
        
        # Exibir o gráfico no Streamlit
        st.pyplot(fig)
    
         # Exibir previsões
        st.subheader("Previsões feitas pelo modelo (xÁgua, xEtanol, xDEC):")
        st.dataframe(pd.DataFrame(y_pred, columns=['xAgua_pred', 'xEtanol_pred', 'xDEC_pred']))
    
    
