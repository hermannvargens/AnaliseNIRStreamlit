import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Função para carregar o modelo PLS salvo
@st.cache_resource

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
def plotar_rmse(linhas, results_df):

    #results_df = pd.read_csv('resultados.csv')
    # Ordenar pelo RMSE (Predito)
    results_df = results_df.iloc[linhas]
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
def plotar_realpredito(df):

    # Criar a figura com subplots
    fig, axs = plt.subplots(1,3, figsize=(30,15))

    # Tamanhos de fonte personalizados
    title_fontsize = 40
    label_fontsize = 36
    tick_fontsize = 32
    point_size = 200  # Tamanho dos pontos do scatter
    
    # Gráfico 1 - xAgua
    axs[0].scatter(df['xAgua_test'], df['xAgua_pred'], s=point_size)
    axs[0].plot(
        [df['xAgua_test'].min(), df['xAgua_test'].max()],
        [df['xAgua_test'].min(), df['xAgua_test'].max()],
        'k--', lw=2
    )
    axs[0].set_xlabel('xAgua (Test)', fontsize=label_fontsize)
    axs[0].set_ylabel('xAgua (Pred)', fontsize=label_fontsize)
    axs[0].set_title('xAgua', fontsize=title_fontsize)
    axs[0].tick_params(axis='both', labelsize=tick_fontsize)
    
    # Gráfico 2 - xEtanol
    
    axs[1].scatter(df['xEtanol_test'], df['xEtanol_pred'], s=point_size )
    axs[1].plot(
        [df['xEtanol_test'].min(), df['xEtanol_test'].max()],
        [df['xEtanol_test'].min(), df['xEtanol_test'].max()],
        'k--', lw=2
    )
    axs[1].set_xlabel('xEtanol (Test)', fontsize=label_fontsize)
    axs[1].set_ylabel('xEtanol (Pred)', fontsize=label_fontsize)
    axs[1].set_title('xEtanol', fontsize=title_fontsize)
    axs[1].tick_params(axis='both', labelsize=tick_fontsize)
    
    # Gráfico 3 - xDEC
    
    axs[2].scatter(df['xDEC_test'], df['xDEC_pred'], s=point_size)
    axs[2].plot(
        [df['xDEC_test'].min(), df['xDEC_test'].max()],
        [df['xDEC_test'].min(), df['xDEC_test'].max()],
        'k--', lw=2
    )
    axs[2].set_xlabel('xDEC (Test)', fontsize=label_fontsize)
    axs[2].set_ylabel('xDEC (Pred)', fontsize=label_fontsize)
    axs[2].set_title('xDEC', fontsize=title_fontsize)
    axs[2].tick_params(axis='both', labelsize=tick_fontsize)
    
    # Ajustar layout e exibir no Streamlit
    plt.tight_layout()
    st.pyplot(fig)

def plot_residuos(df):

    # Tamanhos de fonte personalizados
    title_fontsize = 40
    label_fontsize = 36
    tick_fontsize = 32
    point_size = 200  # Tamanho dos pontos do scatter
    
    # Calcular resíduos
    residuals_agua = df['xAgua_test'] - df['xAgua_pred']
    residuals_etanol = df['xEtanol_test'] - df['xEtanol_pred']
    residuals_dec = df['xDEC_test'] - df['xDEC_pred']
    
    # Criar figura com subplots
    fig, axs = plt.subplots(1,3, figsize=(30,15))
    
    # Função auxiliar para definir limites simétricos do grafico
    def set_symmetric_ylim(ax, residuals):
        limit = max(abs(residuals.min()), abs(residuals.max()))
        ax.set_ylim(-limit, limit)
    
    # Gráfico 1 - Resíduos xÁgua
    axs[0].scatter(df['xAgua_test'], residuals_agua, s=point_size)
    axs[0].axhline(0, color='red', linestyle='--')
    axs[0].set_xlabel("Valor Real (Água)", fontsize=label_fontsize)
    axs[0].set_ylabel("Resíduo", fontsize=label_fontsize)
    axs[0].set_title("Resíduos - Água", fontsize=title_fontsize)
    axs[0].tick_params(axis='both', labelsize=tick_fontsize)
    set_symmetric_ylim(axs[0], residuals_agua)
    
    # Gráfico 2 - Resíduos xEtanol
    axs[1].scatter(df['xEtanol_test'], residuals_etanol, s=point_size)
    axs[1].axhline(0, color='red', linestyle='--')
    axs[1].set_xlabel("Valor Real (Etanol)", fontsize=label_fontsize)
    axs[1].set_ylabel("Resíduo", fontsize=label_fontsize)
    axs[1].set_title("Resíduos - Etanol", fontsize=title_fontsize)
    axs[1].tick_params(axis='both', labelsize=tick_fontsize)
    set_symmetric_ylim(axs[1], residuals_etanol)
    
    # Gráfico 3 - Resíduos xDEC
    axs[2].scatter(df['xDEC_test'], residuals_dec, s=point_size)
    axs[2].axhline(0, color='red', linestyle='--')
    axs[2].set_xlabel("Valor Real (DEC)", fontsize=label_fontsize)
    axs[2].set_ylabel("Resíduo", fontsize=label_fontsize)
    axs[2].set_title("Resíduos - DEC", fontsize=title_fontsize)
    axs[2].tick_params(axis='both', labelsize=tick_fontsize)
    set_symmetric_ylim(axs[2], residuals_dec)
    
    plt.tight_layout()
    st.pyplot(fig)

def plotar_espectro_carregado(df):
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


#######################################################
# Sidebar como menu#
#######################################################
st.sidebar.title("Menu")
page = st.sidebar.selectbox("Escolha a página:", ["Início", "Preparação dos Dados", "Métodos", "Resultados Iniciais", "Resultados Finais", "Novas Predições"])

# Conteúdo principal muda conforme o menu
if page == "Início":
    st.title("Início")
    st.write("Foram analisadas 75 amostras, pré-processadas com Savitzky-Golay, 7 pontos.")

    df = read_csv()
    

    plotar_graf()
    

elif page == "Preparação dos Dados":
    st.title("Preparação dos Dados")
    st.write("Os dados já estavam pré-processados usando Savitzky-Golay com 7 pontos. O conjunto de dados foi então dividido em dados de treino (80%) e teste (20%).")

elif page == "Métodos":
    st.title("Métodos")
    st.write("A Modelagem foi realizada com os algoritmos PLS, KNN, Random Forest, SVR e um meta-modelo do tipo Stacking, com os dois melhores modelos.")
    st.write("Além disso, com exceção do Random Forest, os demais algoritmos foram treinados com e sem Normalização (média em 0 e desvio-padrão 1), para comparação de resultados.")
    st.write("Para o treino dos dados, foi utilizada validação cruzada, com 5 folds.")
    st.write("Foi utilizada Otimização Bayesiana para encontrar os melhores hiperparâmetros dos modelos")
    st.title("Avaliação")
    st.write("Foram utilizadas as métricas RMSE e R², avaliadas no conjunto de treino e teste.")

elif page == "Resultados Iniciais":
    st.title("Resultados")
    st.subheader("Métricas")
    results_df = pd.read_csv('resultados.csv')
    st.table(results_df.head(6))

    #Plotar RMSE de todos os modelos
    results_df = pd.read_csv('resultados.csv')
    linhas=[0,1,2,3,4,5]
    plotar_rmse(linhas,results_df)

    st.write("Em detalhe os modelos com menores RMSE:")
    
    #Plotar RMSE apenas do SVR e PLS
    results_df = pd.read_csv('resultados.csv')
    linhas=[0,4]
    plotar_rmse(linhas,results_df)

    st.write("Vemos que os modelos SVR e PLS não normalizados possuem os menores valores de RMSE.")


elif page == "Resultados Finais":
    st.title("Resultados Finais")
    st.subheader("Stacking Regressor")
    st.write("Uma vez escolhidos os modelos com melhores resultados, tentamos realizar um ensemble, do tipo Stacking, que consiste em realizar um novo ajuste unindo os modelos PLS e SVR em um modelo só, ajustados através de uma Regressão Linear, com a expectativa de que tal modelo possa apresentar resultados ainda melhores.")

    results_df = pd.read_csv('resultados.csv')
    linhas=[0,4,6]
    plotar_rmse(linhas,results_df)

    st.write("Vemos que o meta-modelo apresentou resultados ligeiramente melhores que os demais modelos isolados.")

    # Título do app
    st.subheader("SVR: Real x Predito")
    
    df_svr = pd.read_csv('df_pred_real_svr.csv')
    plotar_realpredito(df_svr)

    # Título do app
    st.subheader("PLS: Real x Predito")

    df_pls = pd.read_csv('df_pred_real_pls.csv')
    plotar_realpredito(df_pls)

    # Título do app
    st.subheader("Stacking: Real x Predito")

    df_stacking = pd.read_csv('df_pred_real_stacking.csv')
    plotar_realpredito(df_stacking)

    st.subheader("Análise de Resíduos - SVR")
    plot_residuos(df_svr)

    st.subheader("Análise de Resíduos - PLS")
    plot_residuos(df_pls)

    st.subheader("Análise de Resíduos - Stacking")
    plot_residuos(df_stacking)


    

elif page == "Novas Predições":

    svr = joblib.load('svr_nao_normalizado.joblib')
    pls = joblib.load('pls_normalizado.joblib')

    # Título do app
    st.subheader("Faça o upload de arquivo CSV contendo o espectro da amostra para realizar uma nova predição.")
    st.write("Se não tiver, clique aqui para baixar um exemplo!")
    st.markdown(
    """
    👉 [Clique aqui para baixar o arquivo](https://raw.githubusercontent.com/hermannvargens/AnaliseNIRStreamlit/refs/heads/main/amostra0.csv)
    """,
    unsafe_allow_html=True
)
    # Upload da amostra
    uploaded_file = st.file_uploader("Envie a amostra espectral (.csv)", type="csv")

    if uploaded_file is not None:
        df = process_csv(uploaded_file)
        st.write("Dados carregados:")
        st.write(df)
        plotar_espectro_carregado(df)


        def prever_svr():
            X = df.values
            y_pred = svr.predict(X)
            return pd.DataFrame(y_pred, columns=['xAgua_pred', 'xEtanol_pred', 'xDEC_pred'])
    
        def prever_pls():
            X = df.values
    
    #########normalizado
            obj = joblib.load('pls_normalizado.joblib')
            model = obj['model']
            scaler_X = obj['scaler_X']
            scaler_y = obj['scaler_y']
            
            X_input = scaler_X.transform(X)
            y_pred = model.predict(X_input)
            y_pred = scaler_y.inverse_transform(y_pred)
            #########
    
            return pd.DataFrame(y_pred, columns=['xAgua_pred', 'xEtanol_pred', 'xDEC_pred'])
        
        st.write("Previsões com SVR")
            
        resultado_svr = prever_svr()
        resultado_svr
    
        st.write("Previsões com PLS")
        resultado_pls = prever_pls()
        resultado_pls

           
         #########


    
        
            
            
