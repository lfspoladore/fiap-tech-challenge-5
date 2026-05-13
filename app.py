import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import __main__

# ===============================
# TRANSFORMER CUSTOMIZADO (necessário para carregar o pipeline)
# ===============================
class PassosMagicosTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # Garante tratamento de Fase que pode vir como string do input ('ALFA', 'FASE 1')
        def clean_fase(x):
            if pd.isna(x): return 0
            x = str(x).upper()
            if 'ALFA' in x: return 0
            import re
            digits = re.findall(r'\d+', x)
            if digits: return int(digits[0])
            return 0
        if 'fase' in X_out.columns:
            X_out['fase'] = X_out['fase'].apply(clean_fase)

        # Garante que features numéricas sejam tratadas como numéricas
        numeric_cols = [c for c in ['iaa', 'ieg', 'ips', 'ida', 'ipp', 'ipv', 'ian'] if c in X_out.columns]
        for col in numeric_cols:
            X_out[col] = pd.to_numeric(X_out[col], errors='coerce')
            
        # O Streamlit já envia os deltas calculados da UI, mas garantimos a consistência das colunas
        cols_esperadas = ['iaa', 'ieg', 'ips', 'ida', 'ipp', 'ipv', 'ian', 'idade_aprox', 'fase', 
                          'delta_iaa', 'delta_ieg', 'delta_ips', 'delta_ida', 'delta_ipp', 'delta_ipv', 'delta_ian', 
                          'instituicao_de_ensino', 'genero']
        for col in cols_esperadas:
            if col not in X_out.columns:
                X_out[col] = 0.0 # Fallback de segurança para features ausentes
        
        return X_out

# Injeta a classe no namespace __main__ para o joblib conseguir desempacotar o pickle no Streamlit
__main__.PassosMagicosTransformer = PassosMagicosTransformer

# ===============================
# CARREGAR PIPELINE
# ===============================
@st.cache_resource
def load_artifacts():
    # Carrega do diretório de artefatos
    pipeline = joblib.load("model_artifacts/pipeline.joblib")
    return pipeline

pipeline = load_artifacts()

# ===============================
# FUNÇÕES SHAP
# ===============================
def get_shap_values(pipeline_model, X_df):
    preprocessor = pipeline_model.named_steps['preprocessor']
    classifier = pipeline_model.named_steps['classifier']
    
    X_transformed = preprocessor.transform(X_df)
    feature_names = preprocessor.get_feature_names_out()
    
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
        
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    # --- MAPEAMENTO PARA NOMES GERENCIAIS ---
    mapping = {
        'num__iaa': 'Autoavaliação (IAA)',
        'num__ieg': 'Engajamento (IEG)',
        'num__ips': 'Psicossocial (IPS)',
        'num__ida': 'Desempenho Acadêmico (IDA)',
        'num__ipp': 'Psicopedagógico (IPP)',
        'num__ipv': 'Ponto de Virada (IPV)',
        'num__ian': 'Adequação de Nível (IAN)',
        'num__idade_aprox': 'Idade do Aluno',
        'num__fase': 'Fase Escolar',
        'num__delta_iaa': 'Evolução IAA (T vs T-1)',
        'num__delta_ieg': 'Evolução IEG (T vs T-1)',
        'num__delta_ips': 'Evolução IPS (T vs T-1)',
        'num__delta_ida': 'Evolução IDA (T vs T-1)',
        'num__delta_ipp': 'Evolução IPP (T vs T-1)',
        'num__delta_ipv': 'Evolução IPV (T vs T-1)',
        'num__delta_ian': 'Evolução IAN (T vs T-1)',
        'cat__instituicao_de_ensino_Privada': 'Ensino Privado',
        'cat__instituicao_de_ensino_Outros': 'Ensino (Outros)',
        'cat__genero_Feminino': 'Gênero Feminino',
        'cat__genero_Masculino': 'Gênero Masculino'
    }
    # Tratando acentuação separadamente por precaução de encoding no get_feature_names_out
    for col in X_transformed_df.columns:
        if 'Pública' in col or 'Pblica' in col:
            mapping[col] = 'Ensino Público'

    X_transformed_df = X_transformed_df.rename(columns=mapping)
    # ----------------------------------------
    
    if type(classifier).__name__ in ['RandomForestClassifier', 'GradientBoostingClassifier']:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
    else:
        explainer = shap.Explainer(classifier, X_transformed_df)
        shap_values = explainer(X_transformed_df).values
        
    return explainer, shap_values, X_transformed_df

# ===============================
# INTERFACE E LAYOUT
# ===============================
st.set_page_config(page_title="Predição de Risco - Passos Mágicos", layout="wide")

st.title("Sistema de Predição de Risco Educacional (T+1)")
st.markdown("Insira os dados do aluno para prever a probabilidade dele sofrer queda de desempenho acadêmico (INDE) ou aumento de defasagem de nível no **próximo ano letivo**.")

tab1, tab2 = st.tabs(["Previsão Individual", "Previsão em Lote (Upload CSV)"])

with tab1:
    # -------- DADOS ESCOLARES --------
    st.header("1. Dados Demográficos e Escolares")
    col1, col2, col3 = st.columns(3)

    with col1:
        idade = st.number_input("Idade Aproximada", min_value=5, max_value=25, value=15, step=1)
    with col2:
        fase = st.number_input("Fase Atual do Aluno (0=Alfa a 8=Universitário)", min_value=0, max_value=8, value=3, step=1)
    with col3:
        genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
        inst_ensino = st.selectbox("Instituição de Ensino", ["Pública", "Privada", "Outros"])

    # -------- INDICADORES --------
    st.header("2. Indicadores de Desempenho")
    st.markdown("Insira as notas (0 a 10) do ano letivo atual (T) e do ano anterior (T-1) para que o sistema calcule as tendências dinâmicas ($\Delta$). Se for o primeiro ano do aluno na ONG, repita as notas em ambos os lados.")

    col_atual, col_anterior = st.columns(2)

    with col_atual:
        st.subheader("Notas do Ano Atual (T)")
        inde_atual = st.slider("INDE (Índice de Desenvolvimento Educacional) - T", 0.0, 10.0, 7.5, step=0.1)
        ida_atual = st.slider("IDA (Indicador de Desempenho Acadêmico) - T", 0.0, 10.0, 7.0, step=0.1)
        ieg_atual = st.slider("IEG (Indicador de Engajamento) - T", 0.0, 10.0, 8.0, step=0.1)
        iaa_atual = st.slider("IAA (Indicador de Autoavaliação) - T", 0.0, 10.0, 8.5, step=0.1)
        ips_atual = st.slider("IPS (Indicador Psicossocial) - T", 0.0, 10.0, 7.5, step=0.1)
        ipp_atual = st.slider("IPP (Indicador Psicopedagógico) - T", 0.0, 10.0, 7.5, step=0.1)
        ipv_atual = st.slider("IPV (Indicador de Ponto de Virada) - T", 0.0, 10.0, 7.0, step=0.1)
        ian_atual = st.slider("IAN (Indicador de Adequação de Nível) - T", 0.0, 10.0, 8.0, step=0.1)

    with col_anterior:
        st.subheader("Notas do Ano Anterior (T-1)")
        inde_ant = st.slider("INDE Anterior - T-1", 0.0, 10.0, 7.0, step=0.1)
        ida_ant = st.slider("IDA Anterior - T-1", 0.0, 10.0, 6.5, step=0.1)
        ieg_ant = st.slider("IEG Anterior - T-1", 0.0, 10.0, 7.5, step=0.1)
        iaa_ant = st.slider("IAA Anterior - T-1", 0.0, 10.0, 8.0, step=0.1)
        ips_ant = st.slider("IPS Anterior - T-1", 0.0, 10.0, 7.0, step=0.1)
        ipp_ant = st.slider("IPP Anterior - T-1", 0.0, 10.0, 7.0, step=0.1)
        ipv_ant = st.slider("IPV Anterior - T-1", 0.0, 10.0, 6.5, step=0.1)
        ian_ant = st.slider("IAN Anterior - T-1", 0.0, 10.0, 7.5, step=0.1)

    # ===============================
    # PREDIÇÃO (MACHINE LEARNING)
    # ===============================
    st.markdown("---")

    if st.button("Realizar Diagnóstico Preditivo (T+1)", type="primary"):

        # 1. Calculando os Deltas de Tendência (Ano atual - Ano Anterior)
        delta_ida = ida_atual - ida_ant
        delta_ieg = ieg_atual - ieg_ant
        delta_iaa = iaa_atual - iaa_ant
        delta_ips = ips_atual - ips_ant
        delta_ipp = ipp_atual - ipp_ant
        delta_ipv = ipv_atual - ipv_ant
        delta_ian = ian_atual - ian_ant

        # 2. Construindo o DataFrame com as features exatas esperadas pelo Pipeline
        input_data = pd.DataFrame([{
            'iaa': iaa_atual, 'ieg': ieg_atual, 'ips': ips_atual, 'ida': ida_atual, 'ipp': ipp_atual, 'ipv': ipv_atual, 'ian': ian_atual,
            'idade_aprox': idade, 'fase': fase,
            'delta_iaa': delta_iaa, 'delta_ieg': delta_ieg, 'delta_ips': delta_ips, 'delta_ida': delta_ida, 'delta_ipp': delta_ipp, 'delta_ipv': delta_ipv, 'delta_ian': delta_ian,
            'instituicao_de_ensino': inst_ensino, 'genero': genero
        }])

        # 3. Inferência (O Pipeline aplica Imputer, Scaler e OneHotEncoder automaticamente)
        pred_proba = pipeline.predict_proba(input_data)[0][1] # Probabilidade de ser classe 1 (Risco)
        threshold = 0.35
        pred_class = 1 if pred_proba >= threshold else 0
        
        # Discretização
        if pred_proba < 0.35:
            risco_label = "Baixo"
        elif pred_proba < 0.60:
            risco_label = "Médio"
        else:
            risco_label = "Alto"

        # 4. Exibição do Resultado na Interface
        st.subheader("Resultado do Diagnóstico:")
        st.info(f"**Faixa de Risco:** {risco_label} ({pred_proba*100:.1f}%)")
        
        if pred_class == 1:
            st.error(f"**ALERTA DE RISCO DETECTADO:** O modelo aponta uma alta probabilidade de que este aluno sofrerá queda de performance (INDE) ou aumento da defasagem no ano seguinte. \n\n**Ação Sugerida:** Inclusão prioritária no programa de mentoria e acompanhamento psicopedagógico preventivo.")
        else:
            st.success(f"**MANUTENÇÃO DE EXCELÊNCIA:** O aluno apresenta baixo risco de queda para o próximo ano letivo. A tendência calculada é de crescimento ou sustentação das notas atuais.")

        st.markdown("---")
        st.subheader("Interpretação do Modelo (SHAP)")
        st.markdown("O gráfico abaixo explica a predição para **este aluno específico**. Variáveis em **vermelho** empurram o risco de evasão para cima, enquanto variáveis em **azul** empurram o risco para baixo.")
        
        with st.spinner("Calculando SHAP values..."):
            explainer, shap_values, X_transformed_df = get_shap_values(pipeline, input_data)
            
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
            else:
                base_value = explainer.expected_value
                
            # Usando waterfall plot para evitar sobreposição de texto
            exp = shap.Explanation(values=shap_values[0], 
                                   base_values=base_value, 
                                   data=X_transformed_df.iloc[0], 
                                   feature_names=X_transformed_df.columns)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(exp, show=False)
            st.pyplot(fig)
            plt.clf()

with tab2:
    st.header("Upload de Base de Dados para Predição em Lote")
    st.markdown("Suba um arquivo CSV contendo os dados dos alunos. O arquivo deve conter as mesmas colunas utilizadas no modelo treinado.")
    
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            # Tenta ler com vírgula primeiro
            try:
                df_upload = pd.read_csv(uploaded_file)
            except Exception:
                # Se falhar (ex: erro de tokenização), volta pro começo do arquivo e tenta com ponto e vírgula
                uploaded_file.seek(0)
                df_upload = pd.read_csv(uploaded_file, sep=';')
                
            st.write("Visualização dos primeiros registros da base enviada:")
            st.dataframe(df_upload.head())
            
            if st.button("Executar Predição em Lote", type="primary"):
                with st.spinner('Realizando predições e calculando SHAP Values globais...'):
                    # O Pipeline fará as transformações e predições
                    # As colunas precisam bater com as esperadas pelo pipeline
                    try:
                        # Garante que o DataFrame tem as colunas corretas usando o nosso Transformer
                        transformer = PassosMagicosTransformer()
                        df_upload_processed = transformer.transform(df_upload)
                        
                        probs = pipeline.predict_proba(df_upload_processed)[:, 1]
                        
                        df_resultado = df_upload.copy()
                        df_resultado['Probabilidade_Risco'] = probs
                        
                        # Discretização conforme a nova regra
                        def categorizar_risco(proba):
                            if proba < 0.35:
                                return 'Baixo'
                            elif proba < 0.60:
                                return 'Médio'
                            else:
                                return 'Alto'
                                
                        df_resultado['Faixa_Risco'] = df_resultado['Probabilidade_Risco'].apply(categorizar_risco)
                        
                        st.success("Predições realizadas com sucesso!")
                        st.dataframe(df_resultado[['Probabilidade_Risco', 'Faixa_Risco'] + [col for col in df_resultado.columns if col not in ['Probabilidade_Risco', 'Faixa_Risco']]])
                        
                        # SHAP Global
                        st.markdown("---")
                        st.subheader("Impacto Global das Features (SHAP)")
                        st.markdown("O gráfico abaixo mostra quais variáveis foram mais importantes para as predições de **toda esta base de alunos**.")
                        
                        explainer, shap_values, X_transformed_df = get_shap_values(pipeline, df_upload_processed)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_transformed_df, show=False)
                        st.pyplot(fig)
                        plt.clf()

                        # Botão para download dos resultados
                        csv_resultado = df_resultado.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Baixar Resultados CSV",
                            data=csv_resultado,
                            file_name='predicoes_risco_lote.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        st.error(f"Erro ao processar as predições. Verifique se o CSV contém todas as colunas necessárias. Erro: {e}")
        except Exception as e:
            st.error(f"Erro ao ler o arquivo CSV: {e}")
