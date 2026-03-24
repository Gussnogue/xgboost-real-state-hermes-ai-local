import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import joblib
import warnings
warnings.filterwarnings('ignore')

# ========== IMPORTAÇÕES DE ML ==========
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ========== MÓDULO NLP (funções locais para não depender de arquivo externo) ==========
def extract_tfidf_features(texts, max_features=100, vectorizer=None, fit=True):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if fit:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', min_df=2, max_df=0.8)
        X = vectorizer.fit_transform(texts).toarray()
        return X, vectorizer
    else:
        X = vectorizer.transform(texts).toarray()
        return X, vectorizer

# ========== CONFIGURAÇÕES ==========
st.set_page_config(page_title="Avaliador Imobiliário - Treinamento Integrado", layout="wide")
st.title("🏠 Avaliador Imobiliário - Treinamento Integrado")
st.markdown("Faça upload do dataset, treine o modelo e use para previsões. Tudo em um só lugar.")

# ========== 1. SEÇÃO DE DADOS ==========
st.header("📂 1. Dados")
uploaded_file = st.file_uploader("Envie o arquivo CSV (indiana_real_estate_2026.csv)", type=["csv"])

if uploaded_file is not None:
    # Carrega o dataset
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset carregado! {df.shape[0]} registros, {df.shape[1]} colunas.")
    st.dataframe(df.head())
else:
    st.info("👆 Envie o arquivo CSV baixado do Kaggle (ou clique no botão abaixo para baixar automaticamente).")
    if st.button("📥 Baixar dataset do Kaggle automaticamente"):
        try:
            import kagglehub
            with st.spinner("Baixando dataset (pode demorar um pouco)..."):
                path = kagglehub.dataset_download("kanchana1990/indiana-real-estate-data-2026")
                # Localiza o CSV
                for file in os.listdir(path):
                    if file.endswith(".csv"):
                        df = pd.read_csv(os.path.join(path, file))
                        st.success(f"✅ Dataset baixado com sucesso! {df.shape[0]} registros.")
                        break
                else:
                    st.error("Nenhum arquivo CSV encontrado no download.")
        except ImportError:
            st.error("Módulo kagglehub não instalado. Execute: pip install kagglehub")
            st.info("Ou baixe manualmente do Kaggle e faça upload.")
        except Exception as e:
            st.error(f"Erro no download: {e}")
    st.stop()  # impede execução se não tiver dados

# Se chegou aqui, df está disponível
# ========== 2. TREINAMENTO ==========
st.header("🚀 2. Treinamento do Modelo")
with st.expander("⚙️ Configurações do modelo (opcional)", expanded=False):
    n_estimators = st.slider("Número de árvores (n_estimators)", 50, 500, 200)
    learning_rate = st.slider("Taxa de aprendizado", 0.01, 0.3, 0.05, step=0.01)
    max_depth = st.slider("Profundidade máxima", 3, 10, 6)
    use_text = st.checkbox("Usar texto (TF‑IDF)", value=True, help="Extrai 50 features da coluna 'text'")

if st.button("🏋️ Treinar Modelo"):
    with st.spinner("Pré-processando dados..."):
        # Pré-processamento (similar ao train.py)
        cols_features = ['type', 'sqft', 'stories', 'beds', 'baths', 'baths_full', 'baths_full_calc', 'garage', 'year_built']
        target = 'listPrice'

        # Trata nulos
        for col in cols_features:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            elif col in df.columns:
                df[col].fillna('missing', inplace=True)

        # Remove outliers de preço
        q99 = df[target].quantile(0.99)
        df = df[df[target] <= q99]

        # Codifica type
        le = LabelEncoder()
        df['type'] = le.fit_transform(df['type'].astype(str))

        # Feature engineering
        df['age'] = 2026 - df['year_built']

        # Features textuais
        tfidf_features = None
        if use_text and 'text' in df.columns:
            texts = df['text'].fillna('').tolist()
            tfidf_features, tfidf_vec = extract_tfidf_features(texts, max_features=50, fit=True)
            tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
            df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

        # Seleciona features
        feature_cols = ['type', 'sqft', 'stories', 'beds', 'baths', 'baths_full', 'baths_full_calc', 'garage', 'age']
        if use_text and tfidf_features is not None:
            feature_cols.extend([f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

        X = df[feature_cols]
        y = df[target]

        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("Treinando XGBoost..."):
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

    # Avaliação
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Salva artefatos na sessão
    st.session_state['model'] = model
    st.session_state['feature_cols'] = feature_cols
    st.session_state['label_encoder'] = le
    st.session_state['tfidf_vec'] = tfidf_vec if use_text else None
    st.session_state['use_text'] = use_text

    st.success("✅ Modelo treinado com sucesso!")
    st.metric("R² (teste)", f"{r2:.4f}")
    st.metric("MAE", f"${mae:,.2f}")
    st.metric("RMSE", f"${rmse:,.2f}")

# ========== 3. SEÇÃO DE PREVISÃO ==========
if 'model' in st.session_state:
    st.header("🔮 3. Faça uma previsão")
    model = st.session_state['model']
    feature_cols = st.session_state['feature_cols']
    le = st.session_state['label_encoder']
    tfidf_vec = st.session_state['tfidf_vec']
    use_text = st.session_state['use_text']

    # Interface de entrada
    with st.sidebar:
        st.header("📋 Características do imóvel")
        tipo = st.selectbox("Tipo de imóvel", options=le.classes_)
        tipo_encoded = le.transform([tipo])[0]
        sqft = st.number_input("Área (sqft)", 0.0, 10000.0, 2000.0)
        stories = st.number_input("Andares", 0.0, 5.0, 1.0)
        beds = st.number_input("Quartos", 0, 10, 3)
        baths = st.number_input("Banheiros (total)", 0.0, 10.0, 2.0)
        baths_full = st.number_input("Banheiros completos", 0.0, 10.0, 2.0)
        baths_full_calc = st.number_input("Banheiros completos (calc)", 0.0, 10.0, 2.0)
        garage = st.number_input("Garagem (vagas)", 0, 10, 2)
        year_built = st.number_input("Ano de construção", 1800, 2026, 2000)
        age = 2026 - year_built

        if use_text:
            user_text = st.text_area("Descrição do imóvel (opcional)", height=150)

    # Botão prever
    if st.button("Calcular Preço"):
        # Monta vetor de entrada
        base = {
            'type': tipo_encoded,
            'sqft': sqft,
            'stories': stories,
            'beds': beds,
            'baths': baths,
            'baths_full': baths_full,
            'baths_full_calc': baths_full_calc,
            'garage': garage,
            'age': age
        }
        if use_text and tfidf_vec is not None:
            if user_text.strip():
                text_features, _ = extract_tfidf_features([user_text], vectorizer=tfidf_vec, fit=False)
                for i, val in enumerate(text_features[0]):
                    base[f'tfidf_{i}'] = val
            else:
                for col in feature_cols:
                    if col.startswith('tfidf_'):
                        base[col] = 0.0

        X_input = np.array([base[col] for col in feature_cols]).reshape(1, -1)
        pred = model.predict(X_input)[0]
        st.success(f"💰 Preço estimado: **${pred:,.2f}**")

        # Gráfico de elasticidade (preço x área)
        st.subheader("📈 Impacto da Área no Preço")
        sqft_range = np.linspace(500, 5000, 50)
        prices = []
        for s in sqft_range:
            base['sqft'] = s
            X_input = np.array([base[col] for col in feature_cols]).reshape(1, -1)
            prices.append(model.predict(X_input)[0])
        fig, ax = plt.subplots()
        ax.plot(sqft_range, prices, color='blue')
        ax.set_xlabel('Área (sqft)')
        ax.set_ylabel('Preço estimado (USD)')
        ax.grid(True)
        st.pyplot(fig)

else:
    st.info("👆 Treine o modelo primeiro (seção 2).")

    