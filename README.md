# 🏠 Avaliador Imobiliário com XGBoost + IA Local (Hermes 3)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6F00?logo=xgboost&logoColor=white)](https://xgboost.ai)
[![Scikit‑learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Simulador interativo que utiliza **XGBoost** para prever o preço de imóveis a partir de dados reais do mercado de Indiana, permitindo simular o impacto de diferentes características (área, garagem, idade) e obter explicações automáticas via **IA local (Hermes 3)**.

---

## 📌 Sobre o Projeto

Este projeto demonstra um pipeline completo de **modelagem preditiva** e **simulação de cenários** aplicado ao mercado imobiliário. Com base no dataset **Indiana Real Estate 2026**, um modelo **XGBoost** é treinado para prever o preço (`listPrice`) de imóveis residenciais e comerciais.

A interface **Streamlit** permite que o usuário ajuste atributos do imóvel (área, quartos, garagem, idade, tipo), visualize a previsão instantânea e explore a **curva de elasticidade** variando a área. Além disso, um agente de IA local (Hermes 3) analisa o cenário atual e gera insights em português, explicando como as variáveis influenciam o resultado.

---

## 🛠️ Stack Principal

| **Linguagem** | **Bibliotecas de ML** | **IA Local** | **Visualização** |
|---------------|-----------------------|--------------|------------------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | ![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | ![LM Studio](https://img.shields.io/badge/LM_Studio-0A0A0A?style=flat-square) ![Hermes 3](https://img.shields.io/badge/Hermes_3-FFD700?style=flat-square) | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=python&logoColor=white) ||

---

## 📊 Dataset

**Fonte:** [Indiana Real Estate Data 2026 – Kaggle](https://www.kaggle.com/datasets/kanchana1990/indiana-real-estate-data-2026)  
**Descrição:**
- 13.532 imóveis listados no primeiro trimestre de 2026.
- Atributos: tipo de imóvel (`type`), área (`sqft`), andares (`stories`), quartos (`beds`), banheiros (`baths`), garagem (`garage`), ano de construção (`year_built`) e texto descritivo (`text`).
- **Objetivo:** prever o preço de venda (`listPrice`).

---

## 🧠 Como funciona

1. **Upload ou Download Automático**  
   - O usuário pode enviar o arquivo CSV ou baixá‑lo automaticamente via `kagglehub`.

2. **Pré‑processamento e Feature Engineering**  
   - Tratamento de valores ausentes, remoção de outliers (1% superior) e criação da feature `age = 2026 – year_built`.
   - Extração de features textuais via **TF‑IDF** da coluna `text` (50 termos mais relevantes).

3. **Treinamento do XGBoost**  
   - Parâmetros ajustáveis (n_estimators, learning_rate, max_depth).
   - Exibição das métricas de avaliação: **R², MAE, RMSE**.

4. **Simulador Streamlit**  
   - Interface com controles para todos os atributos (sliders, selects).
   - A cada alteração, recalcula a previsão e exibe um gráfico de **elasticidade** (preço × área).

5. **IA Explicativa**  
   - Botão **“Analisar com IA”** envia os dados atuais para o **Hermes 3** (via LM Studio) e retorna uma análise em português sobre os fatores que mais impactaram o preço.

---

## 📈 5 Perguntas de Negócio que o Sistema Responde

1. **Qual o preço estimado de um imóvel dadas suas características?**  
   → Ajuste os atributos (área, quartos, garagem, idade, tipo) e obtenha a previsão instantânea.

2. **Quais fatores mais impactam o valor do imóvel?**  
   → O modelo XGBoost calcula a importância das variáveis; o Hermes 3 gera uma explicação em português destacando os principais direcionadores de valor.

3. **Como a área (sqft) influencia o preço?**  
   → O gráfico de elasticidade mostra a curva preço × área, permitindo visualizar o retorno marginal de aumentar os metros quadrados.

4. **Qual o impacto de uma garagem ou da idade do imóvel na avaliação?**  
   → Simule diferentes números de vagas e anos de construção; o Hermes 3 quantifica em linguagem natural o peso desses atributos.

5. **O texto descritivo do anúncio agrega valor à previsão?**  
   → A extração TF‑IDF captura termos como “renovado”, “lago”, “barn”, e o modelo incorpora essas informações, melhorando a acurácia.

---

## 🚀 Como Executar

### Pré‑requisitos
- Python 3.9+
- **LM Studio** com modelo **Hermes 3** carregado e servidor ativo (porta 1234)
- Conta Kaggle (para download automático do dataset – opcional)

### Passo a passo

1. **Clone o repositório**
   ```bash
   git clone https://github.com/Gussnogue/xgboost-real-state-hermes-ai-local.git
   cd xgboost-real-state-hermes-ai-local

2. **Crie e ative um ambiente virtual**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Baixe o dataset**
   ```bash
   acesse Kaggle, clique em Download e coloque o arquivo indiana_real_estate_2026.csv na pasta data/.
   ```

5. **Execute a interface**
   ```bash
   streamlit run app_integrado.py
   ```

# Referências
Dataset: Indiana Real Estate Data 2026 disponibilizado por Kanchana1990 no Kaggle.

**Fonte:** extraído de registros públicos de Indiana (Q1 2026) e anonimizado em duas etapas para remoção de identificadores geográficos e PII. 
- https://www.kaggle.com/datasets/kanchana1990/indiana-real-estate-data-2026

**Modelos Locais:**
Hermes 3 – [Nous Research](https://nousresearch.com/)
