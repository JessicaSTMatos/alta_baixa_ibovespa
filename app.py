import os
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px


# ======================================================
# CONFIGURAÇÃO DA PÁGINA
# ======================================================
st.set_page_config(page_title="Ibovespa - Alta/Baixa", layout="wide")
st.title("📈 Monitoramento de Alta / Baixa do Ibovespa")
st.markdown("Dashboard financeiro com modelo de classificação e análise temporal")


# ======================================================
# CARREGAR MODELO TREINADO
# ======================================================
CAMINHO_MODELO = "pipe_lr_model.joblib"

if not os.path.exists(CAMINHO_MODELO):
    st.error("❌ Modelo não encontrado no caminho definido. Verifique o arquivo!")
    st.stop()

modelo = joblib.load(CAMINHO_MODELO)

# ======================================================
# CARREGAR CSV
# ======================================================
caminho_csv = "Dados Históricos - Ibovespa.csv"

if not os.path.exists(caminho_csv):
    st.error("❌ Arquivo CSV não encontrado.")
    st.stop()

df = pd.read_csv(caminho_csv)

df.rename(columns=lambda x: x.strip(), inplace=True)

# Converter Data
if "Data" in df.columns:
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
    df = df.sort_values("Data")
    df = df.set_index("Data")

st.write("Linhas geradas:", df.shape)
st.dataframe(df.tail())

# ======================================================
# FUNÇÃO DE FEATURE ENGINEERING (IGUAL AO NOTEBOOK)
# ======================================================
def gerar_features(df):
    df = df.copy()

    # --- limpar volume ---
    def clean_volume(vol_str):
        vol_str = str(vol_str).replace(",", ".")
        if "B" in vol_str:
            return float(vol_str.replace("B", "")) * 1e9
        elif "M" in vol_str:
            return float(vol_str.replace("M", "")) * 1e6
        elif "K" in vol_str:
            return float(vol_str.replace("K", "")) * 1e3
        else:
            return float(vol_str)

    df["Vol."] = df["Vol."].apply(clean_volume)

    # Limpar Var%
    df["Var%"] = df["Var%"].replace("%", "", regex=True).str.replace(",", ".").astype(float)

    # Target igual ao notebook
    df["Target"] = (df["Último"].shift(-1) > df["Último"]).astype(int)

    # --- construir dataset base ---
    dataset = pd.DataFrame()
    dataset["Último"] = df["Último"]
    delta = df["Último"].diff()
    threshold = 0.003
    dataset["Target"] = (delta > threshold).astype(int)
    dataset = dataset[:-1]

    dataset["Delta"] = delta.shift(1)
    dataset["Retorno"] = dataset["Último"].pct_change().shift(1)

    # --- funções auxiliares ---
    def make_n_lags(df_local, n_lags, column):
        for i in range(1, n_lags + 1):
            df_local[f"{column}_lag{i}"] = df_local[column].shift(i)
        return df_local

    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def compute_macd(series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def compute_bollinger(series, window=20, num_std=2):
        ma = series.rolling(window).mean()
        std = series.rolling(window).std()
        return ma + num_std * std, ma - num_std * std

    # --- LAGS ---
    dataset = make_n_lags(dataset, 30, "Delta")

    # --- Features do notebook ---
    dataset["Máxima"] = df["Máxima"].diff().shift(1)
    dataset["Mínima"] = df["Mínima"].diff().shift(1)

    dataset["Spread"] = dataset["Máxima"] - dataset["Mínima"]
    dataset["Volatilidade"] = dataset["Target"].rolling(10).std()

    dataset["Abertura"] = df["Abertura"].diff()
    dataset["Volume"] = df["Vol."].diff()

    dataset["High_Prev_Close"] = abs(dataset["Máxima"] - dataset["Último"].shift(1))
    dataset["Low_Prev_Close"] = abs(dataset["Mínima"] - dataset["Último"].shift(1))
    dataset["TR"] = dataset[["Spread", "High_Prev_Close", "Low_Prev_Close"]].max(axis=1)
    dataset["ATR"] = dataset["TR"].ewm(span=14, adjust=False).mean().shift(1)

    dataset["Volume_MM20"] = dataset["Volume"].rolling(10).mean().shift(1)
    dataset["RSI_14"] = compute_rsi(dataset["Delta"])
    dataset["Momentum_5"] = dataset["Delta"] - dataset["Delta"].shift(5)

    dataset["Volatility_10"] = dataset["Delta"].rolling(10).std()
    dataset["Volatilidade_22"] = dataset["Delta"].rolling(22).std()
    dataset["Volatilidade_66"] = dataset["Delta"].rolling(66).std()

    dataset["Return_5D"] = dataset["Último"].pct_change(5).shift(1)
    dataset["Return_10D"] = dataset["Último"].pct_change(10).shift(1)
    dataset["Return_20D"] = dataset["Último"].pct_change(20).shift(1)

    dataset["Daily_Return"] = dataset["Último"].pct_change().shift(1)
    dataset["Historical_Volatility_20D"] = dataset["Daily_Return"].rolling(20).std().shift(1)
    dataset["Historical_Volatility_60D"] = dataset["Daily_Return"].rolling(60).std().shift(1)

    dataset["MACD"], dataset["Signal"] = compute_macd(dataset["Último"])

    for win in [5, 10, 20, 50, 100, 200]:
        dataset[f"Ultimo_MA_{win}"] = dataset["Último"].rolling(win).mean().shift(1)

    dataset["BB_Upper"], dataset["BB_Lower"] = compute_bollinger(dataset["Último"])

    for win in [5, 10, 22, 66, 132, 252]:
        dataset[f"MA_{win}"] = dataset["Abertura"].rolling(win).mean()

    dataset.drop(columns=["Último"], inplace=True)

    df_model = dataset.dropna()

    return df_model


# ======================================================
# GERAR DATASET DE FEATURES
# ======================================================
dados = gerar_features(df)

X = dados.drop(columns=["Target"])
y = dados["Target"]


# ======================================================
# AVALIAÇÃO DO MODELO
# ======================================================
try:
    y_pred = modelo.predict(X)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    st.subheader("📊 Métricas")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acurácia", f"{acc:.2%}")
    col2.metric("Precisão (1)", f"{prec:.2%}")
    col3.metric("Recall (1)", f"{rec:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")

    st.markdown("### 🧮 Matriz de Confusão")
    cm_df = pd.DataFrame(cm, index=["Real Baixa", "Real Alta"], columns=["Prev Baixa", "Prev Alta"])
    st.table(cm_df)

except Exception as e:
    st.error(f"Erro ao fazer a predição: {e}")

# ======================================================
# PREPARAÇÃO DO DATAFRAME PARA O GRÁFICO
# ======================================================

df_plot = dados.copy()

# Previsões
df_plot["Pred"] = modelo.predict(X)
df_plot["Prob_Alta"] = modelo.predict_proba(X)[:, 1]
df_plot["Acerto"] = (df_plot["Target"] == df_plot["Pred"]).astype(int)

# Preço real de abertura (não a feature)
df_plot["Preco_Abertura"] = df["Abertura"].iloc[-len(df_plot):].values

# ======================================================
# CONSTRUÇÃO DO GRÁFICO
# ======================================================

fig = go.Figure()

# ------------------------------------------------------
# 1) Linha do preço real de abertura
# ------------------------------------------------------
fig.add_trace(go.Scatter(
    x=df_plot.index,
    y=df_plot["Preco_Abertura"],
    mode="lines",
    name="Preço (Abertura)",
    line=dict(color="white", width=2)
))

# ------------------------------------------------------
# 2) Previsões de ALTA (cyan)
# ------------------------------------------------------
df_alta = df_plot[df_plot["Pred"] == 1]

fig.add_trace(go.Scatter(
    x=df_alta.index,
    y=df_alta["Preco_Abertura"],
    mode="markers",
    name="Alta Prevista",
    marker=dict(color="cyan", size=9),
    customdata=df_alta[["Prob_Alta", "RSI_14", "Volatilidade"]],
    hovertemplate=(
        "Data: %{x}<br>"
        "Abertura: %{y}<br>"
        "Prob Alta: %{customdata[0]:.2%}<br>"
        "RSI: %{customdata[1]:.2f}<br>"
        "Volatilidade: %{customdata[2]:.4f}<br>"
    )
))

# ------------------------------------------------------
# 3) Previsões de BAIXA (amarelo)
# ------------------------------------------------------
df_baixa = df_plot[df_plot["Pred"] == 0]

fig.add_trace(go.Scatter(
    x=df_baixa.index,
    y=df_baixa["Preco_Abertura"],
    mode="markers",
    name="Baixa Prevista",
    marker=dict(color="yellow", size=9),
    customdata=df_baixa[["Prob_Alta", "RSI_14", "Volatilidade"]],
    hovertemplate=(
        "Data: %{x}<br>"
        "Abertura: %{y}<br>"
        "Prob Alta: %{customdata[0]:.2%}<br>"
        "RSI: %{customdata[1]:.2f}<br>"
        "Volatilidade: %{customdata[2]:.4f}<br>"
    )
))

# ======================================================
# LAYOUT PROFISSIONAL
# ======================================================

fig.update_layout(
    template="plotly_dark",
    height=600,

    xaxis=dict(
        title="Data",
        showgrid=False
    ),

    yaxis=dict(
        title="Preço de Abertura (Real)",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)"
    ),

    # Eixo secundário perfeito
    yaxis2=dict(
        title="Probabilidade de Alta",
        overlaying="y",
        side="right",
        range=[0, 1],
        showgrid=False
    ),

    legend=dict(
        font=dict(size=12),
        bgcolor="rgba(0,0,0,0.3)"
    ),

    title="📈 Modelo Ibovespa — Previsões de Alta e Baixa"
)

st.plotly_chart(fig, use_container_width=True)



# ======================================================
# PREVISÃO USANDO OS DADOS DIGITADOS (CORRETO)
# ======================================================
st.subheader("🔮 Previsão informando os dados do dia anterior")

st.markdown("""
Digite os valores do **último pregão**. O sistema usará **todo o histórico do Ibovespa** 
para gerar todas as features e prever o movimento do próximo dia.
""")

col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

ultimo = col1.number_input("📌 Último (Fechamento)", value=0)
abertura = col2.number_input("🔹 Abertura", value=0)
maxima = col3.number_input("⬆️ Máxima", value=0)
minima = col4.number_input("⬇️ Mínima", value=0)
volume = col5.number_input("📊 Volume", value=0.00, format="%.2f")

if st.button("Gerar Previsão"):

    # 1️⃣ Copiar o histórico original inteiro
    df_novo = df.copy()

    # 2️⃣ Alterar APENAS a última linha com os valores digitados
    idx = df_novo.index[-1]
    df_novo.loc[idx, "Último"] = ultimo
    df_novo.loc[idx, "Abertura"] = abertura
    df_novo.loc[idx, "Máxima"] = maxima
    df_novo.loc[idx, "Mínima"] = minima
    df_novo.loc[idx, "Vol."] = volume

    # 3️⃣ Gerar features para TODOS os dados novamente
    df_features = gerar_features(df_novo)

    if df_features.empty:
        st.error("⚠ Não foi possível gerar features. Verifique os dados.")
    else:
        # 4️⃣ Pegar somente a ÚLTIMA LINHA (a previsão do próximo dia)
        linha_pred = df_features.tail(1)

        X_input = linha_pred.drop(columns=["Target"])
        pred = modelo.predict(X_input)[0]
        prob = modelo.predict_proba(X_input)[0, 1]

        st.success("Previsão gerada com sucesso!")

        st.write(f"🔮 **Movimento previsto:** {'Alta 📈' if pred == 1 else 'Baixa 📉'}")
        st.write(f"📊 **Probabilidade de Alta:** {prob:.2%}")
