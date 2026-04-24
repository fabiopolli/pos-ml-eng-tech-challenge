import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Garante que a raiz do projeto esteja no path para imports absolutos (src.*)
BASE_DIR = Path(__file__).resolve().parent.parent  # front/ → raiz/
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.data_utils import get_data_splits
from src.models.train_mlp import ChurnMLP

# --- Caminhos resolvidos a partir da raiz ---
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
X_TRAIN_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"
Y_TRAIN_PATH = BASE_DIR / "data" / "processed" / "y_train.csv"
MODELS_DIR = BASE_DIR / "models"

# Configurações de Página Estilo Premium
st.set_page_config(page_title="Telco Churn - Dashboard Analítico", layout="wide")

# CSS customizado para Dark Mode Total (Estética High-Tech)
st.markdown("""
    <style>
    /* Fundo Global e Containers */
    .stApp {
        background-color: #0e1117 !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117 !important;
    }

    /* Texto Global */
    h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown {
        color: #e6edf3 !important;
    }

    /* Métricas de Performance */
    div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 800 !important;
        text-shadow: 0px 0px 10px rgba(88, 166, 255, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }

    /* Tabs (Abas) Customizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0e1117;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #161b22;
        border-radius: 8px 8px 0px 0px;
        color: #8b949e;
        border: 1px solid #30363d;
        border-bottom: none;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #ffffff !important;
        border-top: 3px solid #58a6ff !important;
    }

    /* Tabelas e Dataframes */
    .stDataFrame {
        border: 1px solid #30363d;
    }

    /* Estilo para avisos e infos */
    .stAlert {
        background-color: #161b22;
        color: #e6edf3;
        border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Painel Analítico: Telco Customer Churn")
st.markdown("Análise Exploratória, Preparação de Dados e Avaliação de Modelos Preditivos para Retenção de Clientes.")

# Configurar estilo dos gráficos para Dark Mode
plt.style.use("dark_background")
sns.set_palette("husl")
plt.rcParams["figure.facecolor"] = "#0e1117"
plt.rcParams["axes.facecolor"] = "#0e1117"
plt.rcParams["savefig.facecolor"] = "#0e1117"


@st.cache_data
def load_base_data():
    raw_df = pd.read_csv(RAW_DATA_PATH) if RAW_DATA_PATH.exists() else None

    if X_TRAIN_PATH.exists() and Y_TRAIN_PATH.exists():
        x_train = pd.read_csv(X_TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH)
        processed_df = pd.concat([x_train, y_train], axis=1)
    else:
        processed_df = None

    return raw_df, processed_df


raw_df, processed_df = load_base_data()

if raw_df is not None:
    tab1, tab2, tab3 = st.tabs([
        "📊 Análise Exploratória (EDA)",
        "📈 Engenharia de Dados",
        "🎯 Performance dos Modelos",
    ])

    with tab1:
        st.header("Análise Exploratória de Dados (EDA)")
        st.write(f"**Dimensões do Dataset:** {raw_df.shape[0]} registros e {raw_df.shape[1]} variáveis.")
        st.dataframe(raw_df.head(100), use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        churn_counts = raw_df["Churn"].value_counts()
        churn_percent = raw_df["Churn"].value_counts(normalize=True) * 100

        with col1:
            st.subheader("Frequência de Churn")
            fig_churn, ax1 = plt.subplots(figsize=(6, 4))
            churn_counts.plot(kind="bar", ax=ax1, color=["#3498DB", "#E74C3C"])
            ax1.set_title("Volume de Clientes (Churn vs Retenção)")
            ax1.set_xlabel("Status de Churn")
            ax1.set_ylabel("Quantidade")
            ax1.tick_params(axis="x", rotation=0)
            st.pyplot(fig_churn)

        with col2:
            st.subheader("Proporção de Churn (%)")
            fig_percent, ax2 = plt.subplots(figsize=(6, 4))
            churn_percent.plot(kind="bar", ax=ax2, color=["#3498DB", "#E74C3C"])
            ax2.set_title("Distribuição Percentual")
            ax2.set_xlabel("Status de Churn")
            ax2.set_ylabel("Percentual (%)")
            ax2.tick_params(axis="x", rotation=0)
            for i, v in enumerate(churn_percent):
                ax2.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
            st.pyplot(fig_percent)

        st.markdown("---")
        st.subheader("Distribuições de Variáveis Numéricas")
        temp_df = raw_df.copy()
        temp_df["TotalCharges"] = pd.to_numeric(temp_df["TotalCharges"], errors="coerce")
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

        fig_num, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(num_cols):
            sns.histplot(data=temp_df[col].dropna(), kde=True, ax=axes[i], color="#2980B9")
            axes[i].set_title(f"Distribuição: {col}")
            axes[i].set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig_num)

        st.markdown("---")
        st.subheader("Análise de Churn por Categorias Importantes")
        st.write("Identificando grupos de alto risco com base em variáveis categóricas.")

        cat_cols = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen", "Partner", "Dependents"]

        fig_cat, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, col in enumerate(cat_cols):
            if col in raw_df.columns:
                churn_by_cat = pd.crosstab(raw_df[col], raw_df["Churn"], normalize="index") * 100
                churn_by_cat.plot(kind="bar", stacked=True, ax=axes[i], color=["#27AE60", "#C0392B"], alpha=0.85)
                axes[i].set_title(f"Taxa de Churn por {col}", fontsize=14, fontweight="bold")
                axes[i].set_ylabel("Percentual (%)")
                axes[i].set_xlabel("")
                axes[i].legend(title="Churn", labels=["Não", "Sim"], loc="upper right")
                axes[i].tick_params(axis="x", rotation=45)
                axes[i].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig_cat)

    with tab2:
        if processed_df is not None:
            st.header("Pipeline de Transformação")
            st.info("💡 **Feature Engineering aplicada:** Tenure Bins, Services Count, Family Stability, Electronic Check Flag e Charge Difference.")
            st.dataframe(processed_df.head(100), use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Estatísticas Descritivas (Normalizadas)")
                scaled_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Services_Count", "Charge_Difference"]
                available_scaled = [c for c in scaled_cols if c in processed_df.columns]
                st.dataframe(processed_df[available_scaled].describe().T, use_container_width=True)

            with col4:
                st.subheader("Distribuição do Alvo Processado")
                fig_proc, ax_proc = plt.subplots(figsize=(6, 4))
                processed_df["Churn"].value_counts().plot(
                    kind="bar", color=["#27AE60", "#C0392B"], ax=ax_proc
                )
                ax_proc.set_title("Status do Churn Pós-Processamento")
                ax_proc.set_xlabel("Churn (0=Não, 1=Sim)")
                ax_proc.tick_params(axis="x", rotation=0)
                st.pyplot(fig_proc)
        else:
            st.warning("⚠️ Execute o pipeline de preparação para visualizar os dados transformados.")

    with tab3:
        st.header("🎯 Resultados e Performance Analítica")

        path_dummy = MODELS_DIR / "dummy_model.pkl"
        path_lr = MODELS_DIR / "logistic_model.pkl"
        path_mlp = MODELS_DIR / "mlp_model.pth"

        if all(p.exists() for p in [path_dummy, path_lr, path_mlp]):
            with st.spinner("Calculando métricas em tempo real..."):
                _, _, _, _, X_test, y_test, _ = get_data_splits(RAW_DATA_PATH)

                m_dummy_obj = joblib.load(path_dummy)
                m_lr_obj = joblib.load(path_lr)
                m_mlp_obj = ChurnMLP(input_dim=X_test.shape[1])
                m_mlp_obj.load_state_dict(torch.load(path_mlp))
                m_mlp_obj.eval()

                y_pred_dummy = m_dummy_obj.predict(X_test)
                y_pred_lr = m_lr_obj.predict(X_test)
                X_test_t = torch.tensor(X_test.astype(np.float32).values, dtype=torch.float32)
                with torch.no_grad():
                    y_pred_mlp = (
                        torch.sigmoid(m_mlp_obj(X_test_t)) > 0.5
                    ).int().numpy().flatten()

                def calc_metrics(y_true, y_pred):
                    p, r, f, _ = precision_recall_fscore_support(
                        y_true, y_pred, average="binary", zero_division=0
                    )
                    acc = accuracy_score(y_true, y_pred)
                    return {"Acurácia": acc, "Precisão": p, "Recall": r, "F1-Score": f}

                res = pd.DataFrame({
                    "Dummy Baseline": calc_metrics(y_test, y_pred_dummy),
                    "LogReg (Balanced)": calc_metrics(y_test, y_pred_lr),
                    "Neural Net (MLP)": calc_metrics(y_test, y_pred_mlp),
                }).T

                st.subheader("Cockpit de Métricas")
                c1, c2, c3 = st.columns(3)
                best_model = res["Recall"].idxmax()
                c1.metric("Melhor Recall", f"{res['Recall'].max():.2%}", f"Modelo: {best_model}")
                c2.metric("Melhor F1-Score", f"{res['F1-Score'].max():.2%}", f"Modelo: {res['F1-Score'].idxmax()}")
                c3.metric("Acurácia MLP", f"{res.loc['Neural Net (MLP)', 'Acurácia']:.2%}")

                st.markdown("---")
                st.subheader("Comparativo Gráfico de Desempenho")
                fig_m, ax_m = plt.subplots(figsize=(10, 5))
                res.plot(kind="bar", ax=ax_m, colormap="viridis")
                ax_m.set_ylim(0, 1.1)
                ax_m.set_title("Comparação de Métricas (Foco em Classe 1 - Churn)")
                ax_m.grid(axis="y", linestyle="--", alpha=0.3)
                plt.xticks(rotation=0)
                st.pyplot(fig_m)

                st.markdown("---")
                st.subheader("Visualização de Matrizes de Confusão")
                eval_img_path = BASE_DIR / "evaluation_summary.png"
                if eval_img_path.exists():
                    st.image(str(eval_img_path), caption="Comparativo de Matrizes de Confusão (Dados de Teste)")

                st.success(
                    f"💡 **Conclusão:** O modelo **{best_model}** é o recomendado para operações "
                    f"de retenção proativa, pois maximiza o Recall ({res['Recall'].max():.2%}), "
                    f"permitindo que o time de CRM atue sobre a maioria dos clientes propensos ao cancelamento."
                )
        else:
            st.warning("⚠️ Modelos não detectados em `/models`. Por favor, execute os scripts de treinamento primeiro.")

else:
    st.error("Dataset não encontrado em data/raw/Telco-Customer-Churn.csv")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 0.8rem;'>"
    "Painel Analítico de Churn v2.0 | Bill & Data Engineering</div>",
    unsafe_allow_html=True,
)
