# Telco Churn Prediction - Pipeline MLOps

Projeto focado na previsão de cancelamento de clientes (Churn) utilizando modelos de Machine Learning tradicionais e Redes Neurais.

### 🚀 Guia Rápido de Execução

**1. Preparação do Ambiente:**
```bash
uv pip install -e .
```

**2. Processamento e Treinamento (Ordem Obrigatória):**
Execute os scripts abaixo em sequência para preparar os dados, treinar os modelos e gerar os resultados:

```bash
# Opcional: Processa e exporta CSVs para data/processed/
python data_prep.py

# Treina Dummy e Regressão Logística (Scikit-Learn)
python src/train_baselines.py

# Treina Rede Neural MLP (PyTorch)
python src/train_mlp.py

# Gera métricas finais e gráficos comparativos
python src/evaluate_models.py
```

**3. Visualização (Dashboard):**
```bash
streamlit run app_vis.py
```

---

### 📂 Estrutura de Pastas
- `data/`: Datasets brutos e processados.
- `models/`: Modelos persistidos para inferência.
- `src/`: Lógica central, utilitários e treinamento.
- `notebooks/`: Experimentos iniciais (EDA).
- `data_prep.py`: Script para processamento manual e exportação de dados.
- `app_vis.py`: Dashboard interativo (Streamlit).

