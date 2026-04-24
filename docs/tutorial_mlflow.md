# 🚀 Tutorial de Uso do MLflow no Projeto Churn

O **MLflow** é a ferramenta que usamos para gerenciar o ciclo de vida dos nossos modelos de Machine Learning. Ele permite registrar parâmetros, métricas e os próprios modelos de forma organizada.

---

## 1. Como executar o pipeline com MLflow

Você não precisa mudar nada no seu fluxo de trabalho. Ao rodar o pipeline principal, o MLflow já começa a trabalhar em segundo plano:

```bash
make run
```

O que acontece nos bastidores:
1. Uma run chamada `baseline_run` é criada para os modelos Dummy e Regressão Logística.
2. Uma run chamada `mlp_run` é criada para a Rede Neural.
3. Os parâmetros (ex: taxa de aprendizado, épocas) e as métricas são enviados para a pasta local `mlruns/`.

---

## 2. Abrindo a Interface Visual (UI)

Para visualizar os resultados de forma gráfica e comparar experimentos, use o comando:

```bash
make mlflow-ui
```

Isso iniciará um servidor local. Abra o seu navegador em: **[http://localhost:5000](http://localhost:5000)**

---

## 3. O que observar na UI do MLflow

### 📈 Gráficos de Perda (Loss)
Dentro da run `mlp_run`, você encontrará a aba **Metrics**. Clique em `train_loss` ou `val_loss` para ver o gráfico de como a rede neural aprendeu ao longo das épocas. Se a linha de validação começar a subir enquanto a de treino desce, você detectou um *Overfitting*!

### 📋 Comparação de Parâmetros
Você pode selecionar múltiplas runs na página principal e clicar em **Compare**. Isso é útil para entender, por exemplo, como o aumento do número de neurônios afetou o F1-Score final.

### 🗄️ Model Registry (Catálogo de Modelos)
Nossos modelos são registrados automaticamente. Na aba **Models** do menu superior, você verá:
- `ChurnMLP`
- `ChurnLogisticRegression`
- `ChurnDummyClassifier`

Lá você pode ver as versões de cada modelo e marcá-los como `Production` quando estiverem prontos para uso real.

---

## 4. Avaliação Final

Após treinar, você deve rodar o script de avaliação para persistir as métricas finais de teste no MLflow:

```bash
make eval
```

Isso criará uma `evaluation_run` contendo:
- As métricas de Precisão, Recall e F1 de todos os modelos.
- O gráfico `evaluation_summary.png` como um **Artefato** (visível na aba *Artifacts* da run).

---

## 5. Dicas para Desenvolvedores

- **Localização dos dados:** O MLflow salva tudo na pasta `mlruns/` na raiz do projeto. Ela está no `.gitignore` para não subir para o repositório.
- **Isolamento de Testes:** Ao rodar `make test-full`, o MLflow usa uma pasta temporária para não poluir seus experimentos reais.
- **Configuração:** Se precisar mudar o nome do experimento ou o local de salvamento, altere o dataclass `MLFlowConfig` em `src/models/config.py`.
