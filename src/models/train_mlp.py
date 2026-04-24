"""
Treinamento da Rede Neural MLP (Multi-Layer Perceptron)
========================================================
Este script define a arquitetura e executa o treinamento da Rede Neural
para classificação de churn.

O que é uma MLP?
----------------
Uma Multi-Layer Perceptron é o tipo mais básico de Rede Neural. Ela é
composta por camadas de neurônios conectados sequencialmente:

  Input → [Camada Oculta 1] → [Camada Oculta 2] → Output
  (features)  (32 neurônios)   (16 neurônios)    (1 valor)

Cada neurônio recebe entradas, aplica uma transformação matemática e passa
o resultado para a próxima camada. Ao treinar, a rede ajusta os "pesos"
dessas conexões para minimizar os erros nas predições.

Por que usar uma MLP para churn?
---------------------------------
A Regressão Logística aprende apenas relações lineares. A MLP, com suas
camadas ocultas e funções de ativação (ReLU), consegue capturar relações
não-lineares e interações complexas entre as features — potencialmente
melhorando a detecção de padrões sutis de churn.

Integração com MLflow:
-----------------------
Esta run registra as métricas de perda (train_loss e val_loss) A CADA ÉPOCA,
gerando curvas de aprendizado navegáveis na UI do MLflow. Isso permite
identificar visualmente overfitting (val_loss crescendo enquanto train_loss cai)
sem precisar reler os logs de texto.

Artefatos gerados em 'models/' (compatibilidade legada):
  - mlp_model.pth → Pesos da rede neural no formato PyTorch

Execução:
    python src/models/train_mlp.py
"""

import numpy as np
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from mlflow.models.signature import infer_signature
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.models.config import PipelineConfig
from src.models.data_utils import get_data_splits, set_seed


class ChurnMLP(nn.Module):
    """
    Arquitetura da Rede Neural para predição de churn.

    Esta classe herda de nn.Module, que é a classe base para todos os
    modelos do PyTorch. Ao herdar dela, ganhamos automaticamente a capacidade
    de treinar, salvar e carregar o modelo.

    A arquitetura é construída dinamicamente com base nos parâmetros recebidos,
    permitindo experimentar diferentes configurações sem alterar o código.

    Exemplo de arquitetura padrão (hidden_dims=[32, 16]):
        Input (N features)
          ↓
        Linear(N → 32) + ReLU   ← Primeira camada oculta
          ↓
        Linear(32 → 16) + ReLU  ← Segunda camada oculta
          ↓
        Linear(16 → 1)           ← Saída (logit bruto, sem ativação sigmoid)

    Por que a saída não tem sigmoid?
    ---------------------------------
    Usamos BCEWithLogitsLoss como função de perda, que já aplica sigmoid
    internamente de forma numericamente mais estável. Durante a predição,
    aplicamos sigmoid manualmente para converter o logit em probabilidade.

    Args:
        input_dim: Número de features de entrada (colunas do X_train).
        hidden_dims: Lista com o número de neurônios em cada camada oculta.
                     Ex: [32, 16] → duas camadas, 32 e 16 neurônios.
                     Ex: [64, 32, 16] → três camadas.
        dropout_rate: Probabilidade de zerar aleatoriamente neurônios durante
                      o treino. 0.0 = sem dropout. Valores típicos: 0.2–0.5.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()  # Inicializa a classe pai (nn.Module)

        if hidden_dims is None:
            hidden_dims = [32, 16]  # Arquitetura padrão do projeto

        # Constrói as camadas dinamicamente para suportar qualquer arquitetura.
        layers: list[nn.Module] = []
        prev_dim = input_dim  # Dimensão da camada anterior (começa com as features)

        for h_dim in hidden_dims:
            # nn.Linear: a camada principal. Aplica y = xW + b, onde W são
            # os pesos (o que a rede aprende) e b é o viés (bias).
            layers.append(nn.Linear(prev_dim, h_dim))

            # ReLU (Rectified Linear Unit): função de ativação que introduz
            # não-linearidade. Simplesmente: f(x) = max(0, x).
            # Sem ativação, camadas empilhadas seriam equivalentes a uma só.
            layers.append(nn.ReLU())

            # Dropout: durante o treino, desliga neurônios aleatoriamente.
            # Força a rede a aprender representações redundantes e robustas,
            # reduzindo overfitting. Desativado durante a avaliação (model.eval()).
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = h_dim  # Atualiza a dimensão de entrada para a próxima camada

        # Camada de saída: produz 1 valor (logit) por amostra.
        # Sem ativação aqui — a loss BCEWithLogitsLoss aplica sigmoid internamente.
        layers.append(nn.Linear(prev_dim, 1))

        # nn.Sequential: empilha as camadas em sequência.
        # Durante o forward(), os dados fluem pela lista de layers em ordem.
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define o caminho de propagação para frente (forward pass).

        Este método é chamado automaticamente quando você faz model(x).
        Ele passa os dados de entrada pela rede sequencial e retorna o
        logit de saída (antes do sigmoid).

        Args:
            x: Tensor de entrada com shape (batch_size, input_dim).

        Returns:
            Tensor de saída com shape (batch_size, 1) contendo os logits.
        """
        return self.network(x)


def main(
    data_path: Path | None = None,
    models_dir: Path | None = None,
    config: PipelineConfig | None = None,
) -> None:
    """
    Treina a Rede Neural MLP, registra os experimentos no MLflow e salva
    os pesos em disco.

    O loop de treinamento segue o ciclo padrão de ML com PyTorch:
      Para cada época:
        1. Modo treino: passa os batches, calcula loss, atualiza pesos.
        2. Modo avaliação: passa os dados de validação SEM atualizar pesos.
        3. Loga as losses no MLflow (step=epoch) para curvas de aprendizado.

    Curvas de aprendizado no MLflow:
    ---------------------------------
    Ao logar train_loss e val_loss com step=epoch, a UI do MLflow gera
    automaticamente gráficos de linha mostrando a evolução do aprendizado.
    Se val_loss começar a subir enquanto train_loss cai, é sinal de overfitting.

    Args:
        data_path: Caminho para o CSV bruto. Se None, usa o padrão do projeto.
        models_dir: Diretório para salvar o modelo. Se None, usa 'models/'.
        config: Hiperparâmetros. Se None, usa PipelineConfig() com padrões.
    """
    cfg = config or PipelineConfig()

    # 1. Reprodutibilidade — deve vir antes de qualquer operação aleatória.
    set_seed(cfg.seed)

    BASE_DIR = Path(__file__).resolve().parents[2]
    file_path = data_path or (BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv")
    out_dir = models_dir or (BASE_DIR / "models")
    out_dir.mkdir(exist_ok=True)

    # 2. Carregamento de Dados
    # A MLP usa TREINO para aprender e VALIDAÇÃO para monitorar overfitting.
    # O conjunto de TESTE (5º e 6º elementos) é ignorado aqui.
    logger.info("Carregando e preparando dados para PyTorch...")
    X_train, y_train, X_val, y_val, _, _, _ = get_data_splits(file_path)

    # 3. Conversão para Tensores PyTorch
    # O PyTorch trabalha com Tensors, não com DataFrames do Pandas.
    # .values converte o DataFrame para array NumPy, e torch.tensor
    # converte para o formato que a GPU/CPU da rede entende.
    # float32 é o tipo numérico padrão para redes neurais (float64 é
    # desnecessariamente pesado e não traz ganho de precisão aqui).
    X_train_t = torch.tensor(X_train.astype(np.float32).values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val.astype(np.float32).values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    # .view(-1, 1) remodela o vetor de targets para shape (N, 1),
    # que é o formato esperado pelo BCEWithLogitsLoss.

    # 4. DataLoaders
    # Em vez de treinar com todos os dados de uma vez (o que pode exceder
    # a memória), dividimos em "batches" (lotes). O DataLoader faz isso
    # automaticamente. shuffle=True embaralha os batches a cada época
    # para evitar que a rede memorize a ordem dos dados.
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg.mlp.batch_size,
        shuffle=True,  # Embaralha o treino a cada época
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=cfg.mlp.batch_size,
        shuffle=False,  # Validação não precisa ser embaralhada
    )

    # 5. Instanciação do Modelo, Função de Perda e Otimizador
    model = ChurnMLP(
        input_dim=X_train.shape[1],       # Número de features de entrada
        hidden_dims=cfg.mlp.hidden_dims,
        dropout_rate=cfg.mlp.dropout_rate,
    )

    # BCEWithLogitsLoss: "Binary Cross-Entropy with Logits".
    # É a função de perda padrão para classificação binária com PyTorch.
    # Mede o quão erradas são as predições. Quanto menor, melhor.
    criterion = nn.BCEWithLogitsLoss()

    # Adam: algoritmo de otimização moderno que adapta a taxa de aprendizado
    # automaticamente para cada parâmetro. É a escolha padrão para MLP.
    optimizer = optim.Adam(model.parameters(), lr=cfg.mlp.learning_rate)

    # =========================================================================
    # 6. Configuração do MLflow e Loop de Treinamento
    # =========================================================================
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="mlp_run"):

        # --- Log de Parâmetros ---
        # Registramos TODOS os hiperparâmetros da MLP para que cada run
        # seja completamente reproduzível e comparável na UI do MLflow.
        mlflow.log_params({
            "seed": cfg.seed,
            "hidden_dims": str(cfg.mlp.hidden_dims),
            "dropout_rate": cfg.mlp.dropout_rate,
            "epochs": cfg.mlp.epochs,
            "batch_size": cfg.mlp.batch_size,
            "learning_rate": cfg.mlp.learning_rate,
            "input_dim": X_train.shape[1],
        })

        logger.info(
            "Iniciando treinamento | épocas={} | batch={} | lr={}",
            cfg.mlp.epochs, cfg.mlp.batch_size, cfg.mlp.learning_rate,
        )

        # 7. Loop de Treinamento
        for epoch in range(cfg.mlp.epochs):

            # --- FASE DE TREINO ---
            # model.train() ativa comportamentos específicos do treino:
            # - Dropout está ATIVO (desliga neurônios aleatoriamente)
            # - BatchNorm (se houver) usa estatísticas do batch atual
            model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()           # Zera os gradientes do passo anterior
                loss = criterion(model(batch_x), batch_y)  # Calcula a perda
                loss.backward()                 # Backpropagation: calcula gradientes
                optimizer.step()               # Atualiza os pesos com base nos gradientes
                train_loss += loss.item()      # Acumula a perda para calcular a média

            # --- FASE DE VALIDAÇÃO ---
            # model.eval() desativa dropout e usa estatísticas fixas do BatchNorm.
            # torch.no_grad() desabilita o cálculo de gradientes — não precisamos
            # deles aqui, o que economiza memória e acelera o cálculo.
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    val_loss += criterion(model(batch_x), batch_y).item()

            # Calcula as perdas médias por batch para esta época.
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            # --- Log de Métricas por Época no MLflow ---
            # O parâmetro `step=epoch` é fundamental: ele informa ao MLflow
            # que esta métrica pertence ao passo (época) N, não a um valor único.
            # Isso permite que a UI do MLflow gere gráficos de linha mostrando
            # a evolução do aprendizado ao longo do tempo.
            mlflow.log_metrics(
                {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                step=epoch,
            )

            # Loga o progresso a cada 10 épocas (ou na primeira) no terminal.
            # Se a val_loss estiver AUMENTANDO enquanto a train_loss diminui,
            # é sinal de overfitting — considere reduzir épocas ou adicionar dropout.
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Época [{}/{}] | Loss Treino: {:.4f} | Loss Val: {:.4f}",
                    epoch + 1, cfg.mlp.epochs,
                    avg_train_loss,
                    avg_val_loss,
                )

        # 8. Log do Modelo Final no MLflow
        # Preparamos um exemplo de entrada para que o MLflow possa inferir
        # a assinatura (schema) do modelo automaticamente.
        model.eval()
        with torch.no_grad():
            example_input = X_train_t[:5]  # Amostra pequena para inferência de schema
            example_output = torch.sigmoid(model(example_input)).numpy()

        # infer_signature analisa o tensor de entrada e a saída (probabilidades)
        # e gera automaticamente o contrato de I/O do modelo.
        signature = infer_signature(
            example_input.numpy(),
            example_output,
        )

        # mlflow.pytorch.log_model salva o modelo PyTorch completo como
        # artefato da run. O MLflow garante que as dependências (versão do
        # PyTorch) sejam registradas junto para reprodutibilidade futura.
        mlp_model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="mlp_model",
            signature=signature,
            registered_model_name="ChurnMLP" if cfg.mlflow.register_model else None,
        )

        logger.success(
            "MLflow | Modelo MLP registrado em: {}",
            mlp_model_info.model_uri,
        )

        # 9. Salvamento Legado do Modelo (compatibilidade)
        # Mantemos o torch.save() para compatibilidade com evaluate_models.py
        # e quaisquer scripts que ainda dependam do arquivo .pth local.
        model_path = out_dir / "mlp_model.pth"
        torch.save(model.state_dict(), model_path)
        logger.success("Rede Neural treinada! Modelo salvo em: {}", model_path)


if __name__ == "__main__":
    main()
