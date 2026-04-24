"""
Módulo de Configuração Central do Pipeline
==========================================
Este arquivo define todos os hiperparâmetros do projeto em um único lugar.

O que é um hiperparâmetro?
--------------------------
São os "botões" que controlam como um modelo de Machine Learning é treinado.
Diferente dos parâmetros internos do modelo (que ele aprende sozinho com os dados),
os hiperparâmetros são definidos ANTES do treinamento e afetam diretamente
a qualidade do resultado final.

Centralizar os hiperparâmetros aqui evita que valores estejam "espalhados"
pelo código, facilitando experimentos e manutenção. Para alterar o comportamento
do treinamento, basta modificar este arquivo.

Como usar:
----------
    from src.models.config import PipelineConfig

    cfg = PipelineConfig()  # Usa os valores padrão
    print(cfg.mlp.epochs)   # → 50

    # Customizar para um experimento:
    cfg = PipelineConfig(seed=0, mlp=MLPConfig(epochs=100, learning_rate=0.0005))
"""

from dataclasses import dataclass, field


@dataclass
class MLPConfig:
    """
    Hiperparâmetros da Rede Neural (MLP — Multi-Layer Perceptron).

    Uma Rede Neural aprende padrões nos dados ao ajustar seus pesos internos
    ao longo de várias passagens pelo conjunto de treinamento (épocas).
    Cada hiperparâmetro abaixo controla um aspecto diferente desse processo.

    Atributos:
        hidden_dims (list[int]):
            Define a arquitetura das camadas ocultas da rede. Cada número
            representa o tamanho de uma camada. Ex: [32, 16] significa
            duas camadas ocultas, a primeira com 32 neurônios e a segunda
            com 16. Mais camadas ou neurônios aumentam a capacidade de
            aprendizado, mas também o risco de overfitting (decorar os
            dados de treino em vez de generalizar).

        dropout_rate (float):
            Probabilidade de "desligar" aleatoriamente neurônios durante
            o treino. Isso força a rede a aprender representações mais
            robustas e previne overfitting. O valor 0.0 desativa o dropout.
            Valores típicos: entre 0.2 e 0.5.

        epochs (int):
            Número de vezes que a rede verá TODO o conjunto de treinamento.
            Mais épocas dão mais tempo para aprender, mas aumentam o custo
            computacional e o risco de overfitting. Monitore a Loss de
            validação para saber quando parar.

        batch_size (int):
            Número de amostras processadas juntas antes de atualizar os
            pesos da rede. Batches menores (ex: 32) atualizam os pesos
            com mais frequência; batches maiores são mais estáveis.
            Valor comum: 32–128.

        learning_rate (float):
            "Tamanho do passo" que o otimizador (Adam) dá ao ajustar
            os pesos. Muito alto: aprendizado instável. Muito baixo:
            convergência lenta. O valor padrão 0.001 é um bom ponto
            de partida para Adam.
    """

    hidden_dims: list[int] = field(default_factory=lambda: [32, 16])
    dropout_rate: float = 0.0
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3


@dataclass
class BaselineConfig:
    """
    Hiperparâmetros dos Modelos de Baseline (Linha de Base).

    Modelos de baseline são modelos simples usados como ponto de comparação.
    Se um modelo mais complexo (como a Rede Neural) não for significativamente
    melhor que o baseline, algo pode estar errado no pipeline.

    Atributos:
        logistic_max_iter (int):
            Número máximo de iterações para o algoritmo de otimização da
            Regressão Logística convergir (encontrar a melhor solução).
            Aumente se o treinamento gerar avisos de "não convergência".

        logistic_class_weight (str):
            Define como tratar o desbalanceamento de classes. O dataset
            de churn tem muito mais clientes que NÃO cancelam (classe 0)
            do que os que cancelam (classe 1). O valor 'balanced' faz o
            modelo dar mais peso aos casos de churn, evitando que ele
            simplesmente ignore a classe minoritária.

        random_state (int):
            Semente de aleatoriedade para garantir que o treinamento seja
            reproduzível. Com o mesmo valor, o resultado será sempre idêntico.
    """

    logistic_max_iter: int = 1000
    logistic_class_weight: str = "balanced"
    random_state: int = 42


@dataclass
class PipelineConfig:
    """
    Configuração raiz do pipeline completo.

    Agrega todas as configurações de modelos em um único objeto.
    É o ponto de entrada recomendado para customizar qualquer
    aspecto do treinamento.

    Atributos:
        seed (int):
            Semente global de aleatoriedade. Garante que splits de dados,
            inicialização de pesos e shuffling de batches sejam idênticos
            entre execuções, tornando os experimentos reproduzíveis.

        mlp (MLPConfig):
            Configurações específicas da Rede Neural MLP.

        baseline (BaselineConfig):
            Configurações específicas dos modelos de baseline.

    Exemplo de uso — modificar apenas o número de épocas::

        from src.models.config import PipelineConfig, MLPConfig

        cfg = PipelineConfig(mlp=MLPConfig(epochs=100))
        # Mantém todos os outros padrões, apenas aumenta as épocas.
    """

    seed: int = 42
    mlp: MLPConfig = field(default_factory=MLPConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
