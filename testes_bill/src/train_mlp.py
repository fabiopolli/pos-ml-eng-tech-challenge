import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from data_utils import set_seed, get_data_splits

class ChurnMLP(nn.Module):
    """Arquitetura da Rede Neural (Multi-Layer Perceptron)"""
    def __init__(self, input_dim):
        super(ChurnMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Saída linear (sem ativação para uso com BCEWithLogitsLoss)
        )
        
    def forward(self, x):
        return self.network(x)

def main():
    # 1. Configuração e Reprodutibilidade
    set_seed(42)
    
    # Caminhos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'raw', 'Telco-Customer-Churn.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Carregamento de Dados (Treino e Validação)
    print("Carregando e preparando dados para PyTorch...")
    X_train, y_train, X_val, y_val, _, _ = get_data_splits(file_path)
    
    # 3. Conversão para Tensores
    X_train_t = torch.tensor(X_train.astype(np.float32).values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val.astype(np.float32).values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    
    # 4. Construção dos DataLoaders
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    # 5. Instanciação da Rede e Motor de Treino
    model = ChurnMLP(X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Loop de Treinamento
    epochs = 50
    print(f"Iniciando treinamento por {epochs} épocas...")
    
    for epoch in range(epochs):
        # Modo de Treino
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Modo de Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                v_loss = criterion(outputs, batch_y)
                val_loss += v_loss.item()
        
        # Log de progresso
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f"Época [{epoch+1}/{epochs}] | Loss Treino: {avg_train:.4f} | Loss Val: {avg_val:.4f}")

    # 7. Salvamento do Estado do Modelo
    model_path = os.path.join(models_dir, 'mlp_model.pth')
    torch.save(model.state_dict(), model_path)
    
    print("-" * 30)
    print(f"Rede Neural treinada com sucesso!")
    print(f"Modelo salvo em: {model_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()
