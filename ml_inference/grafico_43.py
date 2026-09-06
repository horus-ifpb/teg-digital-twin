import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Definição da Arquitetura do Modelo (idêntica ao train.py / app.py)
class TEGDigitalTwinNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=2):
        super(TEGDigitalTwinNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def treinar_e_gerar_grafico():
    dataset_path = "teg_dataset.csv"

    # Garante a existência do dataset
    if not os.path.exists(dataset_path):
        from generate_dataset import generate_teg_dataset

        generate_teg_dataset()

    # 2. Carregamento e Divisão dos Dados (80% Treino / 20% Teste)
    df = pd.read_csv(dataset_path)

    X = df[
        [
            "t_quente",
            "t_frio",
            "v_real",
            "v_ideal",
            "p_real_mW",
            "p_ideal_mW",
        ]
    ].values
    y = df[["score_degradacao", "eficiencia_relativa"]].values

    # Separando treino e teste via numpy
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    train_idx, test_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Conversão para Tensores PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 3. Treinamento com Registro do Histórico de Loss
    model = TEGDigitalTwinNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    epochs = 60
    history_loss = []

    print("🚀 Treinando modelo e registrando métricas...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        history_loss.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Época [{epoch+1}/{epochs}] - Loss (MSE): {avg_loss:.6f}")

    # 4. Avaliação e Cálculo de Métricas (RMSE e MAE) no Conjunto de Teste
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()

    # Avaliação do Score de Degradação (Coluna 0)
    y_true_deg = y_test[:, 0]
    y_pred_deg = y_pred[:, 0]

    rmse = np.sqrt(np.mean((y_true_deg - y_pred_deg) ** 2))
    mae = np.mean(np.abs(y_true_deg - y_pred_deg))

    print(
        f"\n📊 Métricas no Conjunto de Teste:\n   RMSE: {rmse:.4f}\n   MAE:  {mae:.4f}"
    )

    # 5. Construção do Gráfico 4.3 (Padrão Acadêmico ABNT/IEEE)
    plt.rcParams["font.family"] = "serif"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Painel (A): Curva de Aprendizado / Perda
    ax1.plot(
        range(1, epochs + 1),
        history_loss,
        color="#1f77b4",
        linewidth=2.0,
        label="Erro Quadrático Médio (MSE)",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Épocas de Treinamento", fontsize=10)
    ax1.set_ylabel("Perda (MSELoss - Escala Logarítmica)", fontsize=10)
    ax1.set_title("(A) Convergência do Treinamento PyTorch", fontsize=11)
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Painel (B): Gráfico de Paridade / Regressão (ANSYS ROM vs PyTorch)
    # Seleciona amostra de 200 pontos para manter a visualização limpa
    sample_mask = np.random.choice(len(y_true_deg), size=200, replace=False)

    ax2.scatter(
        y_true_deg[sample_mask],
        y_pred_deg[sample_mask],
        color="#2ca02c",
        alpha=0.6,
        edgecolors="k",
        linewidths=0.5,
        s=35,
        label=f"Amostras de Teste\n(RMSE = {rmse:.3f}, MAE = {mae:.3f})",
    )

    # Linha de Identidade Ideal (y = x)
    ax2.plot(
        [0, 1],
        [0, 1],
        color="red",
        linestyle="--",
        linewidth=1.8,
        label="Referência Ideal ($y = x$)",
    )

    ax2.set_xlabel("Score de Degradação Nominal (ANSYS ROM)", fontsize=10)
    ax2.set_ylabel("Score de Degradação Predito (PyTorch)", fontsize=10)
    ax2.set_title("(B) Correlação com Modelo de Elementos Finitos", fontsize=11)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9)

    plt.tight_layout()

    # Salva o gráfico em alta resolução
    output_filename = "grafico_43_validacao_ia.png"
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Gráfico salvo com sucesso como '{output_filename}'!")


if __name__ == "__main__":
    treinar_e_gerar_grafico()