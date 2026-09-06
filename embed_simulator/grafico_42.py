import numpy as np
import matplotlib.pyplot as plt
from ekf import TEGExtendedKalmanFilter
from app import generate_physical_data

def gerar_grafico_artigo():
    seebeck_alpha = 0.085
    r_init = 1.6
    ekf = TEGExtendedKalmanFilter(seebeck_alpha=seebeck_alpha, r_init=r_init)

    passos = 400
    historico_real = []
    historico_raw = []
    historico_ekf = []
    eixo_tempo = []

    for step in range(passos):
        # 1. Reconstrói a referência do Ground Truth (R_real) com base na anomalia do app.py
        r_real = 3.2 if (step % 200) > 180 else 1.6
        
        # 2. Obtém a telemetria gerada no app.py
        data = generate_physical_data(step, ekf)
        
        # 3. Calcula a Resistência Bruta Ruidosa (R_raw) via relação direta da Lei de Ohm/Seebeck
        delta_t = max(0.1, data["t_quente"] - data["t_frio"])
        v_oc = seebeck_alpha * delta_t
        i_amp = data["corrente_mA"] / 1000.0
        
        # Proteção contra divisão por zero em baixas correntes
        if i_amp > 1e-4:
            r_raw = (v_oc - data["tensao_V"]) / i_amp
        else:
            r_raw = r_init

        historico_real.append(r_real)
        historico_raw.append(r_raw)
        historico_ekf.append(data["r_interna_ekf"])
        eixo_tempo.append(step)

    # Configuração estética (Padrão ABNT / IEEE - 300 DPI)
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(10, 4.5), dpi=300)

    # 1. Sinal Bruto Ruidoso (R_raw)
    plt.plot(
        eixo_tempo,
        historico_raw,
        color="#8c564b",
        alpha=0.35,
        linewidth=1.0,
        label="Resistência Bruta Ruidosa ($R_{raw}$)",
    )

    # 2. Ground Truth - Degradação Real (R_real)
    plt.plot(
        eixo_tempo,
        historico_real,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Degradação Real / Referência ($R_{real}$)",
    )

    # 3. Estimativa do EKF (R_EKF)
    plt.plot(
        eixo_tempo,
        historico_ekf,
        color="#d62728",
        linewidth=2.2,
        label="Estimativa Adaptativa ($\hat{R}_{EKF}$)",
    )

    # Anotação técnica posicionada na área livre à esquerda da anomalia
    plt.annotate(
        "Injeção de Anomalia\n(Degradação Térmica)",
        xy=(181, 3.2),
        xytext=(115, 3.8),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff2ae", ec="#f39c12", alpha=0.8),
    )

    # Ajustes de eixos, títulos e limites
    plt.title(
        "Desempenho do Filtro de Kalman Estendido na Estimação de $R_{in}$",
        fontsize=12,
        pad=10,
    )
    plt.xlabel("Passos de Amostragem ($t$ [s])", fontsize=10)
    plt.ylabel("Resistência Interna ($\Omega$)", fontsize=10)
    plt.ylim(0.5, 4.8)
    plt.xlim(0, passos)
    plt.grid(True, linestyle=":", alpha=0.6)

    # Legenda realocada para o canto superior direito
    plt.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("grafico_42_desempenho_ekf.png", dpi=300)
    print("Gráfico gerado e salvo com sucesso como 'grafico_42_desempenho_ekf.png'!")

if __name__ == "__main__":
    gerar_grafico_artigo()