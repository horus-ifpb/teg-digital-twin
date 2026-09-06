import numpy as np
import matplotlib.pyplot as plt
from ekf import TEGExtendedKalmanFilter
from app import generate_physical_data

def gerar_grafico_telemetria_campo():
    # Inicializa o EKF para manter a execução contínua da simulação
    ekf = TEGExtendedKalmanFilter(seebeck_alpha=0.085, r_init=1.6)

    passos = 400
    eixo_tempo = []
    
    # Vetores de armazenamento da telemetria bruta
    t_quente_list = []
    t_frio_list = []
    delta_t_list = []
    tensao_list = []
    corrente_A_list = []
    potencia_mW_list = []

    # Coleta os dados telemétricos passo a passo do simulator
    for step in range(passos):
        data = generate_physical_data(step, ekf)
        
        t_q = data["t_quente"]
        t_f = data["t_frio"]
        dt = max(0.1, t_q - t_f)
        
        eixo_tempo.append(step)
        t_quente_list.append(t_q)
        t_frio_list.append(t_f)
        delta_t_list.append(dt)
        tensao_list.append(data["tensao_V"])
        corrente_A_list.append(data["corrente_mA"] / 1000.0)  # Converte mA para A
        potencia_mW_list.append(data["potencia_mW"])

    # Configuração de Estilo Acadêmico (300 DPI)
    plt.rcParams["font.family"] = "serif"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, dpi=300)

    # -------------------------------------------------------------------------
    # PAINEL (a): TELEMETRIA TÉRMICA (T_quente, T_frio, Delta_T)
    # -------------------------------------------------------------------------
    ax1.plot(eixo_tempo, t_quente_list, color="#d95f02", linewidth=1.5, label=r"Face Quente ($T_h$)")
    ax1.plot(eixo_tempo, t_frio_list, color="#7570b3", linewidth=1.5, label=r"Face Fria ($T_c$)")
    ax1.plot(eixo_tempo, delta_t_list, color="#1b9e77", linewidth=1.8, linestyle="--", label=r"Gradiente ($\Delta T$)")
    
    ax1.set_title("(a) Monitoramento Térmico Dinâmico da Bancada (1 Hz)", fontsize=11, loc="left", fontweight="bold")
    ax1.set_ylabel("Temperatura (°C)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax1.set_ylim(0, 110)

    # -------------------------------------------------------------------------
    # PAINEL (b): TELEMETRIA ELÉTRICA (Tensão, Corrente e Potência)
    # -------------------------------------------------------------------------
    ax2_twin = ax2.twinx()
    
    p1 = ax2.plot(eixo_tempo, tensao_list, color="#e7298a", linewidth=1.5, label=r"Tensão $V_{teg}$ (V)")
    p2 = ax2.plot(eixo_tempo, corrente_A_list, color="#66a61e", linewidth=1.5, label=r"Corrente $I_{teg}$ (A)")
    p3 = ax2_twin.plot(eixo_tempo, potencia_mW_list, color="#e6ab02", linewidth=1.2, linestyle="-.", label=r"Potência $P_{teg}$ (mW)")

    ax2.set_title("(b) Resposta Elétrica e Potência Gerada", fontsize=11, loc="left", fontweight="bold")
    ax2.set_xlabel("Passos de Amostragem ($t$ [s])", fontsize=10)
    ax2.set_ylabel("Tensão (V) / Corrente (A)", fontsize=10)
    ax2_twin.set_ylabel("Potência Instantânea (mW)", fontsize=10, color="#b38600")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Consolidação das legendas com dois eixos Y
    plots = p1 + p2 + p3
    labels = [l.get_label() for l in plots]
    ax2.legend(plots, labels, loc="upper left", frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig("grafico_41_telemetria_campo.png", dpi=300)
    print("Gráfico gerado com sucesso: 'grafico_41_telemetria_campo.png'!")

if __name__ == "__main__":
    gerar_grafico_telemetria_campo()