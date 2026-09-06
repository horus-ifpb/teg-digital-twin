# ==========================================
# Estágio 1: Builder (Compilação e Instalação)
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Instala ferramentas pesadas de compilação apenas no builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cria ambiente virtual isolado
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pré-compila o bytecode (.pyc) das dependências para acelerar imports no start
#RUN python -m compileall /opt/venv


# ==========================================
# Estágio 2: Runner (Imagem Final de Execução)
# ==========================================
FROM python:3.10-slim AS runner

# Variáveis de ambiente para otimização de runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copia apenas o ambiente virtual já compilado
COPY --from=builder /opt/venv /opt/venv

# Copia os arquivos do projeto
COPY . .

# Pré-compila o código da aplicação e define permissão de execução
RUN python -m compileall /app && \
    chmod +x run.sh

RUN python generate_dataset.py
RUN python grafico_43.py

CMD ["./run.sh"]