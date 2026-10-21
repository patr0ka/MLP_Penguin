"""Hiperparâmetros e Variáveis de Configuração da MLP e do Treinamento.

Edite as variáveis deste arquivo para testar diferentes comportamentos da rede!
Após modificar qualquer valor, execute novamente o treinamento para aplicar:
    python -m penguin_mlp.train

Exemplos de testes didáticos para apresentar:
1. Mudar o número de épocas (ex: EPOCHS = 10 ou EPOCHS = 50)
2. Mudar a taxa de aprendizado (ex: LEARNING_RATE = 0.01 ou LEARNING_RATE = 0.0001)
3. Mudar a arquitetura da MLP:
   - Rede mais profunda: HIDDEN_DIMS = [256, 128], DROPOUT_RATES = [0.30, 0.20]
   - Rede simples:       HIDDEN_DIMS = [64],      DROPOUT_RATES = [0.20]
"""

# ==============================================================================
# 1. REPRODUTIBILIDADE
# ==============================================================================
# Semente para os geradores pseudoaleatórios (Python, NumPy e PyTorch).
# Manter fixa garante que os mesmos resultados possam ser reproduzidos.
SEED = 42


# ==============================================================================
# 2. ARQUITETURA DA MLP (NEURÔNIOS E CAMADAS)
# ==============================================================================
# Neurônios de entrada:
# DEVE ser 512, pois coincide exatamente com o vetor de características da ResNet18.
INPUT_DIM = 512

# Camadas ocultas:
# Lista com a quantidade de neurônios em cada camada intermediária.
# Exemplo padrão: [128, 64] (2 camadas ocultas: 128 neurônios, depois 64 neurônios).
# Para testar uma camada única: [64]
# Para testar mais neurônios:   [256, 128]
HIDDEN_DIMS = [128, 64]

# Neurônios da camada de saída:
# DEVE ser 3, correspondendo às três espécies de pinguins (Adelie, Chinstrap, Gentoo).
NUM_CLASSES = 3

# Taxa de Dropout por camada oculta:
# Regularização estocástica para evitar overfitting.
# IMPORTANTE: Deve possuir a mesma quantidade de elementos que HIDDEN_DIMS!
# Exemplo para HIDDEN_DIMS = [128, 64] -> DROPOUT_RATES = [0.30, 0.20]
# Exemplo para HIDDEN_DIMS = [64]      -> DROPOUT_RATES = [0.20]
DROPOUT_RATES = [0.30, 0.20]


# ==============================================================================
# 3. HIPERPARÂMETROS DO TREINAMENTO
# ==============================================================================
# Número máximo de épocas de treinamento:
# É o total de passagens completas pelo dataset de treino.
# Exemplo: altere para 10 para um treino ultrarrápido, ou 40 para convergência completa.
EPOCHS = 40

# Tamanho do lote (Batch Size):
# Quantidade de amostras processadas antes de atualizar os pesos.
# Valores comuns: 16, 32, 64.
BATCH_SIZE = 32

# Taxa de aprendizado inicial (Learning Rate):
# Controla o tamanho do passo dado pelo otimizador AdamW na direção do gradiente.
# Valores recomendados para testar: 0.001 (padrão), 0.0005 (mais suave), 0.01 (mais rápido).
LEARNING_RATE = 0.001

# Decaimento de pesos (Weight Decay / Regularização L2):
# Penaliza pesos muito grandes para evitar que a rede decore o treino.
WEIGHT_DECAY = 0.0001

# Paciência do Early Stopping:
# Número de épocas consecutivas sem melhora na perda de validação antes de parar o treino.
EARLY_STOPPING_PATIENCE = 6


# ==============================================================================
# 4. INFERÊNCIA E DISPOSITIVO
# ==============================================================================
# Limiar de confiança para aviso de incerteza (de 0.0 a 1.0):
# Se a probabilidade da classe prevista for menor que esse valor, um alerta é emitido.
CONFIDENCE_THRESHOLD = 0.60

# Dispositivo de execução: 'auto' (detecta GPU/CUDA se houver, senão CPU), 'cpu' ou 'cuda'.
DEVICE = "auto"
