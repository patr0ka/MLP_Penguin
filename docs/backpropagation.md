# Fundamentação Matemática do Backpropagation na MLP

Este documento apresenta a dedução matemática e a conexão com o código em PyTorch e NumPy do algoritmo de **Retropropagação do Erro (Backpropagation)** utilizado para treinar a rede neural multicamadas (MLP).

---

## 1. O que é e para que serve o Backpropagation?

O aprendizado de uma rede neural supervisionada consiste em encontrar o conjunto de parâmetros (pesos \(W\) e vieses \(\mathbf{b}\)) que minimiza uma função de custo ou perda \(L\) em relação aos exemplos de treino.

O **Backpropagation** é a aplicação sistemática e eficiente da **Regra da Cadeia do Cálculo Diferencial** para calcular o gradiente da perda em relação a cada peso e viés da rede:
$$\nabla_\theta L = \left[ \frac{\partial L}{\partial W^{(1)}}, \frac{\partial L}{\partial \mathbf{b}^{(1)}}, \dots, \frac{\partial L}{\partial W^{(L)}}, \frac{\partial L}{\partial \mathbf{b}^{(L)}} \right]$$

> **Nota Conceitual Fundamental:**
> - O **Backpropagation calcula os gradientes** (para onde e com qual intensidade a perda varia).
> - O **Otimizador (AdamW)** utiliza esses gradientes para **atualizar os parâmetros**.
> O backpropagation não altera pesos diretamente; ele informa ao otimizador a taxa de variação instantânea do erro.

---

## 2. Passagem Direta (Forward Pass)

Considere um vetor de entrada \(\mathbf{x} \in \mathbb{R}^{512}\) (extraído pela CNN congelada) e seu rótulo verdadeiro \(y \in \{0, 1, 2\}\).

### Camada Oculta 1 (\(512 \to 128\)):
$$\mathbf{z}^{(1)} = W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}, \quad W^{(1)} \in \mathbb{R}^{128 \times 512}, \; \mathbf{b}^{(1)} \in \mathbb{R}^{128}$$
$$\mathbf{a}^{(1)} = \operatorname{ReLU}(\mathbf{z}^{(1)}) = \max(0, \mathbf{z}^{(1)})$$
$$\tilde{\mathbf{a}}^{(1)} = \mathbf{a}^{(1)} \odot \mathbf{m}^{(1)}, \quad \mathbf{m}^{(1)} \sim \operatorname{Bernoulli}(1 - p_1) \cdot \frac{1}{1 - p_1}$$

### Camada Oculta 2 (\(128 \to 64\)):
$$\mathbf{z}^{(2)} = W^{(2)} \tilde{\mathbf{a}}^{(1)} + \mathbf{b}^{(2)}, \quad W^{(2)} \in \mathbb{R}^{64 \times 128}, \; \mathbf{b}^{(2)} \in \mathbb{R}^{64}$$
$$\mathbf{a}^{(2)} = \operatorname{ReLU}(\mathbf{z}^{(2)}) = \max(0, \mathbf{z}^{(2)})$$
$$\tilde{\mathbf{a}}^{(2)} = \mathbf{a}^{(2)} \odot \mathbf{m}^{(2)}, \quad \mathbf{m}^{(2)} \sim \operatorname{Bernoulli}(1 - p_2) \cdot \frac{1}{1 - p_2}$$

### Camada de Saída (\(64 \to 3\)):
$$\mathbf{z}^{(3)} = W^{(3)} \tilde{\mathbf{a}}^{(2)} + \mathbf{b}^{(3)}, \quad W^{(3)} \in \mathbb{R}^{3 \times 64}, \; \mathbf{b}^{(3)} \in \mathbb{R}^{3}$$
Onde \(\mathbf{z}^{(3)} = [z_0, z_1, z_2]^T\) são os **logits brutos**.

---

## 3. Função de Perda (Cross-Entropy Loss) e Softmax

A probabilidade estimada para a classe \(k\) é dada pela função Softmax:
$$p_k = \frac{e^{z_k}}{\sum_{j=0}^{2} e^{z_j}}$$

A perda de Entropia Cruzada para uma amostra cujo rótulo correto é \(y\) é definida como a log-verossimilhança negativa:
$$L = -\sum_{k=0}^{2} y_k \log(p_k) = -\log(p_y)$$
Onde \(y_k = 1\) se \(k = y\) e \(y_k = 0\) caso contrário (representação *one-hot*).

---

## 4. Gradiente da Camada de Saída (\(\frac{\partial L}{\partial \mathbf{z}^{(3)}}\))

Esta é uma das propriedades mais elegantes do aprendizado profundo: a derivada combinada da perda de entropia cruzada com a ativação Softmax resulta em uma diferença linear direta entre a probabilidade prevista e o alvo:

$$\frac{\partial L}{\partial z_k} = p_k - y_k$$

### Demonstração Matemática:
Como \(L = -\sum_j y_j \log(p_j)\), aplicando a regra da cadeia:
$$\frac{\partial L}{\partial z_k} = -\sum_j \frac{y_j}{p_j} \frac{\partial p_j}{\partial z_k}$$

A derivada de \(p_j = \frac{e^{z_j}}{\sum_m e^{z_m}}\) tem dois casos:
1. Se \(j = k\):
   $$\frac{\partial p_k}{\partial z_k} = \frac{e^{z_k}(\sum e^{z_m}) - (e^{z_k})^2}{(\sum e^{z_m})^2} = p_k(1 - p_k)$$
2. Se \(j \neq k\):
   $$\frac{\partial p_j}{\partial z_k} = \frac{0 - e^{z_j}e^{z_k}}{(\sum e^{z_m})^2} = -p_j p_k$$

Substituindo na expressão da perda:
$$\frac{\partial L}{\partial z_k} = -\frac{y_k}{p_k} \cdot p_k(1 - p_k) - \sum_{j \neq k} \frac{y_j}{p_j} (-p_j p_k)$$
$$\frac{\partial L}{\partial z_k} = -y_k(1 - p_k) + \sum_{j \neq k} y_j p_k = -y_k + y_k p_k + p_k \sum_{j \neq k} y_j$$
Como \(\sum_j y_j = 1\) (vetor *one-hot*), temos \(y_k + \sum_{j \neq k} y_j = 1\), logo:
$$\frac{\partial L}{\partial z_k} = p_k - y_k$$

Definimos o erro da camada de saída como:
$$\boldsymbol{\delta}^{(3)} = \mathbf{p} - \mathbf{y}$$

---

## 5. Propagação Reversa nas Camadas Ocultas

### Gradientes dos Parâmetros da Última Camada (\(W^{(3)}\) e \(\mathbf{b}^{(3)}\)):
$$\frac{\partial L}{\partial W^{(3)}} = \boldsymbol{\delta}^{(3)} (\tilde{\mathbf{a}}^{(2)})^T$$
$$\frac{\partial L}{\partial \mathbf{b}^{(3)}} = \boldsymbol{\delta}^{(3)}$$

### Gradiente Retropropagado para a Camada 2:
$$\boldsymbol{\delta}^{(2)} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} = \left( (W^{(3)})^T \boldsymbol{\delta}^{(3)} \right) \odot \mathbf{m}^{(2)} \odot f'(\mathbf{z}^{(2)})$$
Onde a derivada da ReLU \(f'(z)\) é uma função indicadora simples:
$$f'(z) = \begin{cases} 1, & \text{se } z > 0 \\ 0, & \text{se } z \le 0 \end{cases}$$

### Gradientes dos Parâmetros da Camada 2 (\(W^{(2)}\) e \(\mathbf{b}^{(2)}\)):
$$\frac{\partial L}{\partial W^{(2)}} = \boldsymbol{\delta}^{(2)} (\tilde{\mathbf{a}}^{(1)})^T$$
$$\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)}$$

### Gradiente Retropropagado para a Camada 1:
$$\boldsymbol{\delta}^{(1)} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} = \left( (W^{(2)})^T \boldsymbol{\delta}^{(2)} \right) \odot \mathbf{m}^{(1)} \odot f'(\mathbf{z}^{(1)})$$

### Gradientes dos Parâmetros da Camada 1 (\(W^{(1)}\) e \(\mathbf{b}^{(1)}\)):
$$\frac{\partial L}{\partial W^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T$$
$$\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}$$

---

## 6. Atualização dos Parâmetros: SGD vs AdamW

### Descida do Gradiente Estocástico (SGD clássico):
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L$$

### Otimizador AdamW (Utilizado na Produção):
O AdamW mantém médias móveis exponenciais dos gradientes de primeira ordem (momento) e de segunda ordem (variância não centralizada):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Com correção de viés para as primeiras iterações:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

E atualização desacoplada do decaimento de pesos (*weight decay* \(\lambda\)):
$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

---

## 7. Como o PyTorch Executa o Backpropagation

No loop de treinamento (`penguin_mlp/train.py`), o processo é orquestrado pelas seguintes chamadas:

```python
# 1. Zera gradientes acumulados de iterações anteriores
optimizer.zero_grad()

# 2. Passagem direta: constrói dinamicamente o grafo computacional
logits = mlp(features)

# 3. Calcula o escalar de perda
loss = criterion(logits, targets)

# 4. Aciona a retropropagação automática via autograd:
#    Percorre o grafo do nó 'loss' até as folhas (parâmetros da MLP),
#    armazenando o gradiente correspondente no atributo .grad de cada tensor.
loss.backward()

# 5. Otimizador lê os valores em param.grad e aplica as regras do AdamW
optimizer.step()
```

Como os tensores da CNN foram definidos com `requires_grad = False` e executados sob `torch.no_grad()`, o autograd encerra a retropropagação nas características de entrada da MLP, sem gastar memória nem calcular gradientes para a ResNet18.
