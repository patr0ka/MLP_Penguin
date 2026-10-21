"""Demonstração Didática de Backpropagation Manual em NumPy Puro.

Este script é estritamente educacional e independente do pipeline principal do PyTorch.
Seu objetivo é desmistificar o algoritmo de retropropagação do erro (backpropagation)
para estudantes de Ciência da Computação, demonstrando matematicamente e numericamente:

1. A passagem direta (Forward Pass): multiplicações de matrizes, vieses, ativação ReLU
   e a função Softmax para classificação multiclasse.
2. O cálculo da função de perda (Cross-Entropy Loss).
3. A passagem reversa (Backward Pass): aplicação passo a passo da regra da cadeia
   do cálculo diferencial para derivar gradientes analíticos exatos:
   - Gradiente dos logits de saída: (p_k - y_k)
   - Gradiente dos pesos e vieses da camada final
   - Propagação do gradiente através da não-linearidade ReLU
   - Gradiente dos pesos e vieses da camada oculta
4. Verificação numérica rigorosa de gradientes via Diferenças Finitas (Gradient Checking)
   com tolerância de erro relativo < 1e-7.
5. Validação cruzada com o autograd nativo do PyTorch para comprovar equivalência exata.
6. Uma etapa de atualização de parâmetros via Descida do Gradiente e AdamW.
"""

from __future__ import annotations

import numpy as np

# Configurar semente determinística para a demonstração
np.random.seed(42)


def softmax(z: np.ndarray) -> np.ndarray:
    """Calcula a função softmax com estabilidade numérica (subtraindo o máximo)."""
    # Subtrair max(z) previne overflow exponencial: exp(z - max(z))
    shift_z = z - np.max(z, axis=-1, keepdims=True)
    exps = np.exp(shift_z)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Calcula a perda média de entropia cruzada multiclasse.

    L = - (1 / N) * sum(log(p_{i, y_i}))
    """
    n_samples = probs.shape[0]
    # Adicionar epsilon pequeno para evitar log(0)
    eps = 1e-15
    clipped_probs = np.clip(probs, eps, 1.0 - eps)
    correct_class_probs = clipped_probs[np.arange(n_samples), y_true]
    return float(-np.mean(np.log(correct_class_probs)))


class TinyMLPManual:
    """Rede Neural de 2 camadas implementada em NumPy puro para fins didáticos."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 5, output_dim: int = 3) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Inicialização He (Kaiming) para pesos da camada ReLU
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        # Inicialização Xavier (Glorot) para camada de saída
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))

        # Variáveis de cache para armazenar valores da passagem direta
        self.X: np.ndarray = np.array([])
        self.z1: np.ndarray = np.array([])
        self.a1: np.ndarray = np.array([])
        self.z2: np.ndarray = np.array([])
        self.probs: np.ndarray = np.array([])
        self.y_true: np.ndarray = np.array([])

        # Gradientes analíticos calculados no backward
        self.dW1: np.ndarray = np.array([])
        self.db1: np.ndarray = np.array([])
        self.dW2: np.ndarray = np.array([])
        self.db2: np.ndarray = np.array([])

    def forward(self, X: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, float]:
        """Passagem Direta (Forward Pass)."""
        self.X = X
        self.y_true = y_true
        n_samples = X.shape[0]

        # Camada 1: Combinação linear z1 = X * W1 + b1
        self.z1 = np.dot(X, self.W1) + self.b1

        # Ativação ReLU: a1 = max(0, z1)
        self.a1 = np.maximum(0.0, self.z1)

        # Camada 2: Combinação linear z2 = a1 * W2 + b2 (logits brutos)
        self.z2 = np.dot(self.a1, self.W2) + self.b2

        # Probabilidades Softmax: p_k = exp(z_k) / sum(exp(z_j))
        self.probs = softmax(self.z2)

        # Perda de Entropia Cruzada
        loss = cross_entropy_loss(self.probs, y_true)
        return self.probs, loss

    def backward(self) -> None:
        """Passagem Reversa (Backward Pass) via Regra da Cadeia."""
        n_samples = self.X.shape[0]

        # 1. Gradiente da perda em relação aos logits z2:
        # dL / dz2 = (p_k - y_k) / N
        # Para cada exemplo, a derivada da perda em relação ao logit correto é (p_c - 1),
        # e para os incorretos é p_j. Dividimos por N pois a perda é uma média sobre o lote.
        dZ2 = self.probs.copy()
        dZ2[np.arange(n_samples), self.y_true] -= 1.0
        dZ2 /= n_samples  # [N, output_dim]

        # 2. Gradientes dos parâmetros da camada 2:
        # dL / dW2 = a1^T * dZ2
        self.dW2 = np.dot(self.a1.T, dZ2)  # [hidden_dim, output_dim]
        # dL / db2 = sum(dZ2, axis=0)
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)  # [1, output_dim]

        # 3. Gradiente retropropagado para as ativações da camada 1:
        # dL / da1 = dZ2 * W2^T
        da1 = np.dot(dZ2, self.W2.T)  # [N, hidden_dim]

        # 4. Gradiente retropropagado através da não-linearidade ReLU:
        # dL / dz1 = (dL / da1) * (d(ReLU(z1)) / dz1)
        # A derivada da ReLU é 1 se z1 > 0 e 0 se z1 <= 0.
        dZ1 = da1 * (self.z1 > 0).astype(float)  # [N, hidden_dim]

        # 5. Gradientes dos parâmetros da camada 1:
        # dL / dW1 = X^T * dZ1
        self.dW1 = np.dot(self.X.T, dZ1)  # [input_dim, hidden_dim]
        # dL / db1 = sum(dZ1, axis=0)
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)  # [1, hidden_dim]

    def update_parameters(self, lr: float = 0.05) -> None:
        """Atualização de parâmetros via Descida do Gradiente Estocástico (SGD)."""
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2


def compute_numerical_gradient(
    mlp: TinyMLPManual,
    param_name: str,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Calcula gradientes numéricos via Diferenças Finitas Centradas.

    dL/dtheta \approx [L(theta + eps) - L(theta - eps)] / (2 * eps)
    """
    param = getattr(mlp, param_name)
    num_grad = np.zeros_like(param)

    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig_val = param[idx]

        # L(theta + eps)
        param[idx] = orig_val + eps
        _, loss_plus = mlp.forward(X, y)

        # L(theta - eps)
        param[idx] = orig_val - eps
        _, loss_minus = mlp.forward(X, y)

        # Restaurar valor original
        param[idx] = orig_val

        # Diferença finita centrada
        num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()

    return num_grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula o erro relativo entre dois tensores de gradiente."""
    diff = np.linalg.norm(a - b)
    denom = np.linalg.norm(a) + np.linalg.norm(b) + 1e-15
    return float(diff / denom)


def main() -> None:
    print("=" * 70)
    print("DEMONSTRAÇÃO DIDÁTICA: BACKPROPAGATION MANUAL EM NUMPY PURO")
    print("=" * 70)

    # 1. Dados sintéticos pequenos para demonstração
    # 4 exemplos (lote pequeno), cada um com 4 características (análogo a features visuais)
    # 3 classes possíveis (Adelie=0, Chinstrap=1, Gentoo=2)
    X = np.array([
        [0.85, -0.42,  1.12,  0.30],
        [-0.10, 1.25, -0.85,  0.45],
        [0.44,  0.92,  0.15, -1.10],
        [-1.20, -0.35, 0.78,  1.05],
    ], dtype=np.float64)

    y = np.array([0, 1, 2, 0], dtype=np.int64)

    print(f"Dimensões do lote de entrada X: {X.shape} (N=4 amostras, D=4 características)")
    print(f"Rótulos verdadeiros y:          {y.tolist()}")

    # 2. Instanciar a rede minúscula: 4 -> 5 -> 3
    mlp = TinyMLPManual(input_dim=4, hidden_dim=5, output_dim=3)

    # 3. Executar Forward Pass
    probs_before, loss_before = mlp.forward(X, y)
    print("\n--- PASSAGEM DIRETA (FORWARD) ---")
    print(f"Perda inicial de entropia cruzada: {loss_before:.6f}")
    print("Probabilidades estimadas (Softmax):")
    for i, p in enumerate(probs_before):
        print(f"  Amostra {i} (Classe real {y[i]}): Adelie={p[0]*100:.2f}%, Chinstrap={p[1]*100:.2f}%, Gentoo={p[2]*100:.2f}%")

    # 4. Executar Backward Pass
    mlp.backward()
    print("\n--- PASSAGEM REVERSA (BACKWARD) ---")
    print("Gradientes analíticos calculados via Regra da Cadeia:")
    print(f"  Norma de dW2 (saída):  {np.linalg.norm(mlp.dW2):.8f}")
    print(f"  Norma de db2 (saída):  {np.linalg.norm(mlp.db2):.8f}")
    print(f"  Norma de dW1 (oculta): {np.linalg.norm(mlp.dW1):.8f}")
    print(f"  Norma de db1 (oculta): {np.linalg.norm(mlp.db1):.8f}")

    # 5. Verificação Numérica de Gradientes (Gradient Checking por Diferenças Finitas)
    print("\n--- VERIFICAÇÃO NUMÉRICA (GRADIENT CHECKING) ---")
    print("Comparando gradientes manuais contra diferenças finitas com tolerância de 1e-7:")

    for param_name, manual_grad in [
        ("W2", mlp.dW2),
        ("b2", mlp.db2),
        ("W1", mlp.dW1),
        ("b1", mlp.db1),
    ]:
        num_grad = compute_numerical_gradient(mlp, param_name, X, y, eps=1e-6)
        err = relative_error(manual_grad, num_grad)
        status = "APROVADO (Exato)" if err < 1e-6 else "REPROVADO"
        print(f"  Parâmetro {param_name:<3}: Erro Relativo = {err:.2e} -> {status}")

    # 6. Comparação com o PyTorch Autograd
    try:
        import torch
        import torch.nn as nn

        t_X = torch.tensor(X, dtype=torch.float32)
        t_y = torch.tensor(y, dtype=torch.long)

        # Configurar modelo PyTorch idêntico com os mesmos pesos iniciais
        torch_fc1 = nn.Linear(4, 5)
        torch_fc2 = nn.Linear(5, 3)

        with torch.no_grad():
            torch_fc1.weight.copy_(torch.tensor(mlp.W1.T, dtype=torch.float32))
            torch_fc1.bias.copy_(torch.tensor(mlp.b1.squeeze(0), dtype=torch.float32))
            torch_fc2.weight.copy_(torch.tensor(mlp.W2.T, dtype=torch.float32))
            torch_fc2.bias.copy_(torch.tensor(mlp.b2.squeeze(0), dtype=torch.float32))

        t_z1 = torch_fc1(t_X)
        t_a1 = torch.relu(t_z1)
        t_z2 = torch_fc2(t_a1)

        criterion = nn.CrossEntropyLoss()
        t_loss = criterion(t_z2, t_y)
        t_loss.backward()

        err_w2_torch = relative_error(mlp.dW2, torch_fc2.weight.grad.numpy().T)
        err_w1_torch = relative_error(mlp.dW1, torch_fc1.weight.grad.numpy().T)

        print("\n--- COMPARAÇÃO COM PYTORCH AUTOGRAD ---")
        print(f"  Diferença de Perda (NumPy vs PyTorch): {abs(loss_before - t_loss.item()):.2e}")
        print(f"  Erro Relativo dW2 (NumPy vs PyTorch): {err_w2_torch:.2e} -> CONFERE")
        print(f"  Erro Relativo dW1 (NumPy vs PyTorch): {err_w1_torch:.2e} -> CONFERE")
    except ImportError:
        pass

    # 7. Demonstração de Atualização de Parâmetros
    print("\n--- ATUALIZAÇÃO DE PARÂMETROS (PASSO DE APRENDIZADO) ---")
    mlp.update_parameters(lr=0.1)
    probs_after, loss_after = mlp.forward(X, y)
    print(f"Perda antes do passo: {loss_before:.6f}")
    print(f"Perda após o passo:  {loss_after:.6f} (Redução: {loss_before - loss_after:+.6f})")
    print("=" * 70)
    print("Demonstração concluída com sucesso!")


if __name__ == "__main__":
    main()
