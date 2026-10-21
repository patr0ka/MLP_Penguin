"""Módulo da Rede Neural Multicamadas (Multilayer Perceptron - MLP).

Implementa a arquitetura de classificação de espécies de pinguins treinada
a partir dos vetores de 512 características extraídos pela CNN ResNet18.

Arquitetura padrão inicial:
    Entrada:    512 características visuais
    Camada 1:   Linear(512, 128) -> ReLU -> Dropout(p=0.30)
    Camada 2:   Linear(128, 64)  -> ReLU -> Dropout(p=0.20)
    Saída:      Linear(64, 3)    -> 3 logits não normalizados
    Inferência: Softmax(logits)  -> Probabilidades das 3 espécies

Contagem de parâmetros treináveis da arquitetura inicial:
    Camada 1: (512 * 128) + 128 = 65.664 parâmetros
    Camada 2: (128 * 64)  + 64  =  8.256 parâmetros
    Camada 3: (64 * 3)    + 3   =    195 parâmetros
    Total:                        74.115 parâmetros treináveis
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PenguinClassifierMLP(nn.Module):
    """Rede Neural Multicamadas para classificação das três espécies de pinguins.

    A rede recebe vetores densos de características visuais e calcula logits
    associados a cada uma das três espécies canônicas: Adelie (0), Chinstrap (1)
    e Gentoo (2).
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 3,
        dropout_rates: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.num_classes = num_classes
        self.dropout_rates = dropout_rates if dropout_rates is not None else [0.30, 0.20]

        if len(self.dropout_rates) != len(self.hidden_dims):
            raise ValueError(
                f"Quantidade de taxas de dropout ({len(self.dropout_rates)}) deve coincidir "
                f"com a quantidade de camadas ocultas ({len(self.hidden_dims)})."
            )

        # Construção dinâmica das camadas densas
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim, drop_rate in zip(self.hidden_dims, self.dropout_rates):
            # Transformação linear afim: z = x * W^T + b
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Ativação não-linear ReLU: f(z) = max(0, z)
            # Sem ativações não-lineares, múltiplas camadas lineares consecutivas
            # colapsariam em uma única transformação linear global.
            layers.append(nn.ReLU())
            # Regularização Dropout: desativa aleatoriamente neurônios no treino
            if drop_rate > 0.0:
                layers.append(nn.Dropout(p=drop_rate))
            prev_dim = hidden_dim

        # Camada final de saída linear (produz os logits)
        # Nota didática: NÃO aplicamos Softmax aqui no modelo. A função
        # nn.CrossEntropyLoss do PyTorch combina LogSoftmax + NLLLoss internamente
        # de forma numericamente estável através do algoritmo LogSumExp.
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a propagação direta (forward pass) da MLP.

        Args:
            x: Tensor com shape [batch_size, input_dim] contendo características.

        Returns:
            logits: Tensor com shape [batch_size, num_classes].
        """
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"Dimensão de entrada incorreta para a MLP: esperado [batch_size, {self.input_dim}], "
                f"mas recebido {list(x.shape)}."
            )
        return self.network(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula probabilidades estimadas [0, 1] aplicando Softmax aos logits.

        Utilizado na inferência e avaliação final.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def count_parameters(self) -> Tuple[int, int]:
        """Retorna (parâmetros_treináveis, parâmetros_totais)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def build_mlp(
    input_dim: int = 512,
    hidden_dims: Optional[List[int]] = None,
    num_classes: int = 3,
    dropout_rates: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> PenguinClassifierMLP:
    """Instancia a MLP e a transfere para o dispositivo especificado."""
    mlp = PenguinClassifierMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rates=dropout_rates,
    )
    if device is not None:
        mlp.to(device)
    return mlp
