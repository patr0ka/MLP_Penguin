"""Módulo de extração de características visuais via CNN pré-treinada (ResNet18).

Implementa a CNN pré-treinada no ImageNet como extratora fixa de representações
visuais de alta dimensionalidade (512 características).

Papel no projeto:
A CNN atua estritamente como um extrator de características congelado. Ela NÃO
substitui a MLP e NÃO aprende a classificação final. Toda a classificação e o
aprendizado de máquina deste projeto ocorrem na MLP.

Mecanismos de congelamento implementados:
- `cnn.eval()`: Coloca todas as camadas (como BatchNorm) em modo de avaliação,
  impedindo a atualização de médias e variâncias móveis.
- `param.requires_grad = False`: Remove explicitamente todos os tensores de parâmetros
  do grafo de diferenciação automática, garantindo que nenhum gradiente seja acumulado.
- `torch.no_grad()`: Durante a extração, evita alocação de memória para nós intermediários
  do autograd.
"""

from __future__ import annotations

import logging
from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger(__name__)


class ResNet18FeatureExtractor(nn.Module):
    """Extrator de características baseado na ResNet18 pré-treinada no ImageNet.

    Substitui a camada linear final (`fc`) de 1000 classes por `nn.Identity()`,
    retornando o vetor de 512 características após o Adaptive Average Pooling.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        self.feature_dim = 512

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
            logger.info("ResNet18 inicializada com pesos pré-treinados do ImageNet (DEFAULT).")
        else:
            self.model = resnet18(weights=None)
            logger.info("ResNet18 inicializada com pesos aleatórios (sem pré-treino).")

        # Substituir o classificador final de 1000 classes pela identidade
        # Isso preserva as camadas convolucionais e o pooling global, produzindo [B, 512]
        self.model.fc = nn.Identity()

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        """Congela todos os parâmetros da rede convolucional."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Todos os parâmetros da ResNet18 foram congelados (requires_grad = False).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem direta extraindo o vetor de características [B, 512]."""
        # Garante que durante a extração não sejam calculados gradientes para a CNN
        with torch.no_grad():
            features = self.model(x)  # [B, 512]
        return features

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Método explícito para extração de lote de imagens."""
        self.model.eval()
        return self.forward(images)


def build_backbone(pretrained: bool = True, freeze: bool = True, device: Optional[torch.device] = None) -> ResNet18FeatureExtractor:
    """Instancia e configura o extrator de características no dispositivo especificado."""
    extractor = ResNet18FeatureExtractor(pretrained=pretrained, freeze=freeze)
    if device is not None:
        extractor.to(device)
    return extractor
