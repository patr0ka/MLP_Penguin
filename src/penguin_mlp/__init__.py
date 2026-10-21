"""Pacote penguin_mlp: Classificador de Espécies de Pinguins com CNN + MLP.

Combina extração de características visuais via CNN pré-treinada (ResNet18 congelada)
com uma rede neural multicamadas (MLP) desenvolvida e treinada especificamente
para a classificação das espécies Adelie, Chinstrap e Gentoo.
"""

__version__ = "1.0.0"
__all__ = [
    "config",
    "data",
    "transforms",
    "backbone",
    "model",
    "utils",
]
