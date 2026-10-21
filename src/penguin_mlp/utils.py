"""Módulo de utilitários compartilhados do projeto penguin_mlp.

Contém funções para fixação de sementes pseudoaleatórias, detecção e seleção
de dispositivos de processamento (CPU/GPU), contagem de parâmetros treináveis
e salvamento/carregamento seguro de checkpoints com metadados completos.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fixa a semente de geradores pseudoaleatórios para garantir reprodutibilidade.

    Configura Python built-in random, NumPy e PyTorch (tanto CPU quanto CUDA se disponível).
    Habilita operações determinísticas no PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Semente global definida para: %d", seed)


def resolve_device(device_setting: str = "auto") -> torch.device:
    """Resolve o dispositivo de execução (CPU ou CUDA).

    Se 'auto', seleciona GPU (CUDA) se disponível, caso contrário CPU.
    """
    setting = (device_setting or "auto").strip().lower()
    if setting == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif setting.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requisitado, mas indisponível no sistema. Alternando para CPU.")
            dev = torch.device("cpu")
        else:
            dev = torch.device(setting)
    else:
        dev = torch.device("cpu")
    return dev


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Calcula e retorna (parâmetros_treináveis, parâmetros_totais) de um módulo PyTorch."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def log_model_summary(model: nn.Module, model_name: str = "Modelo") -> None:
    """Imprime um resumo amigável da quantidade de parâmetros de um modelo."""
    trainable, total = count_parameters(model)
    logger.info(
        "[%s] Parâmetros treináveis: %s | Parâmetros totais: %s",
        model_name,
        f"{trainable:,}".replace(",", "."),
        f"{total:,}".replace(",", "."),
    )


def save_checkpoint(
    checkpoint_path: str | Path,
    mlp_state_dict: Dict[str, Any],
    epoch: int,
    val_loss: float,
    val_acc: float,
    val_f1_macro: float,
    config: Any,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Salva o checkpoint do modelo contendo pesos treinados e metadados detalhados."""
    dest = Path(checkpoint_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "epoch": epoch,
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_f1_macro": float(val_f1_macro),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "classes": list(class_to_idx.keys()),
        "cnn": {
            "architecture": config.cnn.architecture,
            "feature_dim": config.cnn.feature_dim,
            "pretrained": config.cnn.pretrained,
            "frozen": config.cnn.freeze,
        },
        "mlp": {
            "input_dim": config.mlp.input_dim,
            "hidden_dims": config.mlp.hidden_dims,
            "num_classes": config.mlp.num_classes,
            "dropout": config.mlp.dropout,
        },
        "preprocessing": {
            "image_size": config.dataset.image_size,
            "preserve_aspect_ratio": config.dataset.preserve_aspect_ratio,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "training": {
            "seed": config.seed,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "optimizer": "AdamW",
            "loss_function": "CrossEntropyLoss",
        },
    }
    if extra_meta:
        metadata.update(extra_meta)

    payload = {
        "mlp_state_dict": mlp_state_dict,
        "metadata": metadata,
    }
    torch.save(payload, dest)
    logger.info("Checkpoint salvo com sucesso em: %s (Época %d | Loss Val: %.4f | F1: %.4f)", dest, epoch, val_loss, val_f1_macro)
    return dest


def load_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Carrega o checkpoint salvo, retornando (mlp_state_dict, metadata)."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de checkpoint não encontrado no caminho: '{path}'. "
            f"Certifique-se de executar o treinamento antes de realizar inferências."
        )

    target_device = device or torch.device("cpu")
    payload = torch.load(path, map_location=target_device)
    if "mlp_state_dict" not in payload or "metadata" not in payload:
        raise ValueError(
            f"O checkpoint '{path}' possui estrutura incompatível. Chaves esperadas: "
            f"'mlp_state_dict' e 'metadata'."
        )
    return payload["mlp_state_dict"], payload["metadata"]
