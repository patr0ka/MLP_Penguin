"""Módulo de treinamento da MLP utilizando características extraídas pela CNN congelada.

Implementa o loop completo de treinamento, validação por época, agendamento de taxa
de aprendizado, parada antecipada (early stopping) e persistência de checkpoints
e gráficos de evolução de curvas de aprendizado.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from penguin_mlp.backbone import build_backbone
from penguin_mlp.config import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    ProjectConfig,
    load_config,
)
from penguin_mlp.data import create_dataloaders
from penguin_mlp.model import build_mlp
from penguin_mlp.utils import (
    log_model_summary,
    resolve_device,
    save_checkpoint,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_one_epoch(
    cnn: nn.Module,
    mlp: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Executa uma época de treinamento.

    Garante que a CNN opere em modo de avaliação e sem gradientes,
    enquanto a MLP atualiza seus pesos via backpropagation.
    """
    mlp.train()
    cnn.eval()

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets, _ in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        # 1. Extração de características pela CNN congelada (sem gradientes para a CNN)
        with torch.no_grad():
            features = cnn(images)  # [B, 512]

        # 2. Propagação direta na MLP
        logits = mlp(features)  # [B, 3]

        # 3. Cálculo da perda de entropia cruzada
        loss = criterion(logits, targets)

        # 4. Limpeza de gradientes da iteração anterior
        optimizer.zero_grad()

        # 5. Retropropagação do erro (backpropagation) na MLP
        loss.backward()

        # 6. Atualização dos parâmetros da MLP pelo otimizador
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate_epoch(
    cnn: nn.Module,
    mlp: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Avalia o modelo em um DataLoader (validação ou teste) sem calcular gradientes."""
    mlp.eval()
    cnn.eval()

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        features = cnn(images)
        logits = mlp(features)
        loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    """Gera e salva os gráficos de curvas de treinamento e validação."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Curva de Perda (Loss)
    axes[0].plot(epochs, history["train_loss"], label="Treino", marker="o", markersize=3, color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], label="Validação", marker="s", markersize=3, color="#ff7f0e")
    axes[0].set_title("Evolução da Perda (CrossEntropyLoss)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Perda")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Curva de Acurácia
    axes[1].plot(epochs, history["train_acc"], label="Treino", marker="o", markersize=3, color="#1f77b4")
    axes[1].plot(epochs, history["val_acc"], label="Validação", marker="s", markersize=3, color="#ff7f0e")
    axes[1].set_title("Evolução da Acurácia")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # Curva de F1-Score Macro
    axes[2].plot(epochs, history["train_f1"], label="Treino", marker="o", markersize=3, color="#1f77b4")
    axes[2].plot(epochs, history["val_f1"], label="Validação", marker="s", markersize=3, color="#ff7f0e")
    axes[2].set_title("Evolução do F1 Macro")
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("F1 Macro")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Gráficos de curvas de aprendizado salvos em: %s", output_path)


def run_training(
    config_path: Optional[str | Path] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Executa o pipeline completo de treinamento e validação."""
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    epochs = epochs_override or cfg.training.epochs
    batch_size = batch_size_override or cfg.training.batch_size
    learning_rate = lr_override or cfg.training.learning_rate
    device_str = device_override or cfg.training.device
    device = resolve_device(device_str)

    logger.info("=" * 60)
    logger.info("INICIANDO TREINAMENTO DA MLP COM CNN CONGELADA")
    logger.info("Dispositivo: %s | Semente: %d | Épocas máx: %d | Batch: %d", device, cfg.seed, epochs, batch_size)
    logger.info("=" * 60)

    # 1. Carregar DataLoaders
    train_loader, val_loader, _ = create_dataloaders(
        split_dir=cfg.dataset.split_dir,
        batch_size=batch_size,
        image_size=cfg.dataset.image_size,
        preserve_aspect_ratio=cfg.dataset.preserve_aspect_ratio,
    )

    # 2. Instanciar CNN congelada e MLP
    cnn = build_backbone(pretrained=cfg.cnn.pretrained, freeze=cfg.cnn.freeze, device=device)
    mlp = build_mlp(
        input_dim=cfg.mlp.input_dim,
        hidden_dims=cfg.mlp.hidden_dims,
        num_classes=cfg.mlp.num_classes,
        dropout_rates=cfg.mlp.dropout,
        device=device,
    )

    log_model_summary(cnn, "CNN Backbone (ResNet18 Congelada)")
    log_model_summary(mlp, "MLP Classificadora")

    # 3. Função de perda e otimizador
    # Notar que passamos APENAS os parâmetros da MLP para o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        mlp.parameters(),
        lr=learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Agendador de taxa de aprendizado (reduz taxa quando a perda de validação estagna)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.scheduler_factor,
        patience=cfg.training.scheduler_patience,
        min_lr=cfg.training.min_lr,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state_dict: Optional[Dict[str, Any]] = None
    best_val_metrics: Dict[str, float] = {}

    checkpoint_dest = Path(cfg.artifacts.best_checkpoint_path)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            cnn, mlp, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate_epoch(
            cnn, mlp, val_loader, criterion, device
        )

        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(curr_lr)

        logger.info(
            "Época [%02d/%02d] Treino Loss: %.4f, Acc: %.4f, F1: %.4f | "
            "Val Loss: %.4f, Acc: %.4f, F1: %.4f | LR: %.6f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            train_f1,
            val_loss,
            val_acc,
            val_f1,
            curr_lr,
        )

        # Checagem de melhor modelo e early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            best_val_metrics = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }

            save_checkpoint(
                checkpoint_path=checkpoint_dest,
                mlp_state_dict=best_state_dict,
                epoch=best_epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                val_f1_macro=val_f1,
                config=cfg,
                class_to_idx=CLASS_TO_IDX,
                idx_to_class=IDX_TO_CLASS,
                extra_meta={"best_epoch": best_epoch, "total_epochs_trained": epoch},
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.info(
                    "Early stopping acionado! Sem melhoria na perda de validação por %d épocas consecutivas.",
                    cfg.training.early_stopping_patience,
                )
                break

    # Salvar histórico e gráficos
    metrics_dir = Path(cfg.artifacts.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history_file = metrics_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    curves_file = metrics_dir / "training_curves.png"
    plot_training_curves(history, curves_file)

    logger.info("=" * 60)
    logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info("Melhor Época: %d | Val Loss: %.4f | Val Acc: %.4f | Val F1: %.4f", best_epoch, best_val_metrics.get("val_loss", 0), best_val_metrics.get("val_acc", 0), best_val_metrics.get("val_f1", 0))
    logger.info("Checkpoint salvo em: %s", checkpoint_dest)
    logger.info("=" * 60)

    return {
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "history": history,
        "checkpoint_path": str(checkpoint_dest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento da MLP com CNN congelada.")
    parser.add_argument("--config", default=None, help="Caminho do arquivo YAML de configuração.")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescrever número de épocas.")
    parser.add_argument("--batch-size", type=int, default=None, help="Sobrescrever batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Sobrescrever learning rate.")
    parser.add_argument("--device", default=None, help="Dispositivo ('cpu', 'cuda', 'auto').")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
