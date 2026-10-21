"""Módulo de avaliação independente do modelo classificador de pinguins.

Carrega um checkpoint treinado e avalia o desempenho no conjunto de teste intocado
(ou qualquer outro split especificado).

Calcula métricas científicas:
- Acurácia global
- Precisão, Revocação (Recall) e F1-score por espécie
- Macro F1-score
- Matriz de Confusão visualizada e salva em PNG
- Relatório detalhado em JSON e TXT

Suporta também avaliação do subconjunto de teste externo (espécies fora do domínio
e não-pinguins) para aferir a robustez dos limiares de confiança.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from penguin_mlp.backbone import build_backbone
from penguin_mlp.config import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    load_config,
)
from penguin_mlp.data import PenguinImageDataset
from penguin_mlp.model import build_mlp
from penguin_mlp.transforms import get_base_pipeline
from penguin_mlp.utils import load_checkpoint, resolve_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Matriz de Confusão - Classificação de Pinguins",
) -> None:
    """Plota e salva uma matriz de confusão elegante com Matplotlib."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(cax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)

    # Anotar valores numéricos em cada célula
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                format(val, "d"),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_ylabel("Classe Verdadeira (Ground Truth)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Classe Prevista pelo Modelo", fontsize=11, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Matriz de confusão salva em: %s", output_path)


@torch.no_grad()
def run_evaluation(
    checkpoint_path: str | Path = "artifacts/checkpoints/best_model.pt",
    split_csv: Optional[str | Path] = None,
    output_dir: str | Path = "artifacts/metrics",
    config_path: Optional[str | Path] = None,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Executa a avaliação completa do checkpoint no split de teste."""
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    device = resolve_device(device_str)

    target_csv = Path(split_csv) if split_csv else Path(cfg.dataset.split_dir) / "test.csv"
    if not target_csv.exists():
        raise FileNotFoundError(f"Arquivo de split para avaliação não encontrado: {target_csv}")

    # 1. Carregar checkpoint
    mlp_state, metadata = load_checkpoint(checkpoint_path, device=device)
    class_to_idx = metadata.get("class_to_idx", CLASS_TO_IDX)
    idx_to_class = {int(k): v for k, v in metadata.get("idx_to_class", IDX_TO_CLASS).items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # 2. Reconstruir CNN congelada e MLP
    cnn = build_backbone(pretrained=cfg.cnn.pretrained, freeze=True, device=device)
    mlp_config = metadata.get("mlp", {})
    mlp = build_mlp(
        input_dim=mlp_config.get("input_dim", cfg.mlp.input_dim),
        hidden_dims=mlp_config.get("hidden_dims", cfg.mlp.hidden_dims),
        num_classes=mlp_config.get("num_classes", cfg.mlp.num_classes),
        dropout_rates=mlp_config.get("dropout", cfg.mlp.dropout),
        device=device,
    )
    mlp.load_state_dict(mlp_state)
    mlp.eval()
    cnn.eval()

    # 3. Preparar Dataset e DataLoader de Teste
    prep_meta = metadata.get("preprocessing", {})
    image_size = prep_meta.get("image_size", cfg.dataset.image_size)
    preserve_aspect_ratio = prep_meta.get("preserve_aspect_ratio", cfg.dataset.preserve_aspect_ratio)

    eval_pipeline = get_base_pipeline(image_size=image_size, preserve_aspect_ratio=preserve_aspect_ratio)
    test_dataset = PenguinImageDataset(target_csv, transform=eval_pipeline)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    logger.info("Avaliando modelo em %d amostras de %s...", len(test_dataset), target_csv.name)

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []
    all_paths: List[str] = []

    for images, targets, paths in test_loader:
        images = images.to(device)
        features = cnn(images)
        logits = mlp(features)
        probs = torch.softmax(logits, dim=-1)

        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_paths.extend(paths)

    # 4. Cálculo das Métricas
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    precisions = precision_score(all_targets, all_preds, average=None, zero_division=0).tolist()
    recalls = recall_score(all_targets, all_preds, average=None, zero_division=0).tolist()
    f1s = f1_score(all_targets, all_preds, average=None, zero_division=0).tolist()

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
    cr_text = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "f1_score": float(f1s[idx]),
            "support": int((np.array(all_targets) == idx).sum()),
        }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_dict: Dict[str, Any] = {
        "dataset_split": target_csv.name,
        "sample_count": len(test_dataset),
        "overall_accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "checkpoint_evaluated": str(checkpoint_path),
        "trained_best_epoch": metadata.get("epoch"),
        "validation_loss_at_best_epoch": metadata.get("val_loss"),
    }

    # Salvar metrics.json
    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    # Salvar classification_report.txt
    report_file = out_path / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("RELATÓRIO CIENTÍFICO DE AVALIAÇÃO - MODELO CNN + MLP\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Arquivo avaliado: {target_csv}\n")
        f.write(f"Total de amostras de teste: {len(test_dataset)}\n")
        f.write(f"Acurácia Global: {acc * 100:.2f}%\n")
        f.write(f"F1-Score Macro:  {macro_f1:.4f}\n\n")
        f.write("-" * 65 + "\n")
        f.write("RELATÓRIO DETALHADO POR CLASSE (SCIKIT-LEARN):\n")
        f.write("-" * 65 + "\n")
        f.write(cr_text + "\n\n")
        f.write("-" * 65 + "\n")
        f.write("MATRIZ DE CONFUSÃO NUMÉRICA:\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'':>12}" + "".join([f"{name:>12}" for name in class_names]) + "\n")
        for i, row in enumerate(cm):
            row_str = "".join([f"{val:>12d}" for val in row])
            f.write(f"{class_names[i]:>12}{row_str}\n")

    # Salvar gráfico da matriz de confusão
    cm_img_file = out_path / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_img_file)

    logger.info("=" * 60)
    logger.info("AVALIAÇÃO CONCLUÍDA:")
    logger.info("Acurácia Global: %.2f%% | Macro F1: %.4f", acc * 100, macro_f1)
    logger.info("Resultados gravados em: %s", out_path)
    logger.info("=" * 60)

    return metrics_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação independente do classificador.")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best_model.pt", help="Caminho do checkpoint.")
    parser.add_argument("--split", default=None, help="Caminho do CSV do conjunto a avaliar (padrão: test.csv).")
    parser.add_argument("--output-dir", default="artifacts/metrics", help="Diretório para salvar as métricas.")
    parser.add_argument("--config", default=None, help="Arquivo YAML de configuração.")
    parser.add_argument("--device", default="auto", help="Dispositivo ('cpu', 'cuda', 'auto').")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        split_csv=args.split,
        output_dir=args.output_dir,
        config_path=args.config,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
