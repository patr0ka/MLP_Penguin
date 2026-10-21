"""Módulo de carregamento e estruturação dos conjuntos de dados (PyTorch Dataset & DataLoader).

Fornece a classe PenguinImageDataset e funções de conveniência para instanciar
os DataLoaders de treino, validação e teste com base nos arquivos CSV de divisão estratificada.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from penguin_mlp.config import CLASS_TO_IDX, IDX_TO_CLASS
from penguin_mlp.transforms import (
    get_base_pipeline,
    get_train_pipeline,
    open_and_sanitize_image,
)

logger = logging.getLogger(__name__)


class PenguinImageDataset(Dataset):
    """Dataset PyTorch para carregar fotografias de pinguins e seus rótulos taxonômicos."""

    def __init__(
        self,
        dataframe_or_csv: str | Path | pd.DataFrame,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        base_dir: Optional[str | Path] = None,
    ) -> None:
        """Inicializa o Dataset.

        Args:
            dataframe_or_csv: Caminho para o CSV de split ou DataFrame existente.
            transform: Pipeline de transformações do TorchVision.
            base_dir: Diretório base opcional para resolução de caminhos relativos.
        """
        if isinstance(dataframe_or_csv, (str, Path)):
            self.df = pd.read_csv(dataframe_or_csv)
        elif isinstance(dataframe_or_csv, pd.DataFrame):
            self.df = dataframe_or_csv.copy()
        else:
            raise TypeError("dataframe_or_csv deve ser caminho (str/Path) ou pandas.DataFrame.")

        self.transform = transform
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

        # Validação das colunas obrigatórias
        required_cols = {"image_path", "label", "species"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"O CSV/DataFrame deve conter as colunas: {required_cols}. Encontradas: {list(self.df.columns)}")

        # Assegurar que os tipos estejam corretos
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        raw_path = Path(row["image_path"])
        full_path = raw_path if raw_path.is_absolute() else (self.base_dir / raw_path)

        if not full_path.exists():
            raise FileNotFoundError(f"Fotografia não encontrada: {full_path}")

        try:
            image = open_and_sanitize_image(full_path)
        except Exception as exc:
            logger.error("Falha ao abrir imagem no índice %d (%s): %s", idx, full_path, exc)
            raise RuntimeError(f"Erro ao ler imagem: {full_path}") from exc

        if self.transform is not None:
            tensor = self.transform(image)
        else:
            # Fallback seguro
            tensor = get_base_pipeline()(image)

        label = int(row["label"])
        return tensor, label, str(full_path)


def create_dataloaders(
    split_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    preserve_aspect_ratio: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Cria e retorna os DataLoaders para treino, validação e teste."""
    split_path = Path(split_dir)
    train_csv = split_path / "train.csv"
    val_csv = split_path / "val.csv"
    test_csv = split_path / "test.csv"

    for path, name in [(train_csv, "Treino"), (val_csv, "Validação"), (test_csv, "Teste")]:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo de divisão de {name} não encontrado: {path}. "
                f"Execute a preparação dos dados (prepare_data.py) primeiro."
            )

    train_tf = get_train_pipeline(image_size=image_size, preserve_aspect_ratio=preserve_aspect_ratio)
    eval_tf = get_base_pipeline(image_size=image_size, preserve_aspect_ratio=preserve_aspect_ratio)

    train_ds = PenguinImageDataset(train_csv, transform=train_tf)
    val_ds = PenguinImageDataset(val_csv, transform=eval_tf)
    test_ds = PenguinImageDataset(test_csv, transform=eval_tf)

    logger.info(
        "Conjuntos carregados: Treino=%d amostras | Validação=%d amostras | Teste=%d amostras",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
