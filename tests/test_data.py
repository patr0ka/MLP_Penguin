"""Testes automatizados para processamento de imagens, transformações e datasets."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from penguin_mlp.config import CANONICAL_CLASSES, CLASS_TO_IDX
from penguin_mlp.data import PenguinImageDataset
from penguin_mlp.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LetterboxResize,
    get_base_pipeline,
    get_train_pipeline,
    open_and_sanitize_image,
    preprocess_image,
)


@pytest.fixture
def sample_images(tmp_path: Path):
    """Cria fixtures de imagens sintéticas para testes."""
    # 1. Imagem RGB retangular (proporção 2:1)
    rgb_path = tmp_path / "sample_rgb.jpg"
    img_rgb = Image.new("RGB", (400, 200), color=(200, 50, 50))
    img_rgb.save(rgb_path)

    # 2. Imagem em escala de cinza (L)
    gray_path = tmp_path / "sample_gray.png"
    img_gray = Image.new("L", (150, 150), color=128)
    img_gray.save(gray_path)

    # 3. Imagem RGBA com transparência
    rgba_path = tmp_path / "sample_rgba.png"
    img_rgba = Image.new("RGBA", (100, 100), color=(10, 20, 30, 128))
    img_rgba.save(rgba_path)

    # 4. Arquivo corrompido / lixo
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"LIXO_NAO_IMAGEM_123456789")

    return {
        "rgb": rgb_path,
        "gray": gray_path,
        "rgba": rgba_path,
        "corrupt": corrupt_path,
    }


def test_open_and_sanitize_image(sample_images):
    """Verifica se imagens em diferentes modos de cor são convertidas estritamente para RGB."""
    img_rgb = open_and_sanitize_image(sample_images["rgb"])
    assert img_rgb.mode == "RGB"

    img_gray = open_and_sanitize_image(sample_images["gray"])
    assert img_gray.mode == "RGB"

    img_rgba = open_and_sanitize_image(sample_images["rgba"])
    assert img_rgba.mode == "RGB"


def test_corrupted_image_handling(sample_images):
    """Verifica que arquivos corrompidos geram exceções apropriadas."""
    with pytest.raises(Exception):
        open_and_sanitize_image(sample_images["corrupt"])


def test_letterbox_preserves_aspect_ratio():
    """Verifica que o LetterboxResize gera dimensões exatas sem distorção anatômica."""
    resizer = LetterboxResize(target_size=224)
    # Imagem muito larga (panorâmica 400x100)
    wide_img = Image.new("RGB", (400, 100), color=(255, 0, 0))
    padded_wide = resizer(wide_img)
    assert padded_wide.size == (224, 224)

    # Imagem muito alta (vertical 100x400)
    tall_img = Image.new("RGB", (100, 400), color=(0, 255, 0))
    padded_tall = resizer(tall_img)
    assert padded_tall.size == (224, 224)


def test_pipeline_output_tensor(sample_images):
    """Verifica que o pré-processamento produz tensores com formato e tipos corretos."""
    tensor = preprocess_image(sample_images["rgb"], image_size=224, preserve_aspect_ratio=True)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    # Verificar que os valores são finitos (sem NaN ou Inf)
    assert torch.isfinite(tensor).all()


def test_dataset_item_retrieval(tmp_path, sample_images):
    """Testa a indexação e recuperação de amostras no PenguinImageDataset."""
    csv_path = tmp_path / "test_split.csv"
    df = pd.DataFrame([
        {"image_path": str(sample_images["rgb"]), "label": 0, "species": "Adelie"},
        {"image_path": str(sample_images["gray"]), "label": 1, "species": "Chinstrap"},
        {"image_path": str(sample_images["rgba"]), "label": 2, "species": "Gentoo"},
    ])
    df.to_csv(csv_path, index=False)

    pipeline = get_base_pipeline(image_size=224)
    dataset = PenguinImageDataset(csv_path, transform=pipeline)

    assert len(dataset) == 3
    tensor, label, path = dataset[0]
    assert tensor.shape == (3, 224, 224)
    assert label == 0
    assert Path(path).name == "sample_rgb.jpg"


def test_splits_no_leakage():
    """Se os arquivos de split existirem em data/splits, garante ausência total de vazamento."""
    split_dir = Path("data/splits")
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)

        train_files = set(train_df["image_path"])
        val_files = set(val_df["image_path"])
        test_files = set(test_df["image_path"])

        # Interseção nula obrigatória
        assert len(train_files.intersection(val_files)) == 0, "Vazamento detectado entre treino e validação!"
        assert len(train_files.intersection(test_files)) == 0, "Vazamento detectado entre treino e teste!"
        assert len(val_files.intersection(test_files)) == 0, "Vazamento detectado entre validação e teste!"
