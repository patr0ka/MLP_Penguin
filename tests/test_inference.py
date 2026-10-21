"""Testes automatizados para a etapa de inferência e predição."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from penguin_mlp.config import CLASS_TO_IDX, IDX_TO_CLASS, load_config
from penguin_mlp.model import build_mlp
from penguin_mlp.predict import PenguinPredictor
from penguin_mlp.utils import save_checkpoint


@pytest.fixture
def mock_checkpoint_and_image(tmp_path: Path):
    """Cria um checkpoint válido simulado e uma imagem de teste."""
    # 1. Imagem de teste
    img_path = tmp_path / "test_penguin.jpg"
    img = Image.new("RGB", (300, 300), color=(120, 150, 180))
    img.save(img_path)

    # 2. Checkpoint mock
    cfg = load_config()
    mlp = build_mlp(input_dim=512, hidden_dims=[128, 64], num_classes=3, dropout_rates=[0.3, 0.2])
    ckpt_path = tmp_path / "mock_best_model.pt"

    save_checkpoint(
        checkpoint_path=ckpt_path,
        mlp_state_dict=mlp.state_dict(),
        epoch=1,
        val_loss=0.45,
        val_acc=0.85,
        val_f1_macro=0.84,
        config=cfg,
        class_to_idx=CLASS_TO_IDX,
        idx_to_class=IDX_TO_CLASS,
    )

    return {
        "checkpoint": ckpt_path,
        "image": img_path,
    }


def test_predictor_valid_output(mock_checkpoint_and_image):
    """Testa se o PenguinPredictor executa a inferência completa com probabilidades válidas."""
    predictor = PenguinPredictor(
        checkpoint_path=mock_checkpoint_and_image["checkpoint"],
        confidence_threshold=0.60,
        device="cpu",
    )

    result = predictor.predict(mock_checkpoint_and_image["image"])

    assert "predicted_species" in result
    assert result["predicted_species"] in ["Adelie", "Chinstrap", "Gentoo"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

    # Probabilidades somam 1.0
    probs = result["probabilities"]
    assert set(probs.keys()) == {"Adelie", "Chinstrap", "Gentoo"}
    total_prob = sum(probs.values())
    assert abs(total_prob - 1.0) < 1e-4


def test_low_confidence_warning_trigger(mock_checkpoint_and_image):
    """Verifica que a flag is_low_confidence e o aviso são ativados com threshold elevado."""
    # Forçar limiar irrealisticamente alto (0.999) para disparar aviso
    predictor = PenguinPredictor(
        checkpoint_path=mock_checkpoint_and_image["checkpoint"],
        confidence_threshold=0.999,
        device="cpu",
    )

    result = predictor.predict(mock_checkpoint_and_image["image"])
    assert result["is_low_confidence"] is True
    assert result["warning"] is not None
    assert "baixa confiança" in result["warning"]


def test_missing_checkpoint_raises_error(tmp_path: Path):
    """Verifica que a ausência do checkpoint gera FileNotFoundError compreensível."""
    non_existent_ckpt = tmp_path / "inexistente.pt"
    with pytest.raises(FileNotFoundError) as exc_info:
        PenguinPredictor(checkpoint_path=non_existent_ckpt)

    assert "não encontrado" in str(exc_info.value)


def test_missing_image_raises_error(mock_checkpoint_and_image, tmp_path: Path):
    """Verifica que imagem inexistente produz erro apropriado."""
    predictor = PenguinPredictor(
        checkpoint_path=mock_checkpoint_and_image["checkpoint"],
        device="cpu",
    )
    non_existent_img = tmp_path / "nao_existe.jpg"
    with pytest.raises(ValueError):
        predictor.predict(non_existent_img)
