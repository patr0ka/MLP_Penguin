"""Testes automatizados para a arquitetura da MLP e o extrator CNN ResNet18."""

import pytest
import torch
import torch.nn as nn

from penguin_mlp.backbone import ResNet18FeatureExtractor, build_backbone
from penguin_mlp.model import PenguinClassifierMLP, build_mlp
from penguin_mlp.utils import count_parameters


def test_mlp_input_output_shapes():
    """Verifica que a MLP aceita entrada [B, 512] e retorna logits [B, 3]."""
    batch_size = 8
    mlp = build_mlp(input_dim=512, hidden_dims=[128, 64], num_classes=3, dropout_rates=[0.30, 0.20])
    dummy_input = torch.randn(batch_size, 512)

    logits = mlp(dummy_input)
    assert logits.shape == (batch_size, 3)
    assert torch.isfinite(logits).all()


def test_mlp_parameter_count():
    """Confirma o cálculo analítico exato de 74.115 parâmetros treináveis na MLP inicial.

    Camada 1: 512 * 128 + 128 = 65.664
    Camada 2: 128 * 64 + 64   =  8.256
    Camada 3: 64 * 3 + 3      =    195
    Total:                      74.115
    """
    mlp = build_mlp(input_dim=512, hidden_dims=[128, 64], num_classes=3, dropout_rates=[0.30, 0.20])
    trainable_params, total_params = count_parameters(mlp)

    expected_params = (512 * 128 + 128) + (128 * 64 + 64) + (64 * 3 + 3)
    assert expected_params == 74115
    assert trainable_params == 74115
    assert total_params == 74115


def test_dropout_behavior_train_vs_eval():
    """Verifica que o Dropout é estocástico no modo train() e determinístico no modo eval()."""
    torch.manual_seed(42)
    mlp = build_mlp(input_dim=512, hidden_dims=[128, 64], num_classes=3, dropout_rates=[0.5, 0.5])
    dummy_input = torch.randn(10, 512)

    # Modo Avaliação: duas passagens devem produzir exatamente o mesmo tensor
    mlp.eval()
    with torch.no_grad():
        out_eval_1 = mlp(dummy_input)
        out_eval_2 = mlp(dummy_input)
    assert torch.allclose(out_eval_1, out_eval_2, atol=1e-6)

    # Modo Treinamento: passagens com dropout ativo devem apresentar diferenças estocásticas
    mlp.train()
    out_train_1 = mlp(dummy_input)
    out_train_2 = mlp(dummy_input)
    assert not torch.allclose(out_train_1, out_train_2, atol=1e-4)


def test_cnn_backbone_is_frozen():
    """Garante que todos os parâmetros da ResNet18 estejam congelados (requires_grad = False)."""
    # Testar com pretrained=False para velocidade no teste unitário
    extractor = build_backbone(pretrained=False, freeze=True)

    for name, param in extractor.named_parameters():
        assert not param.requires_grad, f"Parâmetro não congelado encontrado: {name}"

    assert extractor.feature_dim == 512

    # Teste de extração
    dummy_imgs = torch.randn(4, 3, 224, 224)
    features = extractor.extract_features(dummy_imgs)
    assert features.shape == (4, 512)


def test_backprop_gradients_update_only_mlp():
    """Verifica que o backward() gera gradientes para a MLP mas NÃO para a CNN."""
    extractor = build_backbone(pretrained=False, freeze=True)
    mlp = build_mlp(input_dim=512, hidden_dims=[128, 64], num_classes=3, dropout_rates=[0.3, 0.2])

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_targets = torch.tensor([0, 1, 2, 0])

    # Forward
    with torch.no_grad():
        features = extractor(dummy_images)
    logits = mlp(features)
    loss = criterion(logits, dummy_targets)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # 1. Parâmetros da MLP DEVEM possuir gradientes
    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"Gradiente nulo na MLP: {name}"
        assert torch.isfinite(param.grad).all()

    # 2. Parâmetros da CNN NÃO DEVEM possuir gradientes
    for name, param in extractor.named_parameters():
        assert param.grad is None, f"Gradiente indevido encontrado na CNN congelada: {name}"

    # Step do otimizador atualiza MLP
    optimizer.step()


def test_mlp_softmax_probabilities_sum_to_one():
    """Garante que a função predict_probabilities produza probabilidades válidas somando 1."""
    mlp = build_mlp(input_dim=512, num_classes=3)
    dummy_features = torch.randn(5, 512)
    probs = mlp.predict_probabilities(dummy_features)

    assert probs.shape == (5, 3)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    # Soma de cada linha deve ser 1.0
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
