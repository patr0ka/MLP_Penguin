"""Módulo de configuração centralizada do projeto penguin_mlp.

Carrega configurações a partir de arquivos YAML, valida consistência dos hiperparâmetros
(soma das divisões do dataset, dimensões entre CNN e MLP, classes canônicas)
e disponibiliza instâncias tipadas para todos os módulos do sistema.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logger = logging.getLogger(__name__)

# Mapeamento canônico universal de classes e identificadores numéricos
CANONICAL_CLASSES: tuple[str, ...] = ("Adelie", "Chinstrap", "Gentoo")
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS: dict[int, str] = {idx: name for idx, name in enumerate(CANONICAL_CLASSES)}


@dataclass
class DatasetConfig:
    root: str = "data/raw"
    split_dir: str = "data/splits"
    to_predict_dir: str = "data/to_predict"
    external_test_dir: str = "data/external_test"
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    class_names: List[str] = field(default_factory=lambda: list(CANONICAL_CLASSES))
    image_size: int = 224
    preserve_aspect_ratio: bool = True

    def validate(self) -> None:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"As proporções do dataset devem somar 1.0. Soma atual: {total:.4f} "
                f"({self.train_ratio} + {self.validation_ratio} + {self.test_ratio})"
            )
        if tuple(self.class_names) != CANONICAL_CLASSES:
            logger.warning(
                "Classes configuradas (%s) diferem das canônicas (%s). Ajustando para padrão.",
                self.class_names,
                CANONICAL_CLASSES,
            )


@dataclass
class CNNConfig:
    architecture: str = "resnet18"
    pretrained: bool = True
    freeze: bool = True
    feature_dim: int = 512


@dataclass
class MLPConfig:
    input_dim: int = 512
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    num_classes: int = 3
    dropout: List[float] = field(default_factory=lambda: [0.30, 0.20])

    def validate(self, cnn_feature_dim: int) -> None:
        if self.input_dim != cnn_feature_dim:
            raise ValueError(
                f"A dimensão de entrada da MLP ({self.input_dim}) deve coincidir com a saída "
                f"da CNN ({cnn_feature_dim})."
            )
        if self.num_classes != len(CANONICAL_CLASSES):
            raise ValueError(
                f"num_classes ({self.num_classes}) deve ser igual a {len(CANONICAL_CLASSES)}."
            )
        if len(self.dropout) != len(self.hidden_dims):
            raise ValueError(
                f"A quantidade de taxas de dropout ({len(self.dropout)}) deve ser igual à "
                f"quantidade de camadas ocultas ({len(self.hidden_dims)})."
            )


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 6
    scheduler: str = "reduce_on_plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    min_lr: float = 0.00001
    device: str = "auto"


@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.60
    device: str = "auto"


@dataclass
class ArtifactsConfig:
    checkpoint_dir: str = "artifacts/checkpoints"
    metrics_dir: str = "artifacts/metrics"
    best_checkpoint_path: str = "artifacts/checkpoints/best_model.pt"


@dataclass
class ProjectConfig:
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)

    def validate(self) -> None:
        self.dataset.validate()
        self.mlp.validate(self.cnn.feature_dim)


def _apply_mlp_params_overrides(cfg: ProjectConfig) -> None:
    """Aplica as variáveis de mlp_params.py sobre a configuração, se o arquivo existir."""
    candidate_paths = [
        Path("mlp_params.py"),
        Path(__file__).resolve().parents[2] / "mlp_params.py",
    ]
    params_file = next((p for p in candidate_paths if p.exists()), None)
    if not params_file:
        return

    try:
        spec = importlib.util.spec_from_file_location("mlp_params", params_file)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            if hasattr(mod, "SEED"):
                cfg.seed = int(getattr(mod, "SEED"))
            if hasattr(mod, "INPUT_DIM"):
                cfg.mlp.input_dim = int(getattr(mod, "INPUT_DIM"))
            if hasattr(mod, "HIDDEN_DIMS"):
                cfg.mlp.hidden_dims = list(getattr(mod, "HIDDEN_DIMS"))
            if hasattr(mod, "NUM_CLASSES"):
                cfg.mlp.num_classes = int(getattr(mod, "NUM_CLASSES"))
            if hasattr(mod, "DROPOUT_RATES"):
                cfg.mlp.dropout = list(getattr(mod, "DROPOUT_RATES"))
            if hasattr(mod, "EPOCHS"):
                cfg.training.epochs = int(getattr(mod, "EPOCHS"))
            if hasattr(mod, "BATCH_SIZE"):
                cfg.training.batch_size = int(getattr(mod, "BATCH_SIZE"))
            if hasattr(mod, "LEARNING_RATE"):
                cfg.training.learning_rate = float(getattr(mod, "LEARNING_RATE"))
            if hasattr(mod, "WEIGHT_DECAY"):
                cfg.training.weight_decay = float(getattr(mod, "WEIGHT_DECAY"))
            if hasattr(mod, "EARLY_STOPPING_PATIENCE"):
                cfg.training.early_stopping_patience = int(getattr(mod, "EARLY_STOPPING_PATIENCE"))
            if hasattr(mod, "CONFIDENCE_THRESHOLD"):
                cfg.inference.confidence_threshold = float(getattr(mod, "CONFIDENCE_THRESHOLD"))
            if hasattr(mod, "DEVICE"):
                cfg.training.device = str(getattr(mod, "DEVICE"))
                cfg.inference.device = str(getattr(mod, "DEVICE"))
            logger.info("Hiperparâmetros da MLP carregados com sucesso de: %s", params_file)
    except Exception as exc:
        logger.warning("Falha ao carregar hiperparâmetros de %s: %s", params_file, exc)


def load_config(config_path: Optional[str | Path] = None) -> ProjectConfig:
    """Carrega as configurações do arquivo YAML e sobrescreve com mlp_params.py."""
    default_path = Path("configs/default.yaml")
    path_to_use = Path(config_path) if config_path else default_path

    if not path_to_use.exists():
        logger.warning("Arquivo %s não encontrado. Usando configurações padrão.", path_to_use)
        cfg = ProjectConfig()
        _apply_mlp_params_overrides(cfg)
        cfg.validate()
        return cfg

    with open(path_to_use, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    dataset_dict = raw.get("dataset", {})
    cnn_dict = raw.get("cnn", {})
    mlp_dict = raw.get("mlp", {})
    training_dict = raw.get("training", {})
    inference_dict = raw.get("inference", {})
    artifacts_dict = raw.get("artifacts", {})

    cfg = ProjectConfig(
        seed=raw.get("seed", 42),
        dataset=DatasetConfig(**dataset_dict),
        cnn=CNNConfig(**cnn_dict),
        mlp=MLPConfig(**mlp_dict),
        training=TrainingConfig(**training_dict),
        inference=InferenceConfig(**inference_dict),
        artifacts=ArtifactsConfig(**artifacts_dict),
    )
    _apply_mlp_params_overrides(cfg)
    cfg.validate()
    return cfg
