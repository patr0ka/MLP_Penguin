"""Módulo de inferência para classificação de uma fotografia individual.

Carrega o checkpoint salvo, reconstrói o extrator CNN (ResNet18 congelada) e a MLP,
aplica o pipeline de pré-processamento estrito e calcula as probabilidades estimadas
para as espécies Adelie, Chinstrap e Gentoo via Softmax.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image
import torch
import torch.nn.functional as F

from penguin_mlp.backbone import build_backbone
from penguin_mlp.config import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    load_config,
)
from penguin_mlp.model import build_mlp
from penguin_mlp.transforms import preprocess_image
from penguin_mlp.utils import load_checkpoint, resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PenguinPredictor:
    """Classe encapsuladora para inferência eficiente em fotos de pinguins.

    Carrega os modelos CNN e MLP uma única vez na memória e processa imagens
    isoladas ou em lote mantendo total consistência de pré-processamento.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "artifacts/checkpoints/best_model.pt",
        confidence_threshold: float = 0.60,
        device: str = "auto",
        config_path: Optional[str | Path] = None,
    ) -> None:
        self.cfg = load_config(config_path)
        self.device = resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self.checkpoint_path = Path(checkpoint_path)

        # 1. Carregar checkpoint e metadados
        self.mlp_state, self.metadata = load_checkpoint(self.checkpoint_path, device=self.device)
        self.idx_to_class: Dict[int, str] = {
            int(k): v for k, v in self.metadata.get("idx_to_class", IDX_TO_CLASS).items()
        }
        self.class_to_idx: Dict[str, int] = self.metadata.get("class_to_idx", CLASS_TO_IDX)
        self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        # 2. Inicializar CNN congelada
        cnn_meta = self.metadata.get("cnn", {})
        self.cnn = build_backbone(
            pretrained=cnn_meta.get("pretrained", True),
            freeze=True,
            device=self.device,
        )
        self.cnn.eval()

        # 3. Inicializar MLP e carregar pesos
        mlp_meta = self.metadata.get("mlp", {})
        self.mlp = build_mlp(
            input_dim=mlp_meta.get("input_dim", self.cfg.mlp.input_dim),
            hidden_dims=mlp_meta.get("hidden_dims", self.cfg.mlp.hidden_dims),
            num_classes=mlp_meta.get("num_classes", len(self.class_names)),
            dropout_rates=mlp_meta.get("dropout", self.cfg.mlp.dropout),
            device=self.device,
        )
        self.mlp.load_state_dict(self.mlp_state)
        self.mlp.eval()

        # 4. Parâmetros de pré-processamento
        prep_meta = self.metadata.get("preprocessing", {})
        self.image_size = prep_meta.get("image_size", self.cfg.dataset.image_size)
        self.preserve_aspect_ratio = prep_meta.get("preserve_aspect_ratio", self.cfg.dataset.preserve_aspect_ratio)

    @torch.no_grad()
    def predict(self, image_input: str | Path | Image.Image) -> Dict[str, Any]:
        """Classifica uma imagem e retorna previsões detalhadas e probabilidades."""
        # 1. Pré-processar imagem para tensor [1, 3, H, W]
        try:
            tensor = preprocess_image(
                image_input,
                image_size=self.image_size,
                preserve_aspect_ratio=self.preserve_aspect_ratio,
            ).to(self.device)
        except Exception as exc:
            raise ValueError(f"Não foi possível processar a imagem: {exc}") from exc

        # 2. Extração de características pela CNN congelada
        features = self.cnn(tensor)  # [1, 512]

        # 3. Propagação pela MLP e cálculo do Softmax
        logits = self.mlp(features)  # [1, 3]
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(torch.argmax(logits, dim=-1).item())
        pred_species = self.idx_to_class[pred_idx]
        confidence = float(probs[pred_idx])

        # Dicionário de probabilidades por classe
        probabilities_dict = {
            self.idx_to_class[i]: float(probs[i]) for i in range(len(self.class_names))
        }

        is_low_confidence = confidence < self.confidence_threshold

        warning_message = None
        if is_low_confidence:
            warning_message = (
                f"Aviso: Previsão com baixa confiança ({confidence * 100:.1f}% < "
                f"limiar de {self.confidence_threshold * 100:.1f}%). "
                f"A imagem pode conter uma espécie fora do domínio, não-pinguim ou enquadramento adverso."
            )

        return {
            "image": str(image_input) if isinstance(image_input, (str, Path)) else "PIL.Image",
            "predicted_species": pred_species,
            "predicted_index": pred_idx,
            "confidence": confidence,
            "probabilities": probabilities_dict,
            "is_low_confidence": is_low_confidence,
            "warning": warning_message,
        }


def format_prediction_output(result: Dict[str, Any]) -> str:
    """Formata a saída de previsão para exibição no terminal conforme o padrão acadêmico."""
    lines = [
        "",
        "=" * 60,
        f"Imagem: {result['image']}",
        "-" * 60,
        f"Espécie prevista: {result['predicted_species']} (Confiança: {result['confidence'] * 100:.2f}%)",
        "",
        "Probabilidades estimadas por classe:",
    ]
    for species, prob in result["probabilities"].items():
        marker = " -> [PREVISTA]" if species == result["predicted_species"] else ""
        lines.append(f"  - {species:<10}: {prob * 100:>6.2f}%{marker}")

    if result.get("warning"):
        lines.append("")
        lines.append("ALERTA:")
        lines.append(f"  {result['warning']}")

    lines.append("")
    lines.append("Nota: As probabilidades representam a saída do classificador fechado,")
    lines.append("não uma garantia absoluta de identificação biológica em ambiente selvagem.")
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classificação de espécie de pinguim a partir de fotografia.")
    parser.add_argument("--image", required=True, help="Caminho do arquivo de imagem (JPG, PNG, WEBP).")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best_model.pt", help="Caminho do checkpoint do modelo.")
    parser.add_argument("--threshold", type=float, default=0.60, help="Limiar de confiança mínima (padrão: 0.60).")
    parser.add_argument("--device", default="auto", help="Dispositivo ('cpu', 'cuda', 'auto').")
    parser.add_argument("--config", default=None, help="Caminho do arquivo YAML de configuração.")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        logger.error("Arquivo de imagem não encontrado: %s", img_path)
        return

    try:
        predictor = PenguinPredictor(
            checkpoint_path=args.checkpoint,
            confidence_threshold=args.threshold,
            device=args.device,
            config_path=args.config,
        )
        res = predictor.predict(img_path)
        print(format_prediction_output(res))
    except Exception as exc:
        logger.error("Erro durante a inferência: %s", exc)


if __name__ == "__main__":
    main()
