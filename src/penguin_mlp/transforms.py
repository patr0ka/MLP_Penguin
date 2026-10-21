"""Módulo de transformações e pré-processamento de imagens.

Centraliza todas as rotinas de preparação visual para garantir consistência estrita
entre treinamento, validação, teste e inferência.

Destaques da implementação:
1. Correção de orientação baseada em EXIF (ImageOps.exif_transpose).
2. Conversão uniforme para o espaço de cores RGB (suportando imagens P, RGBA, L, CMYK).
3. Redimensionamento com preservação de proporção e preenchimento (Letterbox Padding)
   para 224x224, evitando que bicos, cabeças ou nadadeiras dos pinguins sejam cortados
   por um corte central agressivo.
4. Normalização padrão ImageNet:
   Média:  [0.485, 0.456, 0.406]
   Desvio: [0.229, 0.224, 0.225]
5. Aumentação moderada e biologicamente plausível aplicada exclusivamente ao treino.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


class LetterboxResize:
    """Redimensiona preservando a proporção original e preenche com bordas neutras.

    Evita distorções anatômicas e cortes acidentais de animais que não estejam perfeitamente
    centralizados no enquadramento original.
    """

    def __init__(self, target_size: int = 224, fill_color: Tuple[int, int, int] = (124, 116, 104)) -> None:
        self.target_size = target_size
        # Preenchimento em cor neutra próxima da média ImageNet desnormalizada
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError(f"Dimensões de imagem inválidas: {w}x{h}")

        scale = self.target_size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        # Criar imagem final preenchida
        padded = Image.new("RGB", (self.target_size, self.target_size), self.fill_color)
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        padded.paste(resized, (paste_x, paste_y))
        return padded


def open_and_sanitize_image(image_input: str | Path | Image.Image) -> Image.Image:
    """Abre uma imagem, corrige a orientação EXIF e converte para RGB."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Tipo de entrada não suportado para imagem: {type(image_input)}")

    # Tratar rotação EXIF se presente
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Converter estritamente para RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def get_base_pipeline(image_size: int = 224, preserve_aspect_ratio: bool = True) -> T.Compose:
    """Retorna o pipeline determinístico para validação, teste e inferência."""
    steps = []
    if preserve_aspect_ratio:
        steps.append(LetterboxResize(target_size=image_size))
    else:
        steps.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR))

    steps.extend([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return T.Compose(steps)


def get_train_pipeline(image_size: int = 224, preserve_aspect_ratio: bool = True) -> T.Compose:
    """Retorna o pipeline com aumentação de dados moderada para treino."""
    steps = []
    if preserve_aspect_ratio:
        steps.append(LetterboxResize(target_size=image_size))
    else:
        steps.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR))

    # Aumentações moderadas e realistas (não alteram traços taxonômicos fundamentais)
    steps.extend([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        T.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return T.Compose(steps)


def preprocess_image(
    image_input: str | Path | Image.Image,
    image_size: int = 224,
    preserve_aspect_ratio: bool = True,
) -> torch.Tensor:
    """Função utilitária de conveniência para pré-processar uma única imagem para inferência.

    Retorna um tensor com shape [1, 3, image_size, image_size].
    """
    img = open_and_sanitize_image(image_input)
    pipeline = get_base_pipeline(image_size=image_size, preserve_aspect_ratio=preserve_aspect_ratio)
    tensor = pipeline(img)
    return tensor.unsqueeze(0)  # [1, 3, H, W]
