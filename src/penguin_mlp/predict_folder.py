"""Módulo de inferência em lote para processamento de pastas com múltiplas fotografias.

Varre um diretório de entrada (como data/to_predict/), processa todas as imagens suportadas
usando uma única instância carregada em memória do extrator CNN + MLP, registra previsões
e probabilidades em formato CSV estruturado e emite alertas para imagens com baixa confiança
ou arquivos ilegíveis.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from penguin_mlp.predict import PenguinPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def predict_folder(
    folder_path: str | Path = "data/to_predict",
    checkpoint_path: str | Path = "artifacts/checkpoints/best_model.pt",
    output_csv: Optional[str | Path] = None,
    confidence_threshold: float = 0.60,
    device: str = "auto",
    config_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Executa a inferência em todas as imagens de um diretório e gera relatório tabular."""
    input_dir = Path(folder_path)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Diretório informado não existe ou não é uma pasta: {input_dir}")

    dest_csv = (
        Path(output_csv)
        if output_csv
        else input_dir / "batch_predictions.csv"
    )

    image_files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        logger.warning("Nenhuma imagem com extensão suportada encontrada em: %s", input_dir)
        return pd.DataFrame()

    logger.info("Carregando modelo para predição em lote (%d imagens encontradas)...", len(image_files))
    predictor = PenguinPredictor(
        checkpoint_path=checkpoint_path,
        confidence_threshold=confidence_threshold,
        device=device,
        config_path=config_path,
    )

    results: List[Dict[str, Any]] = []
    skipped_files: List[Dict[str, str]] = []

    for img_path in tqdm(sorted(image_files), desc="Classificando imagens"):
        try:
            pred = predictor.predict(img_path)
            record = {
                "filename": img_path.name,
                "file_path": str(img_path.as_posix()),
                "predicted_species": pred["predicted_species"],
                "confidence": round(pred["confidence"], 4),
                "is_low_confidence": pred["is_low_confidence"],
                "warning": pred["warning"] or "",
            }
            # Adicionar probabilidades individuais de cada espécie
            for sp, prob in pred["probabilities"].items():
                record[f"prob_{sp.lower()}"] = round(prob, 4)

            results.append(record)
        except Exception as exc:
            logger.warning("Falha ao processar arquivo %s: %s", img_path.name, exc)
            skipped_files.append({"filename": img_path.name, "error": str(exc)})

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results.to_csv(dest_csv, index=False, encoding="utf-8")
        logger.info("Resultados da predição em lote salvos com sucesso em: %s", dest_csv)

        # Exibir resumo no console
        print("\n" + "=" * 65)
        print("RESUMO DA CLASSIFICAÇÃO EM LOTE")
        print("=" * 65)
        print(f"Total de arquivos processados com sucesso: {len(df_results)}")
        print(f"Total de arquivos ignorados por erro:    {len(skipped_files)}")
        print("\nDistribuição das espécies previstas:")
        print(df_results["predicted_species"].value_counts().to_string())

        low_conf_count = df_results["is_low_confidence"].sum()
        if low_conf_count > 0:
            print(f"\nAlerta: {low_conf_count} imagem(ns) apresentaram confiança abaixo de {confidence_threshold * 100:.1f}%.")
        print("=" * 65 + "\n")

    return df_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência em lote em uma pasta de fotografias.")
    parser.add_argument("--folder", default="data/to_predict", help="Diretório contendo imagens para predição.")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best_model.pt", help="Caminho do checkpoint.")
    parser.add_argument("--output-csv", default=None, help="Caminho para salvar a tabela CSV com previsões.")
    parser.add_argument("--threshold", type=float, default=0.60, help="Limiar de confiança mínima.")
    parser.add_argument("--device", default="auto", help="Dispositivo ('cpu', 'cuda', 'auto').")
    parser.add_argument("--config", default=None, help="Caminho do arquivo YAML de configuração.")
    args = parser.parse_args()

    predict_folder(
        folder_path=args.folder,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        confidence_threshold=args.threshold,
        device=args.device,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
