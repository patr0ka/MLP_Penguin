"""Módulo de validação, deduplicação e divisão estratificada do dataset.

Executa a preparação completa dos dados brutos:
1. Varredura recursiva de diretórios por espécies conhecidas (Adelie, Chinstrap, Gentoo).
2. Validação de integridade de arquivo e teste de abertura com Pillow.
3. Detecção de duplicatas exatas via hash criptográfico SHA-256.
4. Detecção de imagens quase duplicadas via hash perceptual (pHash).
5. Geração de relatório de saneamento com total de imagens válidas, corrompidas e duplicadas.
6. Divisão estratificada (70% treino, 15% validação, 15% teste) com semente determinística (42).
7. Persistência dos splits em arquivos CSV auditáveis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from penguin_mlp.config import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    load_config,
)
from penguin_mlp.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def compute_sha256(file_path: Path) -> str:
    """Calcula o hash SHA-256 dos bytes brutos de um arquivo."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_perceptual_hash(image: Image.Image) -> str:
    """Calcula o hash perceptual (pHash) para identificação de imagens quase idênticas."""
    return str(imagehash.phash(image))


def scan_and_validate_dataset(
    raw_dir: Path,
    phash_threshold: int = 4,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Analisa arquivos de imagem, descarta corrompidos e remove duplicatas exatas e perceptuais.

    Args:
        raw_dir: Diretório raiz dos dados brutos (deve conter subpastas por espécie).
        phash_threshold: Distância de Hamming máxima para considerar imagens quase duplicadas.

    Returns:
        (valid_records, audit_report)
    """
    valid_records: List[Dict[str, Any]] = []
    seen_sha256: Set[str] = set()
    seen_phashes: Dict[str, List[Tuple[imagehash.ImageHash, Path]]] = {
        cls: [] for cls in CANONICAL_CLASSES
    }

    report: Dict[str, Any] = {
        "species_stats": {},
        "corrupted_files": [],
        "exact_duplicates": [],
        "near_duplicates": [],
        "total_files_scanned": 0,
        "total_valid_retained": 0,
    }

    for species in CANONICAL_CLASSES:
        species_folder = raw_dir / species
        stats = {
            "scanned": 0,
            "corrupted": 0,
            "exact_duplicate": 0,
            "near_duplicate": 0,
            "retained": 0,
        }

        if not species_folder.exists() or not species_folder.is_dir():
            logger.warning("Pasta da espécie não encontrada: %s", species_folder)
            report["species_stats"][species] = stats
            continue

        candidate_files = [
            p for p in species_folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        stats["scanned"] = len(candidate_files)
        report["total_files_scanned"] += len(candidate_files)

        for file_path in sorted(candidate_files):
            # 1. Tentativa de abertura e validação com Pillow
            try:
                with Image.open(file_path) as img:
                    img.verify()
                # Reabrir para ler dimensões e dados de pixels
                with Image.open(file_path) as img:
                    img = ImageOps.exif_transpose(img)
                    width, height = img.size
                    curr_phash = imagehash.phash(img)
            except Exception as exc:
                logger.warning("Arquivo corrompido ou ilegível ignorado: %s (%s)", file_path, exc)
                stats["corrupted"] += 1
                report["corrupted_files"].append({"path": str(file_path), "reason": str(exc)})
                continue

            # 2. Verificação de duplicata exata por SHA-256
            sha256 = compute_sha256(file_path)
            if sha256 in seen_sha256:
                stats["exact_duplicate"] += 1
                report["exact_duplicates"].append({"path": str(file_path), "sha256": sha256})
                logger.info("Duplicata exata descartada: %s", file_path.name)
                continue

            # 3. Verificação de quase duplicata perceptual (pHash)
            is_near_dup = False
            for prev_hash, prev_path in seen_phashes[species]:
                hamming_dist = curr_phash - prev_hash
                if hamming_dist <= phash_threshold:
                    is_near_dup = True
                    stats["near_duplicate"] += 1
                    report["near_duplicates"].append({
                        "path": str(file_path),
                        "original_path": str(prev_path),
                        "distance": int(hamming_dist),
                    })
                    logger.info(
                        "Quase duplicata descartada (dist=%d): %s (original: %s)",
                        hamming_dist,
                        file_path.name,
                        prev_path.name,
                    )
                    break

            if is_near_dup:
                continue

            # Arquivo aprovado
            seen_sha256.add(sha256)
            seen_phashes[species].append((curr_phash, file_path))
            stats["retained"] += 1

            valid_records.append({
                "image_path": str(file_path.as_posix()),
                "filename": file_path.name,
                "species": species,
                "label": CLASS_TO_IDX[species],
                "sha256": sha256,
                "width": width,
                "height": height,
                "phash": str(curr_phash),
            })

        report["species_stats"][species] = stats

    report["total_valid_retained"] = len(valid_records)
    return valid_records, report


def create_stratified_splits(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Realiza a divisão estratificada dos registros em treino, validação e teste."""
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Nenhum registro válido disponível para gerar os splits.")

    labels = df["label"].values

    # Primeiro split: separa treino do restante (val + test)
    remaining_ratio = val_ratio + test_ratio
    train_df, rem_df = train_test_split(
        df,
        test_size=remaining_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Segundo split: separa validação e teste igualmente
    test_rel_ratio = test_ratio / remaining_ratio
    val_df, test_df = train_test_split(
        rem_df,
        test_size=test_rel_ratio,
        stratify=rem_df["label"].values,
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return combined


def prepare_dataset(
    raw_dir: str | Path = "data/raw",
    split_dir: str | Path = "data/splits",
    force: bool = False,
    config_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Função principal que orquestra a validação, deduplicação e salvamento dos splits."""
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    raw_path = Path(raw_dir)
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)

    manifest_csv = split_path / "manifest.csv"
    train_csv = split_path / "train.csv"
    val_csv = split_path / "val.csv"
    test_csv = split_path / "test.csv"
    report_json = split_path / "preparation_report.json"
    report_txt = split_path / "preparation_report.txt"

    # Se já existir divisão e force for False, carregar e retornar
    if not force and train_csv.exists() and val_csv.exists() and test_csv.exists():
        logger.info("Splits já existentes encontrados em %s. Usando divisão existente.", split_path)
        return pd.read_csv(manifest_csv)

    logger.info("Iniciando varredura e validação das imagens em: %s", raw_path)
    records, report = scan_and_validate_dataset(raw_path)

    if not records:
        raise RuntimeError(
            f"Nenhuma imagem válida encontrada em {raw_path}. "
            f"Verifique se o download do dataset foi executado (download_dataset.py)."
        )

    logger.info(
        "Validação concluída: %d arquivos lidos | %d retidos após deduplicação.",
        report["total_files_scanned"],
        report["total_valid_retained"],
    )

    # Divisão estratificada
    combined_df = create_stratified_splits(
        records,
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.validation_ratio,
        test_ratio=cfg.dataset.test_ratio,
        seed=cfg.seed,
    )

    # Salvar CSVs individuais e manifesto
    train_df = combined_df[combined_df["split"] == "train"]
    val_df = combined_df[combined_df["split"] == "val"]
    test_df = combined_df[combined_df["split"] == "test"]

    combined_df.to_csv(manifest_csv, index=False, encoding="utf-8")
    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv, index=False, encoding="utf-8")
    test_df.to_csv(test_csv, index=False, encoding="utf-8")

    # Atualizar relatório com distribuição final por split
    split_distribution = (
        combined_df.groupby(["split", "species"]).size().unstack(fill_value=0).to_dict()
    )
    report["split_distribution"] = split_distribution
    report["split_counts"] = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "total": len(combined_df),
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Relatório legível em texto
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RELATÓRIO DE PREPARAÇÃO E SANEAMENTO DO DATASET\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total de arquivos escaneados: {report['total_files_scanned']}\n")
        f.write(f"Total de arquivos corrompidos: {len(report['corrupted_files'])}\n")
        f.write(f"Total de duplicatas exatas: {len(report['exact_duplicates'])}\n")
        f.write(f"Total de quase-duplicatas:  {len(report['near_duplicates'])}\n")
        f.write(f"Total de imagens retidas:   {report['total_valid_retained']}\n\n")
        f.write("-" * 60 + "\n")
        f.write("ESTATÍSTICAS POR ESPÉCIE:\n")
        f.write("-" * 60 + "\n")
        for sp, stats in report["species_stats"].items():
            f.write(
                f"{sp:<12}: Escaneados={stats['scanned']:<4} | "
                f"Corrompidos={stats['corrupted']:<3} | "
                f"Duplicatas={stats['exact_duplicate'] + stats['near_duplicate']:<3} | "
                f"Retidos={stats['retained']:<4}\n"
            )
        f.write("\n" + "-" * 60 + "\n")
        f.write("DISTRIBUIÇÃO DOS SPLITS ESTRATIFICADOS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Treino     (70%): {len(train_df)} imagens\n")
        f.write(f"Validação  (15%): {len(val_df)} imagens\n")
        f.write(f"Teste      (15%): {len(test_df)} imagens\n")
        f.write(f"Total            : {len(combined_df)} imagens\n")

    logger.info("Splits salvos com sucesso em: %s", split_path)
    return combined_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparação e divisão estratificada do dataset.")
    parser.add_argument("--raw-dir", default="data/raw", help="Caminho para os dados brutos.")
    parser.add_argument("--split-dir", default="data/splits", help="Caminho para salvar os CSVs.")
    parser.add_argument("--force", action="store_true", help="Força a recriação dos splits.")
    parser.add_argument("--config", default=None, help="Caminho para arquivo YAML de configuração.")
    args = parser.parse_args()

    prepare_dataset(
        raw_dir=args.raw_dir,
        split_dir=args.split_dir,
        force=args.force,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
