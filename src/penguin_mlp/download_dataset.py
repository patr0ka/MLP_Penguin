"""Módulo para download reproduzível do dataset de pinguins a partir do Wikimedia Commons.

Coleta fotografias reais das três espécies-alvo:
- Pygoscelis adeliae (Adelie)
- Pygoscelis antarcticus (Chinstrap)
- Pygoscelis papua (Gentoo)

Além de imagens para teste externo (espécies fora do domínio e não-pinguins).
Registra metadados completos de proveniência (URL, autor, licença, dimensões).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

USER_AGENT = "PenguinClassifierAcademic/1.0 (academic research project; contact: student@unicentro.br)"

CATEGORIES = {
    "Adelie": "Category:Pygoscelis_adeliae",
    "Chinstrap": "Category:Pygoscelis_antarcticus",
    "Gentoo": "Category:Pygoscelis_papua",
}

# Espécies externas e não-pinguins para teste de domínio fora da distribuição
EXTERNAL_CATEGORIES = {
    "Emperor_Penguin": "Category:Aptenodytes_forsteri",
    "Non_Penguin_Seal": "Category:Leptonychotes_weddelli",
}


def safe_filename(name: str) -> str:
    """Sanitiza uma string para nome de arquivo seguro."""
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("._")


def query_wikimedia_category(
    category_title: str, max_items: int = 120
) -> List[Dict[str, Any]]:
    """Consulta arquivos de uma categoria do Wikimedia Commons com metadados."""
    results: List[Dict[str, Any]] = []
    gcmcontinue = ""

    while len(results) < max_items:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category_title,
            "gcmtype": "file",
            "gcmlimit": min(50, max_items - len(results)),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 800,
            "format": "json",
        }
        if gcmcontinue:
            params["gcmcontinue"] = gcmcontinue

        url = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Erro ao consultar Wikimedia para %s: %s", category_title, exc)
            break

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            break

        for _, page in pages.items():
            if "imageinfo" not in page or not page["imageinfo"]:
                continue
            info = page["imageinfo"][0]
            mime = info.get("mime", "")
            # Aceitar apenas formatos de imagem válidos (sem vídeos/áudios)
            if not mime.startswith("image/") or mime in ("image/svg+xml", "image/tiff"):
                continue

            metadata = info.get("extmetadata", {})
            license_name = (
                metadata.get("LicenseShortName", {}).get("value")
                or metadata.get("License", {}).get("value")
                or "Unknown"
            )
            artist_raw = metadata.get("Artist", {}).get("value", "Unknown")
            # Limpar tags HTML do nome do artista se houver
            artist = re.sub(r"<[^>]+>", "", artist_raw).strip() or "Unknown"

            results.append(
                {
                    "page_id": page.get("pageid"),
                    "title": page.get("title", ""),
                    "mime": mime,
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "url": info.get("url"),
                    "download_url": info.get("thumburl", info.get("url")),
                    "license": license_name,
                    "artist": artist[:120],
                }
            )
            if len(results) >= max_items:
                break

        if "continue" in data and "gcmcontinue" in data["continue"]:
            gcmcontinue = data["continue"]["gcmcontinue"]
        else:
            break

        time.sleep(0.2)

    return results


def download_file(url: str, dest_path: Path) -> Optional[str]:
    """Baixa um arquivo e retorna seu hash SHA-256 se bem-sucedido."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read()
            if len(content) < 1024:
                return None  # Arquivo suspeito / vazio
            dest_path.write_bytes(content)
            return hashlib.sha256(content).hexdigest()
    except Exception as exc:
        logger.warning("Falha ao baixar %s: %s", url, exc)
        return None


def download_dataset(
    target_dir: Path,
    max_per_class: int = 110,
    download_external: bool = True,
) -> None:
    """Executa o download estruturado das imagens e salva metadados."""
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_records: List[Dict[str, Any]] = []

    for species, category in CATEGORIES.items():
        species_dir = target_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Consultando Wikimedia Commons para %s (%s)...", species, category)
        items = query_wikimedia_category(category, max_items=max_per_class)
        logger.info("Encontrados %d arquivos potenciais para %s.", len(items), species)

        downloaded_count = 0
        for idx, item in enumerate(items, 1):
            title = item["title"]
            base_ext = Path(title).suffix.lower()
            if base_ext not in (".jpg", ".jpeg", ".png", ".webp"):
                base_ext = ".jpg"
            clean_title = safe_filename(Path(title).stem)
            filename = f"{species.lower()}_{idx:03d}_{clean_title[:40]}{base_ext}"
            file_path = species_dir / filename

            if file_path.exists():
                sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
            else:
                sha256 = download_file(item["download_url"], file_path)
                time.sleep(0.15)

            if sha256:
                # Validar se a imagem abre com Pillow
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    downloaded_count += 1
                    metadata_records.append(
                        {
                            "species": species,
                            "filename": filename,
                            "relative_path": str(file_path.relative_to(target_dir.parent)),
                            "wikimedia_title": item["title"],
                            "license": item["license"],
                            "artist": item["artist"],
                            "original_url": item["url"],
                            "download_url": item["download_url"],
                            "sha256": sha256,
                            "width": item["width"],
                            "height": item["height"],
                        }
                    )
                except Exception as exc:
                    logger.warning("Arquivo corrompido descartado: %s (%s)", file_path, exc)
                    if file_path.exists():
                        file_path.unlink()

        logger.info("Download concluído para %s: %d imagens válidas salvas.", species, downloaded_count)

    # Salvar metadados em CSV e JSON
    meta_csv = target_dir / "dataset_sources.csv"
    meta_json = target_dir / "dataset_sources.json"

    if metadata_records:
        keys = list(metadata_records[0].keys())
        with open(meta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metadata_records)

        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(metadata_records, f, indent=2, ensure_ascii=False)

        logger.info("Metadados gravados em %s e %s.", meta_csv, meta_json)

    # Opcional: baixar imagens para external_test
    if download_external:
        ext_dir = target_dir.parent / "external_test"
        ext_dir.mkdir(parents=True, exist_ok=True)
        for name, category in EXTERNAL_CATEGORIES.items():
            logger.info("Baixando imagens de teste externo para %s...", name)
            ext_items = query_wikimedia_category(category, max_items=8)
            for idx, item in enumerate(ext_items, 1):
                clean_title = safe_filename(Path(item["title"]).stem)
                filename = f"ext_{name}_{idx:02d}_{clean_title[:30]}.jpg"
                file_path = ext_dir / filename
                if not file_path.exists():
                    download_file(item["download_url"], file_path)
                    time.sleep(0.15)


if __name__ == "__main__":
    raw_path = Path("data/raw")
    download_dataset(raw_path, max_per_class=105, download_external=True)
