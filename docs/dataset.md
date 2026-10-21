# Documentação do Dataset e Governança de Dados

Este documento registra a proveniência, a metodologia de seleção, a higienização criptográfica e perceptual, os termos de licenciamento e a divisão estratificada do conjunto de dados de fotografias de pinguins.

---

## 1. Fontes de Dados Investigadas

### 1.1. DataLad Penguins (`hub.datalad.org/edu/penguins`)
- **Investigação realizada:** O repositório foi clonado e inspecionado via API Forgejo.
- **Constatação técnica:** O repositório foi concebido para demonstração de gestão distribuída de dados (*Distributed Data Management*) com Git-annex sobre o estudo ecológico de Kristen B. Gorman et al. (2014).
- **Conteúdo de imagens:** A pasta `examples/` contém apenas **3 fotografias no total** (`adelie.jpg`, `chinstrap.jpg`, `gentoo.jpg`), correspondendo a 1 imagem por espécie. As demais pastas (`adelie/`, `chinstrap/`, `gentoo/`) contêm exclusivamente dados tabulares e metadados XML/CSV da iniciativa EDI (*Environmental Data Initiative*).
- **Decisão adotada:** O repositório DataLad é insuficiente para treinar um classificador supervisionado de visão computacional. Ele foi mantido como referência conceitual e educacional, conforme orientação da especificação do projeto.

### 1.2. Repositórios Palmer Penguins (R/Python)
- **Investigação realizada:** Inspecionados os repositórios `lmcinnes/penguins` e `allisonhorst/palmerpenguins`.
- **Constatação técnica:** Ambos reúnem dados biológicos e morfométricos tabulares (tamanho do bico/culmen, comprimento da nadadeira, massa corporal, proporções isotópicas). Não fornecem conjuntos de fotografias de espécimes reais para aprendizado de máquina em visão computacional.

### 1.3. Wikimedia Commons (Fonte Primária Selecionada)
- **Origem:** O [Wikimedia Commons](https://commons.wikimedia.org/) é o maior repositório público aberto de mídias de história natural do mundo.
- **Categorias taxonômicas consultadas:**
  - `Category:Pygoscelis_adeliae` (*Adelie penguin*) — 156 arquivos disponíveis.
  - `Category:Pygoscelis_antarcticus` (*Chinstrap penguin*) — 524 arquivos disponíveis.
  - `Category:Pygoscelis_papua` (*Gentoo penguin*) — 528 arquivos disponíveis.
- **Critério de Inclusão:** Fotografias autênticas capturadas em ambiente natural antártico ou subantártico, metadados verificáveis e formatos de imagem aceitos (`.jpg`, `.jpeg`, `.png`, `.webp`).
- **Critério de Exclusão:** Ilustrações, diagramas vetoriais SVG, arquivos de áudio/vídeo, documentos PDF e imagens com compressão excessiva ou ilegíveis.

---

## 2. Metadados e Licenciamento

Todas as imagens coletadas possuem licenças livres compatíveis com fins acadêmicos e educacionais (principalmente **Creative Commons Attribution**, **CC BY-SA 4.0**, **CC BY-SA 3.0**, **CC BY 2.0** e **Domínio Público**).

O registro completo de proveniência foi gravado em `data/raw/dataset_sources.csv` e `data/raw/dataset_sources.json`, contendo:
- Nome do arquivo local e relativo.
- Título original no Wikimedia Commons.
- Autor / Fotógrafo (*Artist*).
- Licença exata declarada.
- URL original da página do arquivo e URL da mídia.
- Dimensões originais (largura e altura).
- Hash criptográfico SHA-256.

---

## 3. Higienização, Validação e Deduplicação

O script de saneamento (`penguin_mlp/prepare_data.py`) executou três filtros sucessivos:

1. **Validação de Integridade e Abertura:**
   - Abertura com `PIL.Image.open()` seguida de `img.verify()` para descartar cabeçalhos corrompidos.
   - Aplicação de `ImageOps.exif_transpose` para normalização de orientação.
2. **Deduplicação Exata (SHA-256):**
   - Comparação de hashes criptográficos byte-a-byte.
   - **1 duplicata exata detectada e descartada** em Gentoo (`gentoo_068`).
3. **Deduplicação Perceptual (pHash):**
   - Cálculo do hash perceptual via biblioteca `imagehash`.
   - Comparação da Distância de Hamming entre imagens da mesma classe.
   - **2 imagens quase duplicadas descartadas** em Gentoo com distância \(\le 4\) (`gentoo_066` e `gentoo_076`, que consistiam em versões idênticas com pequenas diferenças de metadados ou reenquadramentos mínimos).

### Resumo do Saneamento:
| Espécie | Escaneados | Corrompidos | Duplicatas Exatas | Quase Duplicatas (pHash) | Retidos Finais |
|---|---|---|---|---|---|
| **Adelie** | 105 | 0 | 0 | 0 | **105** |
| **Chinstrap** | 105 | 0 | 0 | 0 | **105** |
| **Gentoo** | 105 | 0 | 1 | 2 | **102** |
| **Total** | **315** | **0** | **1** | **2** | **312** |

---

## 4. Divisão Estratificada dos Dados

A divisão foi realizada pela função `create_stratified_splits` com semente pseudoaleatória fixa `42`:
- **70% Treinamento:** 218 imagens
- **15% Validação:** 47 imagens
- **15% Teste (Intocado):** 47 imagens

### Distribuição por Espécie nos Splits:
| Divisão (Split) | Adelie | Chinstrap | Gentoo | Total do Split | Proporção |
|---|---|---|---|---|---|
| **Treino** | 73 | 74 | 71 | **218** | 69,87% |
| **Validação** | 16 | 15 | 16 | **47** | 15,06% |
| **Teste** | 16 | 16 | 15 | **47** | 15,06% |
| **Total Geral** | **105** | **105** | **102** | **312** | 100,00% |

Os arquivos CSV foram persistidos em:
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- `data/splits/manifest.csv`

### Garantia de Ausência de Vazamento (Data Leakage)
A suíte automatizada de testes (`tests/test_data.py`) verifica formalmente que a interseção dos conjuntos de arquivos de treino, validação e teste é estritamente vazia:
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

---

## 5. Conjunto de Teste Externo (Out-of-Domain)

Para atender ao requisito de robustez e teste de limites do classificador, foi criado o diretório `data/external_test/`, contendo fotografias reais de:
- **Pinguim Imperador** (*Aptenodytes forsteri*) — espécie diferente não pertencente às três classes de treinamento.
- **Foca-de-Weddell** (*Leptonychotes weddellii*) — mamífero antártico (não-pinguim).

Essas imagens permitem demonstrar empiricamente a limitação intrínseca de qualquer classificador discriminativo de mundo fechado (*closed-set classifier*), validando o papel do limiar de confiança na sinalização de anomalias.
