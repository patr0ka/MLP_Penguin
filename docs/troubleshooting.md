# Guia de Resolução de Problemas (Troubleshooting)

Este documento reúne soluções para os erros e comportamentos inesperados mais comuns durante a instalação, preparação de dados, treinamento e inferência do classificador de pinguins.

---

## 1. Problemas de Instalação e Dependências

### Erro: `ModuleNotFoundError: No module named 'penguin_mlp'`
- **Causa:** O pacote `penguin_mlp` reside na pasta `src/` e ainda não foi instalado no ambiente Python ou o `PYTHONPATH` não inclui o diretório raiz.
- **Solução:** No diretório raiz do projeto, instale o pacote em modo editável:
  ```bash
  pip install -e .
  ```
  Ou no Windows PowerShell / Linux antes de executar scripts avulsos:
  ```powershell
  $env:PYTHONPATH = "src"
  ```

### Erro com PyTorch / TorchVision (Incompatibilidade de Versões)
- **Causa:** Conflito entre binários do PyTorch e TorchVision compilados para versões distintas.
- **Solução:** Reinstale as versões coordenadas recomendadas diretamente do canal oficial:
  ```bash
  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

---

## 2. Problemas na Preparação do Dataset

### Erro: `Nenhuma imagem válida encontrada em data/raw`
- **Causa:** O script `download_dataset.py` não foi executado ou a pasta `data/raw` não contém as subpastas `Adelie/`, `Chinstrap/` e `Gentoo/`.
- **Solução:** Execute o download automatizado a partir do Wikimedia Commons:
  ```bash
  python src/penguin_mlp/download_dataset.py
  ```
  Em seguida, execute a preparação e divisão estratificada:
  ```bash
  python -m penguin_mlp.prepare_data --force
  ```

### Arquivo Corrompido ou Ilegível
- **Sintoma:** Aviso no terminal informando que determinada imagem foi descartada.
- **Comportamento normal:** A função `scan_and_validate_dataset` utiliza `PIL.Image.open` com `img.verify()` para capturar falhas em tempo de varredura. Imagens truncadas são isoladas no relatório `data/splits/preparation_report.json` sem interromper a execução do pipeline.

---

## 3. Problemas durante o Treinamento

### Execução Lenta em CPU
- **Causa:** O processamento em CPU é mais lento que em GPU com aceleração CUDA.
- **Mitigação:** 
  1. A CNN está congelada (`requires_grad = False`, `torch.no_grad()`), o que economiza mais de 80% do processamento em relação a um treinamento de ponta a ponta.
  2. Caso possua uma GPU NVIDIA compatível, instale o PyTorch com suporte a CUDA:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     ```
  3. Reduza a resolução para testes rápidos via argumento ou arquivo `configs/default.yaml`.

### Parada Antecipada (Early Stopping) Acionada
- **Sintoma:** O treinamento encerra antes de atingir o número máximo configurado de épocas (ex: para na época 18 de 40).
- **Causa:** Isso **não é um erro**; é o comportamento esperado do mecanismo de parada antecipada. Quando a perda de validação deixa de diminuir pelo número consecutivo de épocas configurado na paciência (`patience: 6`), o treinamento é interrompido para evitar *overfitting*. O melhor checkpoint já está salvo em `artifacts/checkpoints/best_model.pt`.

---

## 4. Problemas de Inferência

### Erro: `FileNotFoundError: Arquivo de checkpoint não encontrado no caminho: artifacts/checkpoints/best_model.pt`
- **Causa:** Tentativa de classificar uma fotografia antes de executar o treinamento da rede.
- **Solução:** O modelo precisa ser treinado primeiro para que os pesos da MLP existam:
  ```bash
  python -m penguin_mlp.train
  ```

### Alerta: `Previsão com baixa confiança (X% < limiar de 60.0%)`
- **Causa:** A maior probabilidade entre as três classes ficou abaixo do limiar configurado (padrão: 0.60).
- **Diagnóstico:**
  1. A imagem pode conter uma espécie diferente de pinguim (ex: Pinguim-imperador, Pinguim-de-magalhães).
  2. A imagem pode ser de um animal que não é um pinguim (ex: foca, gaivota).
  3. O pinguim pode estar excessivamente distante, ocluído por rochas ou mal iluminado.
- **Ajuste do Limiar:** Caso queira alterar a sensibilidade do alerta, utilize o parâmetro `--threshold`:
  ```bash
  python -m penguin_mlp.predict --image "data/to_predict/foto.jpg" --threshold 0.50
  ```

### Múltiplos Pinguins na Fotografia
- **Limitação Conhecida:** O extrator ResNet18 opera na imagem global. Fotografias contendo colônias com dezenas de espécimes distantes ou espécies mistas podem produzir vetores de características com sinais mistos. A recomendação da primeira versão é fornecer fotografias onde o espécime principal esteja razoavelmente em destaque.
