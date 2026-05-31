# ERCP Image Classifier

Sistema de classificação automática de imagens fluoroscópicas de CPRE (*Colangiopancreatografia Retrógrada Endoscópica*) usando **Deep Learning**. O objetivo é classificar imagens em quatro classes diagnósticas:

- `Biliary_Leaks` — fugas de bílis;
- `Lithiasis` — cálculos biliares;
- `Normal` — achados normais;
- `Stricture` — estenoses ductais.

O projeto foi desenvolvido no âmbito da unidade curricular de **Aprendizagem Profunda**, usando o dataset **MIQR-CC** e tendo como referência a baseline indicada no enunciado: **F1-macro = 0.738**. A solução final usa modelos pré-treinados em ImageNet, pré-processamento com **CLAHE**, augmentação robusta, **Focal Loss**, **WeightedRandomSampler**, *fine-tuning* diferenciado, **Test Time Augmentation** e interpretabilidade com **Grad-CAM++**.

---

## 1. Estrutura do projeto

A estrutura principal do repositório é a seguinte:

```text
ERCP_Image_Classifier/
├── swin/
│   ├── processed_dataset/
│   │   ├── train/
│   │   │   ├── Biliary_Leaks/
│   │   │   ├── Lithiasis/
│   │   │   ├── Normal/
│   │   │   └── Stricture/
│   │   ├── val/
│   │   │   ├── Biliary_Leaks/
│   │   │   ├── Lithiasis/
│   │   │   ├── Normal/
│   │   │   └── Stricture/
│   │   └── test/
│   │       ├── Biliary_Leaks/
│   │       ├── Lithiasis/
│   │       ├── Normal/
│   │       └── Stricture/
│   │
│   ├── eda_out/
│   │   ├── class_distribution.csv
│   │   ├── class_distribution.png
│   │   ├── dataset_index.csv
│   │   ├── image_sizes.png
│   │   ├── intensity_stats.csv
│   │   └── samples_*.png
│   │
│   ├── gradcam_out/
│   │   └── ...
│   │
│   ├── dataset.py
│   ├── transforms.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── ensemble_tta.py
│   ├── predict.py
│   ├── gradcam.py
│   ├── eda.py
│   ├── requirements.txt
│   └── vai_ser_este-2.ipynb
│
├── .gitignore
└── README.md
```

### Descrição dos ficheiros principais

| Ficheiro | Função |
|---|---|
| `dataset.py` | Define o `ERCPDataset`, lê imagens por classe, aplica CLAHE e devolve imagem + etiqueta. |
| `transforms.py` | Define transformações de treino, validação/teste e TTA com Albumentations. |
| `model.py` | Define o `ERCPClassifier`, um wrapper genérico para modelos da biblioteca `timm`. |
| `losses.py` | Implementa a `FocalLoss` com pesos por classe e `label_smoothing`. |
| `train.py` | Script principal de treino, validação, seleção do melhor checkpoint e avaliação final com TTA. |
| `ensemble_tta.py` | Avalia um ou mais checkpoints com TTA e permite calcular ensemble ponderado. |
| `predict.py` | Faz inferência numa imagem individual e devolve probabilidades por classe. |
| `gradcam.py` | Gera mapas de calor Grad-CAM/Grad-CAM++ para análise de interpretabilidade. |
| `eda.py` | Executa a análise exploratória do dataset e gera gráficos/tabelas descritivas. |
| `requirements.txt` | Lista as dependências necessárias para correr o projeto. |
| `vai_ser_este-2.ipynb` | Notebook auxiliar usado durante a exploração/experimentação. |

---

## 2. Ambiente recomendado

O treino foi pensado para correr com GPU CUDA. O uso de GPU é altamente recomendado, sobretudo para modelos como Swin Transformer Base ou ConvNeXt Large.

Requisitos recomendados:

- Python 3.10 ou superior;
- CUDA disponível, se possível;
- PyTorch com suporte CUDA;
- GPU com pelo menos 12 GB de VRAM para os modelos principais;
- sistema Linux, Google Colab ou ambiente equivalente.

---

## 3. Instalação

Clonar o repositório:

```bash
git clone https://github.com/diogoazevedo04/ERCP_Image_Classifier.git
cd ERCP_Image_Classifier/swin
```

Criar e ativar um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instalar dependências:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Nota: se a instalação do PyTorch não ficar com suporte CUDA, instalar a versão adequada a partir das instruções oficiais do PyTorch, de acordo com a versão CUDA disponível na máquina.

---

## 4. Dataset

O projeto espera que o dataset já esteja dividido em três partições fixas:

```text
processed_dataset/train
processed_dataset/val
processed_dataset/test
```

Cada partição deve conter uma pasta por classe:

```text
processed_dataset/
├── train/
│   ├── Biliary_Leaks/
│   ├── Lithiasis/
│   ├── Normal/
│   └── Stricture/
├── val/
│   ├── Biliary_Leaks/
│   ├── Lithiasis/
│   ├── Normal/
│   └── Stricture/
└── test/
    ├── Biliary_Leaks/
    ├── Lithiasis/
    ├── Normal/
    └── Stricture/
```

O código **não faz um train-test split aleatório em tempo de execução**. A divisão é fixa e é carregada diretamente destas pastas. Isto garante que todos os modelos são treinados, validados e testados exatamente nas mesmas imagens, permitindo comparação justa entre arquiteturas.

Distribuição usada no trabalho:

| Partição | Número de imagens | Utilização |
|---|---:|---|
| Treino | 1067 | Otimização dos pesos do modelo |
| Validação | 234 | Seleção do melhor checkpoint e early stopping |
| Teste | 267 | Avaliação final |
| Total | 1568 | — |

Distribuição do conjunto de teste:

| Classe | Imagens |
|---|---:|
| `Biliary_Leaks` | 17 |
| `Lithiasis` | 123 |
| `Normal` | 43 |
| `Stricture` | 84 |

---

## 5. O que o pipeline faz

### 5.1 Leitura e pré-processamento

O carregamento das imagens é feito pela classe `ERCPDataset`, definida em `dataset.py`.

Para cada imagem:

1. a imagem é lida com OpenCV em formato BGR;
2. se a leitura com OpenCV falhar, é usado fallback com PIL;
3. é aplicado CLAHE por defeito;
4. a imagem é convertida para RGB;
5. são aplicadas as transformações adequadas à fase: treino, validação/teste ou TTA;
6. a imagem é normalizada com estatísticas ImageNet;
7. a imagem é convertida para tensor PyTorch.

### 5.2 CLAHE

O CLAHE (*Contrast Limited Adaptive Histogram Equalization*) é usado para melhorar o contraste local das imagens fluoroscópicas.

A aplicação é feita assim:

1. conversão da imagem BGR para LAB;
2. separação do canal de luminância `L`;
3. aplicação de CLAHE apenas ao canal `L`, com:
   - `clipLimit = 2.5`;
   - `tileGridSize = (8, 8)`;
4. recombinação dos canais LAB;
5. conversão final para RGB.

Isto permite melhorar a visibilidade das estruturas biliares sem alterar diretamente os canais cromáticos. A imagem final é mantida em três canais para garantir compatibilidade com modelos pré-treinados em ImageNet.

### 5.3 Normalização

Todas as imagens são normalizadas com as estatísticas ImageNet:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Esta normalização é necessária porque os modelos usados foram inicializados com pesos pré-treinados em ImageNet.

---

## 6. Data augmentation

As transformações de augmentação são aplicadas **apenas ao conjunto de treino**.

O pipeline de treino inclui:

- `LongestMaxSize` para preservar proporção;
- `PadIfNeeded` para completar a imagem;
- `RandomResizedCrop` para obter a dimensão final;
- `HorizontalFlip`, com `p=0.5`;
- `VerticalFlip`, com `p=0.2`;
- rotação até ±20 graus, com `p=0.7`;
- transformação afim com translação, escala e shear;
- ruído ou blur: `GaussNoise`, `GaussianBlur` ou `MotionBlur`;
- alteração de intensidade: `RandomBrightnessContrast` ou `RandomGamma`;
- `CoarseDropout`, removendo regiões retangulares da imagem;
- normalização ImageNet;
- conversão para tensor.

Além das transformações sobre imagem, são usadas duas técnicas ao nível do batch:

- **Mixup**, com `alpha=0.2` e probabilidade 30%;
- **CutMix**, com `alpha=1.0` e probabilidade 30%.

Estas técnicas ajudam a reduzir overfitting e melhoram a generalização num dataset pequeno e desbalanceado.

---

## 7. Modelo

A arquitetura principal é o `ERCPClassifier`, definido em `model.py`.

Este modelo é um wrapper genérico para qualquer backbone da biblioteca `timm`:

1. carrega um modelo pré-treinado;
2. remove a cabeça original com `num_classes=0`;
3. usa o backbone como extrator de características;
4. adiciona uma cabeça de classificação própria:
   - `LayerNorm`;
   - `Dropout(0.3)`;
   - camada linear para 4 classes.

O melhor modelo obtido foi o **Swin Transformer Base**, especificamente:

```text
swin_base_patch4_window12_384.ms_in22k_ft_in1k
```

Este modelo usa *Shifted Window Attention*, permitindo capturar relações locais e globais nas imagens, mantendo custo computacional controlado.

---

## 8. Treino

O treino é executado com `train.py`.

Componentes principais:

- otimizador `AdamW`;
- `weight_decay = 0.05`;
- dois grupos de parâmetros:
  - backbone com learning rate reduzido por fator `0.2`;
  - cabeça de classificação com learning rate completo;
- `FocalLoss` com:
  - `gamma = 1.5`;
  - `label_smoothing = 0.05`;
  - pesos por classe;
- `WeightedRandomSampler` com pesos proporcionais a `1 / sqrt(freq)`;
- scheduler com 3 épocas de warmup e `cosine annealing`;
- mixed precision com `autocast` e `GradScaler`;
- gradient clipping com `max_norm = 1.0`;
- early stopping com paciência de 18 épocas;
- seleção do melhor checkpoint com base no F1-macro de validação.

O conjunto de teste só é usado no final, depois de escolhido o melhor checkpoint.

---

## 9. Como correr

### 9.1 Análise exploratória dos dados

```bash
python eda.py \
  --data_dir processed_dataset \
  --output_dir eda_out
```

Este comando gera:

- distribuição de classes por split;
- índice do dataset;
- histogramas de dimensões das imagens;
- estatísticas de intensidade;
- amostras visuais por classe;
- comparação original vs CLAHE.

Os resultados ficam em:

```text
eda_out/
```

---

### 9.2 Treinar o Swin Transformer Base

```bash
python train.py \
  --data_dir processed_dataset \
  --model_name swin_base_patch4_window12_384.ms_in22k_ft_in1k \
  --img_size 384 \
  --batch_size 12 \
  --epochs 80 \
  --lr 4e-4 \
  --backbone_lr_mult 0.2 \
  --weight_decay 0.05 \
  --warmup_epochs 3 \
  --patience 18 \
  --focal_gamma 1.5 \
  --output_dir checkpoints
```

Durante o treino, o script:

1. carrega treino, validação e teste;
2. aplica CLAHE e augmentação;
3. usa sampler balanceado no treino;
4. treina o modelo;
5. calcula F1-macro na validação;
6. guarda o melhor checkpoint;
7. faz avaliação final no teste com TTA.

O melhor checkpoint fica em:

```text
checkpoints/
```

---

### 9.3 Treinar outros modelos candidatos

Exemplo para ConvNeXt Base:

```bash
python train.py \
  --data_dir processed_dataset \
  --model_name convnext_base.fb_in22k_ft_in1k \
  --img_size 384 \
  --batch_size 16 \
  --epochs 80 \
  --lr 5e-4 \
  --backbone_lr_mult 0.2 \
  --output_dir checkpoints
```

Exemplo para EfficientNetV2-M:

```bash
python train.py \
  --data_dir processed_dataset \
  --model_name tf_efficientnetv2_m.in21k_ft_in1k \
  --img_size 384 \
  --batch_size 12 \
  --epochs 80 \
  --lr 5e-4 \
  --backbone_lr_mult 0.2 \
  --output_dir checkpoints
```

Exemplo para ConvNeXt Large:

```bash
python train.py \
  --data_dir processed_dataset \
  --model_name convnext_large.fb_in22k_ft_in1k \
  --img_size 448 \
  --batch_size 8 \
  --epochs 80 \
  --lr 3e-4 \
  --backbone_lr_mult 0.2 \
  --output_dir checkpoints
```

> Os nomes exatos dos modelos dependem da versão instalada da biblioteca `timm`. Caso algum nome não seja reconhecido, confirmar os modelos disponíveis com `timm.list_models()`.

---

### 9.4 Avaliação com TTA e ensemble

Para avaliar um checkpoint com TTA:

```bash
python ensemble_tta.py \
  --data_dir processed_dataset \
  --split test \
  --checkpoints checkpoints/best_swin_base_patch4_window12_384_ms_in22k_ft_in1k.pt \
  --batch_size 16
```

Para avaliar um ensemble ponderado de vários modelos:

```bash
python ensemble_tta.py \
  --data_dir processed_dataset \
  --split test \
  --checkpoints \
    checkpoints/best_swin.pt \
    checkpoints/best_convnext_base.pt \
    checkpoints/best_efficientnetv2_m.pt \
    checkpoints/best_convnext_large.pt \
  --weights 0.471 0.353 0.118 0.059 \
  --batch_size 16
```

O script calcula:

- probabilidades médias com TTA;
- predições finais;
- F1-macro;
- `classification_report`;
- matriz de confusão;
- ficheiros `.npy` com probabilidades e labels.

---

### 9.5 Inferência numa imagem individual

```bash
python predict.py \
  --checkpoint checkpoints/best_swin.pt \
  --image caminho/para/imagem.png
```

O script devolve uma probabilidade para cada uma das quatro classes e imprime a classe prevista.

Exemplo de saída esperada:

```text
Imagem: exemplo.png
  Biliary_Leaks    0.0342
  Lithiasis        0.8121
  Normal           0.1045
  Stricture        0.0492

>>> Previsão: Lithiasis (conf=0.812)
```

---

### 9.6 Grad-CAM++

Para gerar mapas de interpretabilidade:

```bash
python gradcam.py \
  --checkpoint checkpoints/best_swin.pt \
  --data_dir processed_dataset \
  --split test \
  --output_dir gradcam_out \
  --n_per_class 8 \
  --method gradcam++
```

O script guarda exemplos por classe em:

```text
gradcam_out/
├── Biliary_Leaks/
├── Lithiasis/
├── Normal/
└── Stricture/
```

Cada imagem gerada contém, lado a lado:

1. imagem original após pré-processamento;
2. mapa de ativação Grad-CAM++ sobreposto.

No caso do Swin Transformer, o código aplica uma função de `reshape_transform`, necessária para converter tokens do transformer em mapas espaciais compatíveis com Grad-CAM++.

---

## 10. Test Time Augmentation

Na avaliação final, cada imagem é avaliada em seis variantes determinísticas:

1. imagem original;
2. flip horizontal;
3. flip vertical;
4. rotação +10 graus;
5. rotação -10 graus;
6. flip horizontal seguido de rotação +10 graus.

Para cada variante, o modelo calcula probabilidades com `softmax`. A probabilidade final é a média das probabilidades das seis variantes. A classe prevista é aquela com maior probabilidade média.

A TTA não altera os pesos do modelo. Serve apenas para tornar a inferência mais estável perante pequenas variações geométricas.

---

## 11. Métricas

A métrica principal é o **F1-macro**, calculado com:

```python
f1_score(labels, preds, average="macro")
```

O F1-macro é a média simples dos F1-scores das quatro classes. Esta métrica é adequada porque o dataset é desbalanceado: classes raras, como `Biliary_Leaks`, têm o mesmo peso que classes frequentes, como `Lithiasis`.

Além do F1-macro, são reportados:

- accuracy;
- precision macro;
- recall macro;
- classification report por classe;
- matriz de confusão.

Resultados principais obtidos no conjunto de teste com TTA:

| Modelo | Tipo | F1-macro | Accuracy |
|---|---|---:|---:|
| Swin-B | Transformer | 0.7465 | 78.65% |
| ConvNeXt Base | CNN | 0.7025 | 76.78% |
| EfficientNetV2-M | CNN | 0.6521 | 75.66% |
| ConvNeXt Large | CNN | 0.5455 | 71.54% |

Resultados por classe para o melhor modelo, Swin-B:

| Classe | Precision | Recall | F1-score | Suporte |
|---|---:|---:|---:|---:|
| Biliary Leaks | 0.7143 | 0.5882 | 0.6452 | 17 |
| Lithiasis | 0.8261 | 0.7724 | 0.7983 | 123 |
| Normal | 0.6667 | 0.7442 | 0.7033 | 43 |
| Stricture | 0.8111 | 0.8690 | 0.8391 | 84 |
| Macro avg | 0.7545 | 0.7435 | 0.7465 | 267 |

Matriz de confusão do Swin-B no teste com TTA:

```text
[[10,  2,  5,  0],
 [ 2, 95, 11, 15],
 [ 2,  7, 32,  2],
 [ 0, 11,  0, 73]]
```

As linhas representam as classes reais e as colunas representam as classes previstas, na ordem:

```text
Biliary_Leaks, Lithiasis, Normal, Stricture
```

---

## 12. Resumo do fluxo completo

O fluxo experimental completo é:

1. preparar dataset processado em `train`, `val` e `test`;
2. correr `eda.py` para análise exploratória;
3. treinar os modelos candidatos com `train.py`;
4. selecionar o melhor checkpoint pelo F1-macro de validação;
5. avaliar no teste com TTA;
6. comparar os modelos com base no F1-macro;
7. gerar Grad-CAM++ para interpretabilidade;
8. usar `predict.py` para inferência individual.

---

## 13. Notas de reprodutibilidade

Para garantir reprodutibilidade:

- a divisão treino/validação/teste é fixa;
- o treino define semente aleatória por defeito (`seed=42`);
- o melhor checkpoint é escolhido apenas com base no conjunto de validação;
- o conjunto de teste é usado apenas na avaliação final;
- as métricas são calculadas com funções padrão do `scikit-learn`;
- os outputs de EDA, probabilidades e Grad-CAM++ são guardados em disco.

---

## 14. Referência do dataset

Dataset MIQR-CC:

```text
Curated endoscopic retrograde cholangiopancreatography images dataset
DOI: 10.6084/m9.figshare.31079236
GitHub: https://github.com/monicaccmartins/MIQR-CC-Dataset
```

---

## 15. Autores

Projeto desenvolvido por:

- Diogo Azevedo;
- João Azevedo;
- João Loureiro;
- Mariana Neiva.

Universidade do Minho — Aprendizagem Profunda — 2025/2026.
