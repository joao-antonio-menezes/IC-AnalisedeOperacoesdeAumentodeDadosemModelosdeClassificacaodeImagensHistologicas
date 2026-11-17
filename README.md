# # Análise de Operações de Aumento de Dados na Classificação de Imagens Histológicas

Repositório contendo os artefatos desenvolvidos na Iniciação Científica realizada na **Universidade Federal de Uberlândia (UFU)**. O projeto investiga como diferentes operações e volumetrias de *data augmentation* impactam o treinamento e a capacidade de generalização de uma Rede Neural Convolucional (ResNet50) na classificação de displasias epiteliais da cavidade oral em quatro graus de severidade (*Healthy*, *Mild*, *Moderate* e *Severe*)[cite: 2, 4, 8].

---

## 📁 Arquivos do Repositório

* **`DataAugmentation_2.ipynb`** *(Notebook Jupyter)*: Código responsável por aplicar transformações geométricas e morfológicas (`Horizontal Flip`, `Vertical Flip`, `Rotation`, `Elastic Transformation`, `Grid Distortion` e `Optical Distortion`) sobre o conjunto original de imagens histológicas (`.tif`, $450 \times 250$ pixels)[cite: 3, 8, 10]. Permite configurar o fator de multiplicação e a probabilidade de aplicação das operações para gerar os novos datasets sintéticos[cite: 3, 10, 12].
* **`TrainingModel.py`** *(Script Python)*: Código responsável pelo pipeline de treinamento e avaliação do modelo. Realiza o carregamento estratificado (80% treino/validação e 20% teste), pré-processamento, construção da arquitetura baseada em **ResNet50** (com *transfer learning* da ImageNet, *Global Max Pooling* e camadas densas regularizadas), execução do treinamento e exportação automática das curvas de aprendizado, matrizes de confusão e métricas estatísticas[cite: 2, 8, 9].
* **`Análise de Operações de Aumento de Dados em Modelos de Classificação de Imagens Histológicas.pdf`**: Artigo científico completo contendo a fundamentação teórica, o detalhamento dos experimentos realizados, a comparação quantitativa entre os arranjos de aumento testados e as conclusões do estudo[cite: 4, 5, 8, 16].
* **Modelos Treinados (`<TamanhoAumento>x_<Op1>-<Op2>-<OpN>.keras`)**: Arquivos contendo os pesos e as arquiteturas dos modelos já treinados e avaliados durante os experimentos (ex.: `13x_Hf-Vf-R.keras`, `13x_Hf-Vf-Gd.keras`)[cite: 2, 14]. A nomenclatura segue o fator multiplicador aplicado ao dataset original seguido pelas siglas das operações utilizadas (`Hf`: *Horizontal Flip*, `Vf`: *Vertical Flip*, `R`: *Rotation*, `Et`: *Elastic Transformation*, `Gd`: *Grid Distortion*, `Od`: *Optical Distortion*)[cite: 14].

---

## 🔄 Passo a Passo para Reprodução dos Experimentos

1. **Preparação do Ambiente**:
   Instale as bibliotecas necessárias para manipulação de imagens e treinamento das redes neurais:
   ```bash
   pip install tensorflow keras torch torchvision albumentations opencv-python-headless pillow numpy scikit-learn matplotlib seaborn
Organização do Dataset Original:
Estruture o diretório raiz das imagens originais separando-as em subpastas correspondentes às quatro classes (Healthy, Mild, Moderate e Severe)[cite: 2, 3].Geração dos Datasets Aumentados:Abra o notebook DataAugmentation_2.ipynb.   Na célula principal (__main__), defina o caminho de entrada (dataset_path), o diretório de saída (output_path), a lista de operações desejadas (augmentation) e o fator multiplicador (aumentos).   Execute todas as células para gerar e salvar as imagens expandidas.   Treinamento e Avaliação do Modelo:Abra o arquivo TrainingModel.py e localize lista_de_configuracoes no bloco principal.   Configure data_title com o identificador do experimento (ex.: 13x_Hf-Vf-Gd) e data_directory com o caminho da pasta gerada na etapa anterior.   Execute o script:   Bashpython TrainingModel.py
Ao término da execução, o modelo treinado (.keras) será salvo na pasta Modelos/ e os gráficos de perda/acurácia e a matriz de confusão (.png) serão gerados na pasta Resultados/[cite: 2].✒️ AutoresJoão Antônio Menezes Jordão — Pesquisador / Graduando (UFU)   Marcelo Zanchetta do Nascimento — Orientador (UFU)[cite: 4]
Repositório da Iniciação Científica de João Antônio Menezes Jordão e Marcelo Zanchetta do Nascimento. O projeto analisa o impacto de *data augmentation* na classificação de imagens histológicas orais, englobando o código de aumento, o treinamento via ResNet50 e o artigo científico. 
