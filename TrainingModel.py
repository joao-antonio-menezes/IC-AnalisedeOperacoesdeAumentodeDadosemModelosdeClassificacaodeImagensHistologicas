import os
import numpy as np
import keras
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# Configuração de GPUs -> utiliza GPU se reconhece no sistema

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Configuração de crescimento de memória da GPU ativada.")
    except RuntimeError as e:
        print(e)

# Variáveis e constantes globais

HIST_CLASSES = ['Healthy', 'Mild', 'Moderate', 'Severe']
NUM_CLASSES = len(HIST_CLASSES)
INPUT_SHAPE = (450, 250, 3)
data_title = ""


# FUNÇÕES AUXILIARES (PLOTS E CARREGAMENTOS)

def plotar_imgs(imgs, labels, num_imgs):
    plt.figure(figsize=(10, 5))
    for i in range(num_imgs):
        ax = plt.subplot(5, 2, i + 1) # 5 linhas, 2 colunas, i + 1 para ir inserindo as imagens no quadro
        plt.imshow(imgs[i], cmap='gray')
        plt.title(f"Label: {labels[i]}") # labels das classes
        plt.axis("off")
    plt.tight_layout() # manter um "padding"
    plt.show()


def carregar_imagens_custom(path, target_size=(450, 250), test_size=0.2, random_state=42):
    X = []
    y = []
    class_names = sorted(os.listdir(path))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_path = os.path.join(path, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.tif')):
                img_path = os.path.join(class_path, fname)
                img = Image.open(img_path).convert("RGB").resize(target_size)
                X.append(np.array(img))
                y.append(class_to_idx[class_name])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return (x_train, y_train), (x_test, y_test)


def plot_hist2(hist, title=f"Curvas de Validação e Perda - {data_title}", filename=f"{data_title}_curvas.png"):
    plt.figure(figsize=(12, 5))
    plt.suptitle(title, fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(hist.history["categorical_accuracy"], label="Train Accuracy")
    if "val_categorical_accuracy" in hist.history:
        plt.plot(hist.history["val_categorical_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.ylim(0, 1)

    # perda
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"], label="Train Loss")
    if "val_loss" in hist.history:
      plt.plot(hist.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")

    # ajusta o intervalo do eixo y para o intervalo entre 0 e 1
    plt.ylim(0, 1)

    # para os gráficos não ficarem em cima das curvas
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # salva imagem
    plt.savefig(filename)
    print(f"Gráfico de curvas salvo como '{filename}'")

    plt.show()


def exibir_matriz_confusao(model, x_test, y_test, class_labels, title=f"Matriz de Confusão - {data_title}", filename=f"{data_title}_cm.png"):
   
    predicoes = model.predict(x_test)
    
    p_classes = np.argmax(predicoes, axis=1) # classes predições

    v_classes = np.argmax(y_test, axis=1) # classes verdadeiro


    report_str = classification_report(v_classes, p_classes, target_names=class_labels)
    print("Relatório de Classificação:")
    print(report_str)

    # 4. Calcular a matriz de confusão
    cm = confusion_matrix(v_classes, p_classes)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(title, fontsize=16)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsão')

    fig.subplots_adjust(left=0.1, right=0.6, top=0.9, bottom=0.1)

    fig.text(0.65, 0.85, "Relatório de Classificação", fontsize=12, weight='bold')
    fig.text(0.65, 0.75, report_str,
             fontfamily='monospace', 
             fontsize=11,
             va='top')
    
    plt.savefig(filename, bbox_inches='tight')
    print(f"Matriz de confusão e relatório salvos como '{filename}'")
    plt.show()

hist_classes = ['Healthy','Mild','Moderate','Severe']

# EXECUÇÃO DO TREINAMENTO

def executar_treinamento(data_directory):
    print(f"\n{'='*80}")
    print(f"INICIANDO TREINAMENTO PARA: {data_title}")
    print(f"{'='*80}")

    (x_train, y_train), (x_test, y_test) = carregar_imagens_custom(data_directory, target_size=(450, 250))
    
    # Parâmetros de controle e análise
    plotar_imgs(x_train, y_train, 2)
    plotar_imgs(x_test, y_test, 2)
    print(f"x teste: {x_test.shape}")
    print(f"x train: {x_train.shape}")
    print(f"y teste: {y_test.shape}")
    print(f"y train: {y_train.shape}")

    # escala as imagens para [0, 1]
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = x_train / 255
    x_test = x_test / 255

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # converte para one hot encoding 1 -> [0,1,0,0,0,0,0,0,0,0]
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # plotando imagens após converter para one hot
    plotar_imgs(x_train, y_train, NUM_CLASSES)

    # organizando conjuntos de teste e validação
    num_train_samples = x_train.shape[0]
    num_val_samples = int(0.2*x_train.shape[0])
    x_val, x_train = x_train[:num_val_samples], x_train[num_val_samples:]
    y_val, y_train = y_train[:num_val_samples], y_train[num_val_samples:]
    print(f"Dados carregados de '{data_directory}': {x_train.shape[0]} amostras de treino, {x_val.shape[0]} de validação.")

    # SHAPE DO RESNET50

    input = tf.keras.Input(shape=(250,450,3))

    resnet = tf.keras.applications.ResNet50(weights='imagenet',
                                                include_top = False,
                                                input_tensor = input)
    # global max pooling
    gap = tf.keras.layers.GlobalMaxPooling2D()(resnet.output)
    bn_0 = BatchNormalization()(gap)
    dropout_0 = tf.keras.layers.Dropout(0.5)(bn_0)
    dense_1 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(dropout_0)
    bn_1 = BatchNormalization()(dense_1)
    dropout_1 = tf.keras.layers.Dropout(0.3)(bn_1)
    dense_2 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(dropout_1)
    bn_2 = BatchNormalization()(dense_2)
    dropout_2 = tf.keras.layers.Dropout(0.3)(bn_2)

    # camada de classificação (considerando one hot encoding)
    output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(dropout_2)

    func_model = tf.keras.Model(resnet.input, output)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

    # COMPILE DO MODELO

    func_model.compile(
        loss  = tf.keras.losses.CategoricalCrossentropy(),
        metrics = [tf.keras.metrics.CategoricalAccuracy()],
        optimizer = tf.keras.optimizers.Adam())
    
    print("--> Modelo criado e compilado")

    # TREINAMENTO DO MODELO

    hist = func_model.fit( 
                        x_train, y_train, 
                        batch_size=32,
                        epochs=3,
                        verbose=2,
                        validation_data=(x_val, y_val),
                        callbacks=[reduce_lr]
    )

    print("--> Treinamento concluído")

    # SALVANDO MODELOS E EXIBINDO RESULTADOS

    if not os.path.exists('Modelos'):
        os.makedirs('Modelos')

    if not os.path.exists('Resultados'):
        os.makedirs('Resultados')

    curves_filename = f"Resultados/{data_title}_curvas.png"
    cm_filename = f"Resultados/{data_title}_cm.png"
    model_filename = f"Modelos/{data_title}.keras"

    plot_hist2(hist, title=f"Curvas de Treinamento (Validação e Perda) - {data_title}", filename=curves_filename)
    exibir_matriz_confusao(func_model, x_test, y_test, hist_classes, title=f"Matriz de Confusão - {data_title}", filename=cm_filename)

    func_model.save(model_filename)

    print(f"--> Modelo salvo como '{model_filename}'")
    print(f"FIM DO TREINAMENTO PARA: {data_title}")

# =======================================================================

if __name__ == "__main__":
    
    lista_de_configuracoes = [
        {
            "data_title": "Aumento10x",
            "data_directory": "Testes Aumentos Diferentes/TesteTamanhosAumentos/Aumento10x"
        },
    ]

    # Loop principal que executa cada configuração da lista
    for config in lista_de_configuracoes:
        try:
            data_title = config["data_title"]
            executar_treinamento(data_directory=config["data_directory"])
        except Exception as e:
            print(f"\n!!!!!! ERRO NA EXECUÇÃO '{config['data_title']}' !!!!!!")
            print(f"Ocorreu um erro: {e}")
            continue
            
    print("\nTODOS OS TREINAMENTOS FORAM CONCLUÍDOS.")