# Brain MRI Tumor Detection - Model Benchmark

import itertools
import os
import shutil
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm # type: ignore

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import RMSprop # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img # type: ignore

import imutils # type: ignore
warnings.filterwarnings("ignore", category=UserWarning, module='keras.src.trainers.data_adapters.py_dataset_adapter')

# Configuración
RANDOM_SEED = 123
IMG_SIZE = (224, 224)
EPOCHS = 100
BATCH_SIZE = 32
VAL_BATCH_SIZE = 16

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# 1. Preparación de datos
def create_folders():
    """Crea la estructura de carpetas necesaria"""
    folders = ['TRAIN/YES', 'TRAIN/NO', 'TEST/YES', 'TEST/NO', 'VAL/YES', 'VAL/NO',
               'TRAIN_CROP/YES', 'TRAIN_CROP/NO', 'TEST_CROP/YES', 'TEST_CROP/NO', 'VAL_CROP/YES', 'VAL_CROP/NO']
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def split_dataset(input_path):
    """Divide el dataset en train/val/test"""
    for CLASS in os.listdir(input_path):
        if (os.path.isfile(CLASS) == False) and (CLASS.lower() in ["yes", "no"]):
            IMG_NUM = len(os.listdir(os.path.join(input_path, CLASS)))
            for n, FILE_NAME in enumerate(os.listdir(os.path.join(input_path, CLASS))):
                img = os.path.join(input_path, CLASS, FILE_NAME)
                if n < 5:
                    shutil.copy(img, f'TEST/{CLASS.upper()}/{FILE_NAME}')
                elif n < 0.8 * IMG_NUM:
                    shutil.copy(img, f'TRAIN/{CLASS.upper()}/{FILE_NAME}')
                else:
                    shutil.copy(img, f'VAL/{CLASS.upper()}/{FILE_NAME}')

def load_data(dir_path):
    """Carga imágenes y etiquetas desde directorio"""
    X = []
    y = []
    i = 0
    labels = dict()

    for path in sorted(os.listdir(dir_path)):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(os.path.join(dir_path, path)):
                if not file.startswith('.'):
                    img = cv2.imread(os.path.join(dir_path, path, file))
                    X.append(img)
                    y.append(i)
            i += 1

    return np.array(X, dtype=object), np.array(y), labels

def plot_samples(X, y, labels_dict, n=50):
    """Muestra una galería de imágenes"""
    for index in range(len(labels_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        j = 10
        i = int(n/j)

        plt.figure(figsize=(15,6))
        c = 1
        for img in imgs:
            plt.subplot(i,j,c)
            plt.imshow(img[0])
            plt.xticks([])
            plt.yticks([])
            c += 1
        plt.suptitle('Tumor: {}'.format(labels_dict[index]))
        plt.show()

# 2. Preprocesamiento de imágenes
def crop_imgs(set_name, add_pixels_value=0):
    """Recorta la región cerebral de cada imagen"""
    set_new = []
    
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()
        set_new.append(new_img)
        
    return np.array(set_new, dtype=object)

def plot_samples_class(X, y, labels, target_class, num_samples=15):
    """Visualiza imágenes de una clase específica"""
    plt.figure(figsize=(15, 3))
    count = 0
    for i in range(len(X)):
        if y[i] == target_class:
            plt.subplot(1, num_samples, count + 1)
            plt.imshow(cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            count += 1
            if count == num_samples:
                break
    plt.suptitle(f"Tumor: {labels[target_class]}")
    plt.tight_layout()
    plt.show()

def save_new_images(x_set, y_set, folder_name):
    """Guarda imágenes en carpetas según su clase"""
    i = 0
    for (img, imclass) in zip(x_set, y_set):
        path = folder_name + ('NO/' if imclass == 0 else 'YES/')
        cv2.imwrite(path + str(i) + '.jpg', img)
        i += 1

# 3. Funciones para modelos
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusión', cmap=plt.cm.Blues):
    """Muestra la matriz de confusión"""
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    cm = np.round(cm, 2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predicha')
    plt.show()

def plot_training_history(history, name):
    """Muestra el historial de entrenamiento"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(history.epoch) + 1)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Entrenamiento')
    plt.plot(epochs_range, val_acc, label='Validación')
    plt.legend(loc="best")
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title(f'{name} - Precisión')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Validación')
    plt.legend(loc="best")
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title(f'{name} - Pérdida')

    plt.tight_layout()
    plt.show()

def create_generators(preprocess_func):
    """Crea generadores de imágenes"""
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_func
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)

    train_generator = train_datagen.flow_from_directory(
        'TRAIN_CROP/', color_mode='rgb', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', seed=RANDOM_SEED
    )
    
    validation_generator = val_datagen.flow_from_directory(
        'VAL_CROP/', color_mode='rgb', target_size=IMG_SIZE, batch_size=VAL_BATCH_SIZE, class_mode='binary', seed=RANDOM_SEED
    )

    return train_generator, validation_generator

def evaluate_model(model, preprocess_func, name, X_val_crop, y_val, X_test_crop, y_test, labels):
    """Evalúa el modelo"""
    # Preprocesar imágenes
    X_val_prep = np.array([preprocess_func(cv2.resize(img, IMG_SIZE)) for img in X_val_crop])
    X_test_prep = np.array([preprocess_func(cv2.resize(img, IMG_SIZE)) for img in X_test_crop])

    # Predicciones
    y_val_probs = model.predict(X_val_prep).flatten()
    y_val_preds = (y_val_probs > 0.5).astype(int)

    y_test_probs = model.predict(X_test_prep).flatten()
    y_test_preds = (y_test_probs > 0.5).astype(int)

    # Métricas
    val_acc = accuracy_score(y_val, y_val_preds)
    test_acc = accuracy_score(y_test, y_test_preds)
    print(f'Precisión en validación ({name}): {val_acc:.2f}')
    print(f'Precisión en test ({name}): {test_acc:.2f}')

    # Matrices de confusión
    cm_val = confusion_matrix(y_val, y_val_preds)
    cm_test = confusion_matrix(y_test, y_test_preds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in zip(
        axes,
        [cm_val, cm_test],
        [f'{name} - Matriz de Confusión (Validación)', f'{name} - Matriz de Confusión (Test)']
    ):
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title)
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([labels[0], labels[1]])
        ax.set_yticklabels([labels[0], labels[1]])
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, cm[i, j], ha="center", color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel('Etiqueta real')
        ax.set_xlabel('Etiqueta predicha')

    fig.tight_layout()
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.show()

    # ROC y AUC
    fpr, tpr, _ = roc_curve(y_val, y_val_probs)
    roc_auc = auc(fpr, tpr)
    print(f"AUC ({name}): {roc_auc:.4f}")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title(f'{name} - Curva ROC (Validación)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Classification Report
    print(f"\nInforme de Clasificación ({name} - Validación):")
    print(classification_report(y_val, y_val_preds, target_names=[labels[0], labels[1]]))

    return {"Modelo": name, "Precisión Validación": round(val_acc, 4), "Precisión Test": round(test_acc, 4)}

def train_model(model, preprocess_func, callbacks, epochs):
    """Entrena el modelo"""
    train_generator, validation_generator = create_generators(preprocess_func)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    return history

def build_vgg16():
    """Construye modelo VGG16"""
    model = Sequential(name="VGG16")
    model.add(VGG16(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)))
    model.layers[0].trainable = False
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model

def build_resnet50():
    """Construye modelo ResNet50"""
    model = Sequential(name="ResNet50")
    model.add(ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)))
    model.layers[0].trainable = False
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model

def build_mobilenetv2():
    """Construye modelo MobileNetV2"""
    model = Sequential(name="MobileNetV2")
    model.add(MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)))
    model.layers[0].trainable = False
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model

# Ejecución principal
def main():
    # 1. Preparar datos
    print("Preparando datos...")
    create_folders()
    
    # Cambia esta ruta por la ubicación de tu dataset
    input_path = 'brain_tumor_dataset/'  # Asume que tienes subcarpetas 'yes' y 'no' dentro
    split_dataset(input_path)
    
    # Cargar datos
    X_train, y_train, labels = load_data("TRAIN/")
    X_val, y_val, _ = load_data("VAL/")
    X_test, y_test, _ = load_data("TEST/")
    
    # 2. Preprocesamiento
    print("Preprocesando imágenes...")
    X_train_crop = crop_imgs(X_train)
    X_val_crop = crop_imgs(X_val)
    X_test_crop = crop_imgs(X_test)
    
    # Guardar imágenes recortadas
    save_new_images(X_train_crop, y_train, folder_name='TRAIN_CROP/')
    save_new_images(X_val_crop, y_val, folder_name='VAL_CROP/')
    save_new_images(X_test_crop, y_test, folder_name='TEST_CROP/')
    
    # 3. Entrenar modelos
    model_accuracies = []
    
    print("\nEntrenando VGG16...")
    vgg_model = build_vgg16()
    history_vgg = train_model(vgg_model, vgg_preprocess, [early_stop, reduce_lr], EPOCHS)
    plot_training_history(history_vgg, "VGG16")
    vgg_metrics = evaluate_model(vgg_model, vgg_preprocess, "VGG16", X_val_crop, y_val, X_test_crop, y_test, labels)
    model_accuracies.append(vgg_metrics)
    
    print("\nEntrenando ResNet50...")
    resnet_model = build_resnet50()
    history_resnet = train_model(resnet_model, resnet_preprocess, [early_stop, reduce_lr], EPOCHS)
    plot_training_history(history_resnet, "ResNet50")
    resnet_metrics = evaluate_model(resnet_model, resnet_preprocess, "ResNet50", X_val_crop, y_val, X_test_crop, y_test, labels)
    model_accuracies.append(resnet_metrics)
    
    print("\nEntrenando MobileNetV2...")
    mobilenet_model = build_mobilenetv2()
    history_mobilenet = train_model(mobilenet_model, mobilenet_preprocess, [early_stop, reduce_lr], EPOCHS)
    plot_training_history(history_mobilenet, "MobileNetV2")
    mobilenet_metrics = evaluate_model(mobilenet_model, mobilenet_preprocess, "MobileNetV2", X_val_crop, y_val, X_test_crop, y_test, labels)
    model_accuracies.append(mobilenet_metrics)
    
    # 4. Resultados
    results_df = pd.DataFrame(model_accuracies)
    results_df_sorted = results_df.sort_values(by="Precisión Validación", ascending=False)
    print("\nResumen de resultados:")
    display(results_df_sorted)

if __name__ == "__main__":
    main()