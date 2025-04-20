"""
This script is used to train and evaluate different CNN models for brain MRI tumor detection.
"""
import itertools
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm # type: ignore
import imutils # type: ignore

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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import RMSprop # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

warnings.filterwarnings("ignore", category=UserWarning, module='keras.src.trainers.data_adapters.py_dataset_adapter')

class BrainMRITumorDetector:
    def __init__(self):
        # Configuration
        self.RANDOM_SEED = 123
        self.IMG_SIZE = (224, 224)
        self.EPOCHS = 100
        self.BATCH_SIZE = 32
        self.VAL_BATCH_SIZE = 16
        
        # Path setup
        self.current_dir = Path(__file__).parent
        self.models_dir = self.current_dir.parent / 'trained_model_parameters'
        self.models_dir.mkdir(exist_ok=True)
        
        # Data generators
        self.train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            brightness_range=[0.5, 1.5],
            horizontal_flip=True,
            vertical_flip=True
        )
        
        self.val_datagen = ImageDataGenerator()
        
        # Callbacks
        self.early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        self.reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

    def create_folders(self):
        """
            Create necessary folder structure
        """
        folders = ['TRAIN/YES', 'TRAIN/NO', 'TEST/YES', 'TEST/NO', 'VAL/YES', 'VAL/NO',
                   'TRAIN_CROP/YES', 'TRAIN_CROP/NO', 'TEST_CROP/YES', 'TEST_CROP/NO', 'VAL_CROP/YES', 'VAL_CROP/NO']
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def split_dataset(self, input_path):
        """
            Split dataset into train/val/test
        """
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

    def load_data(self, dir_path):
        """
            Load images and labels from directory
        """
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

    def plot_samples(self, X, y, labels_dict, n=50):
        """
            Display a gallery of images
        """
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

    def crop_imgs(self, set_name):
        """
            Crop brain region from each image
        """
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

    def save_new_images(self, x_set, y_set, folder_name):
        """
            Save images in folders according to their class
        """
        i = 0
        for (img, imclass) in zip(x_set, y_set):
            path = folder_name + ('NO/' if imclass == 0 else 'YES/')
            cv2.imwrite(path + str(i) + '.jpg', img)
            i += 1

    def create_generators(self, preprocess_func):
        """
            Create image generators
        """
        train_datagen = self.train_datagen
        train_datagen.preprocessing_function = preprocess_func
        
        val_datagen = self.val_datagen
        val_datagen.preprocessing_function = preprocess_func

        train_generator = train_datagen.flow_from_directory(
            'TRAIN_CROP/', 
            color_mode='rgb', 
            target_size=self.IMG_SIZE, 
            batch_size=self.BATCH_SIZE, 
            class_mode='binary', 
            seed=self.RANDOM_SEED
        )
        
        validation_generator = val_datagen.flow_from_directory(
            'VAL_CROP/', 
            color_mode='rgb', 
            target_size=self.IMG_SIZE, 
            batch_size=self.VAL_BATCH_SIZE, 
            class_mode='binary', 
            seed=self.RANDOM_SEED
        )

        return train_generator, validation_generator

    def build_resnet50(self):
        """
            Build ResNet50 model
        """
        model = Sequential(name="ResNet50")
        model.add(ResNet50(weights="imagenet", include_top=False, input_shape=self.IMG_SIZE + (3,)))
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

    def train_model(self, model, preprocess_func, model_name):
        """
            Train the model
        """
        train_generator, validation_generator = self.create_generators(preprocess_func)
        
        dt = datetime.now()
        ts = datetime.timestamp(dt)
        
        model_checkpoint = ModelCheckpoint(
            filepath=self.models_dir / f'best_{model_name.lower()}_{ts}.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        history = model.fit(
            train_generator,
            epochs=self.EPOCHS,
            validation_data=validation_generator,
            callbacks=[self.early_stop, self.reduce_lr, model_checkpoint]
        )

        return history

    def plot_training_history(self, history, name):
        """
            Plot training history
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, len(history.epoch) + 1)

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training')
        plt.plot(epochs_range, val_acc, label='Validation')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'{name} - Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training')
        plt.plot(epochs_range, val_loss, label='Validation')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{name} - Loss')

        plt.tight_layout()
        plt.show()

    def evaluate_model(self, model, preprocess_func, name, X_val_crop, y_val, X_test_crop, y_test, labels):
        """Evaluate the model"""
        # Preprocess images
        X_val_prep = np.array([preprocess_func(cv2.resize(img, self.IMG_SIZE)) for img in X_val_crop])
        X_test_prep = np.array([preprocess_func(cv2.resize(img, self.IMG_SIZE)) for img in X_test_crop])

        # Predictions
        y_val_probs = model.predict(X_val_prep).flatten()
        y_val_preds = (y_val_probs > 0.5).astype(int)

        y_test_probs = model.predict(X_test_prep).flatten()
        y_test_preds = (y_test_probs > 0.5).astype(int)

        # Metrics
        val_acc = accuracy_score(y_val, y_val_preds)
        test_acc = accuracy_score(y_test, y_test_preds)
        print(f'Validation Accuracy ({name}): {val_acc:.2f}')
        print(f'Test Accuracy ({name}): {test_acc:.2f}')

        # Confusion matrices
        cm_val = confusion_matrix(y_val, y_val_preds)
        cm_test = confusion_matrix(y_test, y_test_preds)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, cm, title in zip(
            axes,
            [cm_val, cm_test],
            [f'{name} - Confusion Matrix (Validation)', f'{name} - Confusion Matrix (Test)']
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
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')

        fig.tight_layout()
        plt.colorbar(im, ax=axes.ravel().tolist())
        plt.show()

        # ROC and AUC
        fpr, tpr, _ = roc_curve(y_val, y_val_probs)
        roc_auc = auc(fpr, tpr)
        print(f"AUC ({name}): {roc_auc:.4f}")

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} - ROC Curve (Validation)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

        # Classification Report
        print(f"\nClassification Report ({name} - Validation):")
        print(classification_report(y_val, y_val_preds, target_names=[labels[0], labels[1]]))

        return {"Model": name, "Validation Accuracy": round(val_acc, 4), "Test Accuracy": round(test_acc, 4)}

    def run(self, input_path='brain_tumor_dataset/'):
        """Run the complete pipeline"""
        # 1. Prepare data
        print("Preparing data...")
        self.create_folders()
        self.split_dataset(input_path)
        
        # Load data
        X_train, y_train, labels = self.load_data("TRAIN/")
        X_val, y_val, _ = self.load_data("VAL/")
        X_test, y_test, _ = self.load_data("TEST/")
        
        # 2. Preprocessing
        print("Preprocessing images...")
        X_train_crop = self.crop_imgs(X_train)
        X_val_crop = self.crop_imgs(X_val)
        X_test_crop = self.crop_imgs(X_test)
        
        # Save cropped images
        self.save_new_images(X_train_crop, y_train, folder_name='TRAIN_CROP/')
        self.save_new_images(X_val_crop, y_val, folder_name='VAL_CROP/')
        self.save_new_images(X_test_crop, y_test, folder_name='TEST_CROP/')
        
        # 3. Train and evaluate models
        model_accuracies = []
        
        print("\nTraining ResNet50...")
        resnet_model = self.build_resnet50()
        history_resnet = self.train_model(resnet_model, resnet_preprocess, "ResNet50")
        self.plot_training_history(history_resnet, "ResNet50")
        resnet_metrics = self.evaluate_model(resnet_model, resnet_preprocess, "ResNet50", X_val_crop, y_val, X_test_crop, y_test, labels)
        model_accuracies.append(resnet_metrics)
        
        # 4. Results
        results_df = pd.DataFrame(model_accuracies)
        results_df_sorted = results_df.sort_values(by="Validation Accuracy", ascending=False)
        print("\nResults summary:")
        print(results_df_sorted)

        return results_df_sorted

if __name__ == "__main__":
    detector = BrainMRITumorDetector()
    results = detector.run()