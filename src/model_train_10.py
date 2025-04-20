import os
import shutil
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # type: ignore

class BrainTumorDetector:
    def __init__(self, dataset_name="brain_tumor_dataset", split_ratio=0.8, batch_size=32, epochs=30):
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.epochs = epochs

        # Organizar y dividir el dataset
        self.organize_and_split_dataset()

        print("Dataset organizado y dividido correctamente.")

        # Construir y entrenar el modelo
        model = self.build_mobilenetv2()
        base_path = Path(__file__).resolve().parent.parent
        dataset_dir = base_path / "data" / self.dataset_name
        train_dir = dataset_dir / "Train"
        val_dir = dataset_dir / "Validation"
        history = self.train_model(model, train_dir, val_dir)

        print("Entrenamiento completado. Modelo guardado en 'trained_model_parameters'.")

        # Evaluar el modelo
        test_dir = dataset_dir / "Test"
        test_loss, test_acc = self.evaluate_model(model, test_dir)
        print(f"Resultado de evaluación en conjunto Test: Pérdida = {test_loss:.4f}, Precisión = {test_acc:.4f}")

    def organize_and_split_dataset(self):
        base_path = Path(__file__).resolve().parent.parent
        dataset_dir = base_path / "data" / self.dataset_name

        if not dataset_dir.exists():
            raise ValueError(f"Dataset no encontrado en: {dataset_dir}")

        train_dir = dataset_dir / "Train"
        val_dir = dataset_dir / "Validation"
        test_dir = dataset_dir / "Test"

        for folder in [train_dir, val_dir, test_dir]:
            for label in ["YES", "NO"]:
                os.makedirs(folder / label, exist_ok=True)

        def has_images(path):
            return path.exists() and any(path.glob("*.[jp][pn]g"))

        if all(has_images(train_dir / label) for label in ["YES", "NO"]) and \
           all(has_images(val_dir / label) for label in ["YES", "NO"]) and \
           all(has_images(test_dir / label) for label in ["YES", "NO"]):
            print("Las carpetas Train, Validation y Test ya existen y contienen imágenes.")
            return

        for label in ["YES", "NO"]:
            label_dir = dataset_dir / label
            if not label_dir.exists():
                raise ValueError(f"No se encontró la subcarpeta {label} en {dataset_dir}")

            image_paths = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.jpeg")) + list(label_dir.glob("*.png"))
            train_paths, temp_paths = train_test_split(image_paths, train_size=self.split_ratio, random_state=42)
            val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)

            self._copy_files(train_paths, train_dir / label)
            self._copy_files(val_paths, val_dir / label)
            self._copy_files(test_paths, test_dir / label)

        print(f"Dataset organizado en: {train_dir}, {val_dir}, {test_dir}")

    def _copy_files(self, files, target_dir):
        for file in files:
            dest = target_dir / file.name
            if not dest.exists():
                shutil.copy(file, dest)

    def build_mobilenetv2(self):
        # Modelo secuencial utilizando MobileNetV2 preentrenado
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        model = tf.keras.Sequential([ 
            base_model,  # MobileNetV2 como modelo base
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),  # Capa densa 
            tf.keras.layers.Dense(32, activation='relu'),  # Capa densa adicional
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Capa de salida para clasificación binaria
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def train_model(self, model, train_dir, val_dir):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.25,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),  # Redimensionamos las imágenes a (224, 224)
            batch_size=self.batch_size,
            class_mode='binary'
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(224, 224),  # Redimensionamos las imágenes a (224, 224)
            batch_size=self.batch_size,
            class_mode='binary'
        )

        models_dir = Path("trained_model_parameters")
        models_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint = ModelCheckpoint(
            filepath=models_dir / f'best_model_{ts}.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        earlystop = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )

        history = model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=val_generator,
            callbacks=[checkpoint, earlystop, reduce_lr]
        )

        self.plot_training_history(history)
        return history

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Pérdida entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida validación')
        plt.title('Pérdida')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Precisión validación')
        plt.title('Precisión')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def evaluate_model(self, model, test_dir):
        models_dir = Path("trained_model_parameters")
        if not models_dir.exists():
            raise FileNotFoundError("No se encontró el directorio 'trained_model_parameters'.")

        saved_models = list(models_dir.glob("*.keras"))
        if not saved_models:
            raise FileNotFoundError("No se encontraron modelos .keras en 'trained_model_parameters'.")

        latest_model = max(saved_models, key=os.path.getctime)
        model.load_weights(latest_model)

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),  # Redimensionamos las imágenes a (224, 224)
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

        test_loss, test_acc = model.evaluate(test_generator)
        return test_loss, test_acc


if __name__ == "__main__":
    Model_10 = BrainTumorDetector()  
