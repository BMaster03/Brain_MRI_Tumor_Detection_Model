import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

class BrainTumorDetector:
    def __init__(self):
        pass

    def find_dataset_dir(self, dataset_name="brain_tumor_dataset"):
        base_path = Path(__file__).resolve().parent.parent
        dataset_dir = base_path / "data" / dataset_name
        if not dataset_dir.exists():
            raise ValueError(f"Dataset no encontrado en: {dataset_dir}")
        return dataset_dir

    def split_dataset(self, dataset_dir, split_ratio=0.8):
        train_dir = dataset_dir / "TRAIN"
        val_dir = dataset_dir / "VAL"
        test_dir = dataset_dir / "TEST"

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for label in ["YES", "NO"]:
            label_dir = dataset_dir / label
            if not label_dir.exists():
                raise ValueError(f"No se encontró la subcarpeta {label} en {dataset_dir}")
            image_paths = list(label_dir.glob('*.jpg')) + list(label_dir.glob('*.png')) + list(label_dir.glob('*.jpeg'))
            train_paths, temp_paths = train_test_split(image_paths, train_size=split_ratio, random_state=42)
            val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)

            self.move_images(train_paths, train_dir / label)
            self.move_images(val_paths, val_dir / label)
            self.move_images(test_paths, test_dir / label)

    def move_images(self, image_paths, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for image_path in image_paths:
            shutil.copy(image_path, target_dir / image_path.name)

    def build_resnet50(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = True  # Habilita entrenamiento

        # Solo entrena las últimas 50 capas
        for layer in base_model.layers[:-50]:
            layer.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=RMSprop(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, model, train_dir, val_dir, batch_size=64, epochs=20):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary'
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary'
        )

        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator
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

if __name__ == "__main__":
    detector = BrainTumorDetector()
    dataset_dir = detector.find_dataset_dir()
    print(f"Dataset encontrado en: {dataset_dir}")

    if not (dataset_dir / "TRAIN").exists():
        detector.split_dataset(dataset_dir)
        print("Dataset dividido en TRAIN, VAL y TEST")

    model = detector.build_resnet50()
    train_dir = dataset_dir / "TRAIN"
    val_dir = dataset_dir / "VAL"
    history = detector.train_model(model, train_dir, val_dir)

    print("Entrenamiento completado.")
