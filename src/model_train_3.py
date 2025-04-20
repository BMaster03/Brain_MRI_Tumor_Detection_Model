import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore 
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2 # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

class BrainTumorDetector:
    def __init__(self):
        pass
    
    # Método para encontrar el dataset
    def find_dataset_dir(self, base_path, dataset_name):
        dataset_dir = Path(base_path) / dataset_name
        if not dataset_dir.exists():
            raise ValueError(f"Dataset no encontrado en {dataset_dir}")
        return dataset_dir
    
    # Método para dividir el dataset en TRAIN, VAL, TEST
    def split_dataset(self, dataset_dir, split_ratio=0.8):
        # Definir las rutas para las carpetas
        train_dir = dataset_dir / "TRAIN"
        val_dir = dataset_dir / "VAL"
        test_dir = dataset_dir / "TEST"
        
        # Crear las carpetas si no existen
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Ahora dividimos las imágenes en las subcarpetas de entrenamiento, validación y prueba
        # Aquí debes tener imágenes en subdirectorios como 'YES' y 'NO'
        for label in ["YES", "NO"]:
            label_dir = dataset_dir / label
            image_paths = list(label_dir.glob('*.jpg'))  # Asegúrate de que sean las extensiones correctas
            train_paths, temp_paths = train_test_split(image_paths, train_size=split_ratio)
            val_paths, test_paths = train_test_split(temp_paths, test_size=0.5)

            # Mover imágenes a las carpetas correspondientes
            self.move_images(train_paths, train_dir / label)
            self.move_images(val_paths, val_dir / label)
            self.move_images(test_paths, test_dir / label)
    
    # Método auxiliar para mover imágenes
    def move_images(self, image_paths, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for image_path in image_paths:
            shutil.copy(image_path, target_dir / image_path.name)

    # Método para crear el modelo de ResNet50
    def build_resnet50(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False  # Congelamos el modelo base
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Salida binaria
        ])
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    # Método para entrenar el modelo
    def train_model(self, model, train_dir, val_dir, batch_size=32, epochs=10):
        # Usamos ImageDataGenerator para cargar imágenes
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

        # Generadores de datos
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary'  # Esto es porque tenemos 2 clases (YES/NO)
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary'
        )
        
        # Entrenar el modelo
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator
        )
        
        # Graficar precisión y pérdida durante el entrenamiento
        self.plot_training_history(history)

        return history

    # Método para graficar la precisión y la pérdida
    def plot_training_history(self, history):
        # Graficar la pérdida
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida de validación')
        plt.title('Pérdida durante el entrenamiento')
        plt.xlabel('Epochs')
        plt.ylabel('Pérdida')
        plt.legend()

        # Graficar la precisión
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Precisión de validación')
        plt.title('Precisión durante el entrenamiento')
        plt.xlabel('Epochs')
        plt.ylabel('Precisión')
        plt.legend()

        # Mostrar las gráficas
        plt.tight_layout()
        plt.show()

# Bloque principal de ejecución
if __name__ == "__main__":
    detector = BrainTumorDetector()

    # Paso 1: Encuentra el directorio del dataset
    dataset_dir = detector.find_dataset_dir(Path(__file__).parent, 'brain_tumor_dataset')
    print(f"Dataset encontrado en: {dataset_dir}")

    # Paso 2: Dividir el dataset en TRAIN, VAL, TEST
    detector.split_dataset(dataset_dir)

    # Paso 3: Crear el modelo (puedes elegir otro modelo como VGG16 o MobileNetV2)
    model = detector.build_resnet50()

    # Paso 4: Entrenar el modelo
    train_dir = dataset_dir / "TRAIN"
    val_dir = dataset_dir / "VAL"
    history = detector.train_model(model, train_dir, val_dir)

    # Mostrar las métricas de entrenamiento
    print("Entrenamiento completado!")
    print(history.history)
