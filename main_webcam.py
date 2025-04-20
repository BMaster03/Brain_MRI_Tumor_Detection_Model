import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore


def load_latest_model():
    """
    Carga el modelo entrenado más reciente desde 'trained_model_parameters'.
    """
    models_dir = Path("trained_model_parameters")
    if not models_dir.exists():
        raise FileNotFoundError("No se encontró la carpeta 'trained_model_parameters'.")

    model_paths = list(models_dir.glob("*.keras"))
    if not model_paths:
        raise FileNotFoundError("No se encontraron modelos entrenados (.keras) en la carpeta.")

    latest_model_path = max(model_paths, key=os.path.getctime)
    print(f"Modelo cargado desde: {latest_model_path}")
    return load_model(latest_model_path)


def preprocess_frame(frame):
    """
    Preprocesa un frame de la webcam para ser compatible con el modelo.
    """
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def run_webcam_tumor_detection():
    """
    Enciende la webcam, predice en tiempo real si hay tumor en las imágenes mostradas,
    y despliega los resultados sobre la imagen en pantalla.
    """
    model = load_latest_model()

    # Abre la cámara
    cap = cv2.VideoCapture(0)  # Usa 0 o 1 dependiendo de tu cámara
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    print("Webcam activa. Muestra una imagen médica frente a la cámara.")
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar imagen.")
            break

        input_img = preprocess_frame(frame)
        prediction = model.predict(input_img)[0][0]

        label = "Tumor Cerebral" if prediction > 0.5 else "No hay tumor"
        color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

        # Mostrar el texto en pantalla
        cv2.putText(frame, f"Diagnóstico: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Detección de Tumor Cerebral - Webcam", frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_tumor_detection()
