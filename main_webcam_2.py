import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model  # type: ignore

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

def detect_mri_region(frame):
    """
    Detecta la región de la MRI en el frame usando procesamiento de imágenes.
    - Retorna las coordenadas del rectángulo que encierra la MRI (o None si no se detecta).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:  # Si es un cuadrilátero
            x, y, w, h = cv2.boundingRect(approx)
            return (x, y, x + w, y + h)
    
    return None

def run_webcam_tumor_detection():
    """
    Enciende la webcam, detecta la región de la MRI, predice si hay tumor
    y muestra los resultados con un cuadro verde alrededor de la MRI.
    """
    model = load_latest_model()
    cap = cv2.VideoCapture(0)

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

        # 1. Detección del área de la MRI (cuadro verde)
        mri_coords = detect_mri_region(frame)
        if mri_coords:
            x1, y1, x2, y2 = mri_coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            roi = frame[y1:y2, x1:x2]  # Recorta la región de interés (MRI)
        else:
            roi = frame  # Si no se detecta MRI, usa el frame completo

        # 2. Preprocesamiento y predicción
        input_img = preprocess_frame(roi)
        prediction = model.predict(input_img)[0][0]

        # 3. Mostrar resultados
        label = "Si hay Tumor" if prediction > 0.5 else "No hay Tumor"
        color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)
        confidence = f"{prediction * 100:.2f}%"

        # Texto del diagnóstico
        cv2.putText(frame, f"Diagnostic: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Confidence: {confidence}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mensaje si no se detecta MRI
        if not mri_coords:
            cv2.putText(frame, "Place an MRI in front of the camera", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Brain Tumor Detection - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_tumor_detection()