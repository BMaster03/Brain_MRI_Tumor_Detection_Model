import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore 

class TumorDetector_2:
    def __init__(self):
        self.model = self.load_model()
        self.prediction_history = []
        
    def load_model(self):
        models_dir = Path("trained_model_parameters")
        if not models_dir.exists():
            raise FileNotFoundError("No se encontraron modelos entrenados.")
            
        model_path = max(models_dir.glob("*.keras"), key=os.path.getctime)
        print(f"Cargando modelo: {model_path.name}")
        return load_model(model_path)
    
    def preprocess_frame(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    
    def detect_mri_region(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 30, 150)
        
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) >= 4 or (len(approx) > 4 and cv2.isContourConvex(approx)):
                x, y, w, h = cv2.boundingRect(approx)
                return (x, y, x+w, y+h)
        return None
    
    def smooth_prediction(self, prediction):
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > 5:
            self.prediction_history.pop(0)
        return np.mean(self.prediction_history)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error al abrir la cámara")
            return

        print("\n Instrucciones:")
        print("1. Coloca una imagen de MRI frente a la cámara")
        print("2. Presiona 'q' para salir \n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar frame")
                break
                
            # Detección de región MRI
            mri_coords = self.detect_mri_region(frame)
            if mri_coords:
                x1, y1, x2, y2 = mri_coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                roi = frame[y1:y2, x1:x2]
            else:
                roi = frame
                
            # Predicción
            processed_img = self.preprocess_frame(roi)
            prediction = self.model.predict(processed_img, verbose=0)[0][0]
            smoothed_pred = self.smooth_prediction(prediction)
            
            # Visualización
            label = "Tumor detectado" if smoothed_pred > 0.65 else "Sano"
            color = (0, 0, 255) if smoothed_pred > 0.65 else (0, 255, 0)
            confidence = f"{smoothed_pred*100:.1f}%"
            
            cv2.putText(frame, f"Estado: {label}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            """
            cv2.putText(frame, f"Confianza: {confidence}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            """
            if not mri_coords:
                cv2.putText(frame, "Favor de enfocar una imagen MRI", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            cv2.imshow("Detector de Tumores Cerebrales", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_2 = TumorDetector_2()
    detect_2.run()