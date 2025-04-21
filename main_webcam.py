import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore 

class TumorDetector:
    def __init__(self):
        self.model = self.load_model()
        
    def load_model(self):
        models_dir = Path("trained_model_parameters")
        if not models_dir.exists():
            raise FileNotFoundError("No se encontraron modelos entrenados.")
            
        model_path = max(models_dir.glob("*.keras"), key=os.path.getctime)
        print(f"Cargando modelo: {model_path.name}")
        return load_model(model_path)
    
    def preprocess_image(self, image):
        img = cv2.resize(image, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    
    def detect_mri_region(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    
    def analyze_image(self, image_path):
        # Leer la imagen
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")
        
        # Detección de región MRI (opcional)
        mri_coords = self.detect_mri_region(image)
        if mri_coords:
            x1, y1, x2, y2 = mri_coords
            roi = image[y1:y2, x1:x2]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            roi = image
            print("Advertencia: No se detectó claramente la región MRI, analizando toda la imagen")
        
        # Preprocesamiento y predicción
        processed_img = self.preprocess_image(roi)
        prediction = self.model.predict(processed_img, verbose=0)[0][0]
        confidence = prediction * 100
        
        # Mostrar resultados
        label = "TUMOR DETECTADO" if prediction > 0.65 else "Sano"
        color = (0, 0, 255) if prediction > 0.65 else (0, 255, 0)
        
        # Añadir texto a la imagen
        cv2.putText(image, f"Resultado: {label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(image, f"Confianza: {confidence:.1f}%", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar la imagen
        cv2.imshow("Resultado del Análisis", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return {
            "prediction": "Tumor" if prediction > 0.65 else "Sano",
            "confidence": f"{confidence:.1f}%",
            "image_path": image_path
        }

if __name__ == "__main__":
    detector = TumorDetector()
    
    # Ejemplo de cómo usar (reemplaza con la ruta de tu imagen)
    image_path = "/Users/tunombreusuario/Downloads/tumor_mri.jpg"  # Cambia esto por la ruta real de tu imagen
    
    try:
        results = detector.analyze_image(image_path)
        print("\nResultados del análisis:")
        print(f"Imagen analizada: {results['image_path']}")
        print(f"Diagnóstico: {results['prediction']}")
        print(f"Confianza: {results['confidence']}")
    except Exception as e:
        print(f"Error: {str(e)}")