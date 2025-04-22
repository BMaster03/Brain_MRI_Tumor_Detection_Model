import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model  # type: ignore

class TumorDetector_1:
    def __init__(self):
        self.model = self.load_model()
        self.test_images = []
        
    def load_model(self):
        models_dir = Path("trained_model_parameters")
        if not models_dir.exists():
            raise FileNotFoundError("No se encontraron modelos entrenados.")
            
        model_path = max(models_dir.glob("*.keras"), key=os.path.getctime)
        print(f"\nCargando modelo: {model_path.name}")
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
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo leer la imagen: {image_path}")
            
            height, width = image.shape[:2]
            font_scale = min(width, height) / 1000
            thickness = max(1, int(min(width, height) / 500))
            line_type = cv2.LINE_AA
            
            mri_coords = self.detect_mri_region(image)
            if mri_coords:
                x1, y1, x2, y2 = mri_coords
                roi = image[y1:y2, x1:x2]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            else:
                roi = image
                print("\n Advertencia: No se detectó claramente la región MRI, analizando toda la imagen")
            
            processed_img = self.preprocess_image(roi)
            prediction = self.model.predict(processed_img, verbose=0)[0][0]
            confidence = prediction * 100
            
            label = "Detect Tumor" if prediction > 0.65 else "Healty"
            color = (0, 0, 255) if prediction > 0.65 else (0, 255, 0)
            
            text_y1 = max(30, int(height * 0.05))
            text_y2 = max(60, int(height * 0.10))
            
            cv2.putText(image, f"Resultado: {label}", 
                       (int(width * 0.05), text_y1), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale * 1.2, 
                       color, 
                       thickness * 2, 
                       line_type) 
            
            cv2.imshow("Resultado del Análisis", image)
            cv2.waitKey(3000)  # Muestra la imagen por 3 segundos
            cv2.destroyAllWindows()
            
            return {
                "prediction": "Tumor" if prediction > 0.65 else "Sano",
                "image_path": image_path
            }
            
        except Exception as e:
            print(f"\n Error al analizar la imagen: {str(e)}")
            return None

    def load_test_images(self):
        test_dir = Path("image/Test")
        if not test_dir.exists():
            raise FileNotFoundError(f"No se encontró la carpeta de imágenes de prueba: {test_dir}")
        
        self.test_images = sorted(test_dir.glob("*.*"))
        if not self.test_images:
            raise FileNotFoundError(f"No se encontraron imágenes en: {test_dir}")
        
    def show_image_menu(self):
        print("Images: ")
        for i, img_path in enumerate(self.test_images, 1):
            print(f"{i}. {img_path.name}")

    def run(self):
        try:
            self.load_test_images()
            
            while True:
                self.show_image_menu()
                selection = input("\n Seleccione un numero de imagen: ").upper()
                
                if selection == 'q':
                    print("\n Saliendo del programa...")
                    break
                    
                elif selection == 'l':
                    continue
                    
                try:
                    img_num = int(selection)
                    if 1 <= img_num <= len(self.test_images):
                        image_path = str(self.test_images[img_num-1])
                        print(f"\n Analizando imagen: {image_path}")
                        
                        results = self.analyze_image(image_path)
                        if results:
                            print("\nResultados del análisis:")
                            print(f"Imagen analizada: {results['image_path']}")
                            print(f"Diagnóstico: {results['prediction']}")
                    else:
                        print(f"\n Error: Por favor ingrese un número entre 1 y {len(self.test_images)}")
                except ValueError:
                    print("\n Error: Entrada inválido")
                    
        except Exception as e:
            print(f"\n Error inicial: {str(e)}")

if __name__ == "__main__":
    try:
        detect_1 = TumorDetector_1()
        detect_1.run()
    except KeyboardInterrupt:
        print("\n Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n Error fatal: {str(e)}")