"""
Main file for the Brain Tumor Detection Project.

This script serves as the entry point for different project functionalities:
- Model training (different versions)
- Image processing (from webcam or internal storage)
"""

import sys
from src.model_train_10 import BrainTumorDetector as Model10
from src.model_train_11 import BrainTumorDetector as Model11
from src.model_train_12 import BrainTumorDetector as Model12
from src.process_image_intern import TumorDetector_1 as ImageIntern
from src.process_image_webcam_2 import TumorDetector as WebcamDetector

def run(program_to_run):
    """
    Execute the selected program module.
    
    Args:
        program_to_run (str): Command specifying which module to run
    """
    try:
        if program_to_run == 'model_12':
            Model12()
        elif program_to_run == 'image_intern':
            detect_1 = ImageIntern()
            detect_1.run()  
        elif program_to_run == 'image_webcam':
            detector = WebcamDetector()
            detector.run()
        else:
            print("\n Error: Comando no reconocido")
            
    except Exception as e:
        print(f"\n Error al ejecutar {program_to_run}: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    if sys.argv[1] in ['model_12', 'image_webcam', 'image_intern']:
        run(sys.argv[1])
    else:
        print("\n Error: Comando no válido")
        sys.exit(1)
        