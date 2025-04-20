"""
    Main file for the project.

    This file is used to run the BrainTumorDetector class for dataset handling, training, and evaluation.
"""

import sys
from src.model_train_10 import BrainTumorDetector
from src.model_train_11 import BrainTumorDetector
from src.model_train_12 import BrainTumorDetector

def run(program_to_run):
    """
    Run the project based on model selection.
    """
    if program_to_run == 'model_10':
        Model_10 = BrainTumorDetector()  
    elif program_to_run == 'model_11':
        Model_11 = BrainTumorDetector()
    elif program_to_run == 'model_12':
        Model_12 = BrainTumorDetector()
    elif program_to_run == 'model_13':
        Model_13 = BrainTumorDetector()
    else:
        print("Modelo no reconocido")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['model_10', 'model_11', 'model_12', 'model_13']:
        run(sys.argv[1])
    else:
        print("Comando inválido")
        