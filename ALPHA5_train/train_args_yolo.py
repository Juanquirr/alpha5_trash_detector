import torch
import argparse
import os
from ultralytics import YOLO
import csv
import yaml
from datetime import datetime


def buildingArguments() -> argparse.Namespace:
    """
    Builds the command-line argument parser for YOLO training.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments from the user input.
    """
    parser = argparse.ArgumentParser(
        description="Script para entrenar modelos YOLO para detección de basura PLOCAN"
    )
    
    # Argumentos obligatorios
    parser.add_argument('data', type=str, 
                        help='Ruta al archivo data.yaml del dataset')
    
    # Argumentos de entrenamiento
    parser.add_argument('--epochs', type=int, default=300,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch', type=int, default=-1,
                        help='Tamaño del batch (-1 para AutoBatch)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Tamaño de imagen para entrenamiento')
    parser.add_argument('--workers', type=int, default=8,
                        help='Número de workers para DataLoader')
    parser.add_argument('--patience', type=int, default=15,
                        help='Épocas de paciencia para early stopping')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device para entrenamiento (cuda, cpu, o cuda:0)')
    
    # Directorios
    parser.add_argument('--project', type=str, 
                        default='/ultralytics/plocania/runs/detect/trainPLOCAN',
                        help='Directorio del proyecto para guardar resultados')
    parser.add_argument('--name', type=str, default=None,
                        help='Nombre del experimento (subfolder en project)')

    
    # Hiperparámetros de optimización
    parser.add_argument('--optimizer', type=str, default='AdamW',
                    choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'],
                    help='Optimizador a utilizar (especificar para usar hiperparámetros custom)')
    
    parser.add_argument('--hyperparams', type=str,
                    help='Ruta al YAML de hiperparámetros')
    
    # Opciones adicionales
    parser.add_argument('--verbose', action='store_true',
                        help='Mostrar información detallada de CUDA')
    
    return parser.parse_args()


def print_cuda_info():
    """Imprime información sobre PyTorch y CUDA."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    print(f"CUDA version en PyTorch: {torch.version.cuda}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print("-" * 50)


def train_yolo(args):
    """
    Función principal de entrenamiento de YOLO.
    
    Args:
        args: Argumentos parseados desde la línea de comandos
    """
    # Variables globales para el callback
    global best_map50, patience_counter, epoch_log
    
    best_map50 = 0.0
    patience_counter = 0
    epoch_log = []
    
    def on_fit_epoch_end(trainer):
        """
        Callback para imprimir y registrar mAP50 de cada época y la paciencia.
        """
        global best_map50, patience_counter
        current_map50 = trainer.metrics['metrics/mAP50(B)']
        
        if current_map50 > best_map50:
            best_map50 = current_map50
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Epoch {trainer.epoch}: mAP50 = {current_map50:.4f}, "
              f"Best = {best_map50:.4f}, Paciencia = {patience_counter}/{args.patience}")
        
        epoch_log.append({
            'epoch': trainer.epoch,
            'map50': current_map50,
            'best_map50': best_map50,
            'patience_counter': patience_counter
        })
    
    # Cargar modelo
    print(f"\n🚀 Cargando modelo: YOLO12X")
    model = YOLO("yolo12x.pt")
    # model = YOLO("/ultralytics/plocania/yolo12x.pt")
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # Configurar device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device != args.device:
        print(f"⚠️  CUDA no disponible, usando CPU en lugar de {args.device}")
    
    # Iniciar entrenamiento
    print(f"\n📊 Iniciando entrenamiento con {args.epochs} épocas...")
    print(f"📁 Dataset: {args.data}")
    print(f"💾 Resultados en: {args.project}")
    print("-" * 50)

    hparams = {}
    if args.hyperparams is not None:
        with open(args.hyperparams, "r") as f:
            hparams = yaml.safe_load(f)
        hparams['close_mosaic'] = int(hparams['close_mosaic'])
        hparams = hparams
    
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        imgsz=args.imgsz,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        # Hiperparámetros de optimización
        optimizer=args.optimizer,
        **hparams
    )

    print(f"✓ Total de épocas entrenadas: {len(epoch_log)}")

if __name__ == '__main__':
    args = buildingArguments()
    
    if args.verbose:
        print_cuda_info()
    
    if not os.path.exists(args.data):
        print(f"❌ Error: No se encuentra el archivo {args.data}")
        exit(1)
    
    train_yolo(args)
