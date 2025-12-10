import torch
import argparse
import os
from ultralytics import YOLO
import csv
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
    
    # Argumentos de configuración del modelo
    # parser.add_argument('--model', type=str, default='models/yolo11x.pt',
    #                     choices=['models/yolo11n.pt', 'models/yolo11x.pt', 'models/yolo12x.pt'],
    #                     help='Modelo base de YOLO a utilizar')
    
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
    parser.add_argument('--log_dir', type=str, default='/runs/detect/trainPLOCAN',
                        help='Directorio para guardar logs CSV')
    
    # Hiperparámetros de optimización
    # Añade este argumento en buildingArguments()
    parser.add_argument('--optimizer', type=str, default='SGD',
                    choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'],
                    help='Optimizador a utilizar (especificar para usar hiperparámetros custom)')
    parser.add_argument('--lr0', type=float, default=0.0056,
                        help='Learning rate inicial')
    parser.add_argument('--lrf', type=float, default=0.01969,
                        help='Learning rate final (fracción de lr0)')
    parser.add_argument('--momentum', type=float, default=0.93412,
                        help='Momentum del optimizador')
    parser.add_argument('--weight_decay', type=float, default=0.0004,
                        help='Weight decay del optimizador')
    parser.add_argument('--warmup_epochs', type=float, default=4.09514,
                        help='Épocas de warmup')
    parser.add_argument('--warmup_momentum', type=float, default=0.30372,
                        help='Momentum durante warmup')
    
    # Hiperparámetros de pérdida
    parser.add_argument('--box', type=float, default=5.69681,
                        help='Peso de box loss')
    parser.add_argument('--cls', type=float, default=0.56072,
                        help='Peso de classification loss')
    parser.add_argument('--dfl', type=float, default=2.13634,
                        help='Peso de distribution focal loss')
    
    # Hiperparámetros de augmentation
    parser.add_argument('--hsv_h', type=float, default=0.01654,
                        help='HSV-Hue augmentation')
    parser.add_argument('--hsv_s', type=float, default=0.85488,
                        help='HSV-Saturation augmentation')
    parser.add_argument('--hsv_v', type=float, default=0.58432,
                        help='HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=0.0,
                        help='Rotación de imagen (grados)')
    parser.add_argument('--translate', type=float, default=0.08927,
                        help='Traslación de imagen (fracción)')
    parser.add_argument('--scale', type=float, default=0.39442,
                        help='Escala de imagen (ganancia)')
    parser.add_argument('--shear', type=float, default=0.0,
                        help='Shear de imagen (grados)')
    parser.add_argument('--perspective', type=float, default=0.0,
                        help='Transformación de perspectiva')
    parser.add_argument('--flipud', type=float, default=0.0,
                        help='Probabilidad de flip vertical')
    parser.add_argument('--fliplr', type=float, default=0.32289,
                        help='Probabilidad de flip horizontal')
    parser.add_argument('--bgr', type=float, default=0.0,
                        help='Probabilidad de conversión BGR')
    parser.add_argument('--mosaic', type=float, default=0.98711,
                        help='Probabilidad de mosaic augmentation')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='Probabilidad de mixup augmentation')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='Probabilidad de cutmix augmentation')
    parser.add_argument('--copy_paste', type=float, default=0.0,
                        help='Probabilidad de copy-paste augmentation')
    parser.add_argument('--close_mosaic', type=int, default=10,
                        help='Desactivar mosaic en las últimas N épocas')
    
    # Opciones adicionales
    parser.add_argument('--version_suffix', type=str, default='v3.3X',
                        help='Sufijo de versión para el archivo CSV de logs')
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
    print(f"\n🚀 Cargando modelo: YOLO11X")
    model = YOLO('models/yolo11x.pt')
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
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        # Hiperparámetros de pérdida
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        # Hiperparámetros de augmentation
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        bgr=args.bgr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        close_mosaic=args.close_mosaic,
    )
    
    # Guardar logs en CSV
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(
        args.log_dir, 
        f"training_log_{timestamp}_{args.version_suffix}.csv"
    )
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'map50', 'best_map50', 'patience_counter']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_log)
    
    print(f"\n✓ Registro guardado en: {csv_filename}")
    print(f"✓ Total de épocas entrenadas: {len(epoch_log)}")


if __name__ == '__main__':
    args = buildingArguments()
    
    # Mostrar info de CUDA si se solicita
    if args.verbose:
        print_cuda_info()
    
    # Verificar que el archivo data.yaml existe
    if not os.path.exists(args.data):
        print(f"❌ Error: No se encuentra el archivo {args.data}")
        exit(1)
    
    # Iniciar entrenamiento
    train_yolo(args)
