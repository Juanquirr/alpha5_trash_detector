import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

PURPLE = "#8000ff"
BAR_FORMAT = "{desc:<28} |{bar}| {percentage:6.2f}% ({n_fmt}/{total_fmt}) [{elapsed}<{remaining}, {rate_fmt}]"


class SuperResolutionPreprocessor:
    """Handles different SR methods"""
    
    def __init__(self, method="clahe"):
        self.method = method
        self.sr_model = None
        
        if method == "opencv_dnn":
            self._init_opencv_sr()
        elif method == "real_esrgan":
            self._init_realesrgan()
    
    def _init_opencv_sr(self):
        """Initialize OpenCV DNN SR (requires model file)"""
        try:
            self.sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
            # Download models from: https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres
            model_path = "ESPCN_x2.pb"  # or EDSR_x2.pb, FSRCNN_x2.pb
            if Path(model_path).exists():
                self.sr_model.readModel(model_path)
                self.sr_model.setModel('espcn', 2)
                print(f"✓ OpenCV SR loaded: {model_path}")
            else:
                print(f"⚠️  Model not found: {model_path}")
                self.sr_model = None
        except Exception as e:
            print(f"❌ OpenCV SR error: {e}")
            self.sr_model = None
    
    def _init_realesrgan(self):
        """Initialize Real-ESRGAN"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self.sr_model = RealESRGANer(
                scale=2,
                model_path='RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True
            )
            print("✓ Real-ESRGAN loaded")
        except ImportError:
            print("❌ Real-ESRGAN not installed: pip install basicsr realesrgan")
            self.sr_model = None
    
    def process(self, image):
        """Apply SR preprocessing"""
        if self.method == "clahe":
            return self._apply_clahe(image)
        elif self.method == "unsharp":
            return self._apply_unsharp_mask(image)
        elif self.method == "opencv_dnn" and self.sr_model:
            return self.sr_model.upsample(image)
        elif self.method == "real_esrgan" and self.sr_model:
            output, _ = self.sr_model.enhance(image, outscale=2)
            return output
        else:
            return image
    
    @staticmethod
    def _apply_clahe(image):
        """CLAHE: Contrast Limited Adaptive Histogram Equalization"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def _apply_unsharp_mask(image, kernel_size=5, strength=1.5):
        """Unsharp masking for edge enhancement"""
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 1.0)
        sharpened = cv2.addWeighted(image, strength, blurred, -(strength - 1), 0)
        return sharpened


def process_image_sr(
    model: YOLO,
    img_path: Path,
    out_dir: Path,
    conf: float,
    iou: float,
    device: str,
    imgsz: int,
    sr_processor: SuperResolutionPreprocessor,
    pbar: tqdm = None
):
    """Process image with SR preprocessing"""
    img = cv2.imread(str(img_path))
    if img is None:
        tqdm.write(f"⚠️  Skipping: {img_path}")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if pbar:
        pbar.set_description_str(f"SR {img_path.name}")
    
    # Apply SR
    img_processed = sr_processor.process(img)
    
    # Detect
    results = model.predict(
        img_processed,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        device=device
    )
    
    detections = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy()
            s = float(box.conf[0].cpu().numpy())
            c = int(box.cls[0].cpu().numpy())
            detections.append((b, s, c))
    
    tqdm.write(f"  ✓ {img_path.name}: {len(detections)} detections")
    
    # Annotate (on processed image)
    annotator = Annotator(img_processed, line_width=2, example=model.names)
    
    for box, score, cls_id in detections:
        name = model.names.get(cls_id, str(cls_id))
        annotator.box_label(
            [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            f"{name} {score:.2f}",
            color=colors(cls_id, bgr=True)
        )
    
    img_out = annotator.result()
    output_path = out_dir / f"sr_{img_path.stem}.jpg"
    cv2.imwrite(str(output_path), img_out)


def iter_images(source: Path, recursive: bool = False):
    """Iterate over images"""
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    
    if source.is_file():
        return [source] if source.suffix.lower() in supported else []
    
    if recursive:
        return sorted([p for ext in supported for p in source.rglob(f"*{ext}")])
    else:
        return sorted([p for ext in supported for p in source.glob(f"*{ext}")])


def build_args():
    """Build argument parser"""
    p = argparse.ArgumentParser(
        description="Super Resolution preprocessing for YOLO detection"
    )
    
    p.add_argument("source", type=str, help="Input image or directory")
    p.add_argument("model", type=str, help="Path to YOLO model (.pt)")
    p.add_argument("--out_dir", type=str, default="sr_results", help="Output directory")
    p.add_argument("--sr_method", type=str, default="clahe",
                   choices=["clahe", "unsharp", "opencv_dnn", "real_esrgan"],
                   help="SR method (default: clahe)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Input size")
    p.add_argument("--device", type=str, default="cuda:0", help="Device")
    p.add_argument("--recursive", action="store_true", help="Recursive search")
    
    return p.parse_args()


def main():
    """Main entry point"""
    args = build_args()
    
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    
    # Initialize SR
    sr_processor = SuperResolutionPreprocessor(method=args.sr_method)
    
    model = YOLO(args.model)
    images = iter_images(source, args.recursive)
    
    if not images:
        raise SystemExit(f"❌ No images found in: {source}")
    
    print(f"✓ Found {len(images)} image(s)")
    print(f"✓ SR method: {args.sr_method}")
    print(f"✓ Confidence: {args.conf}")
    
    pbar = tqdm(
        total=len(images),
        desc="Images processed",
        unit="img",
        bar_format=BAR_FORMAT,
        colour=PURPLE,
        dynamic_ncols=True
    )
    
    try:
        for img_path in images:
            process_image_sr(
                model=model,
                img_path=img_path,
                out_dir=out_dir,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                imgsz=args.imgsz,
                sr_processor=sr_processor,
                pbar=pbar
            )
            pbar.update(1)
    finally:
        pbar.close()
    
    print(f"\n✓ SR inference complete! Results in {out_dir}")


if __name__ == '__main__':
    main()
