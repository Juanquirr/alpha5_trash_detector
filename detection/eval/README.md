# Common metrics

See `SPEC.md`. Usage:

```bash
# 1) Predictions as YOLO txt with a low confidence threshold (for YOLO alone and for YOLO + VLM)
yolo predict model=best.pt source=val/images conf=0.001 save_txt save_conf project=runs name=pred_yolo
# 2) Measure and compare
python detection/eval/eval_detections.py --gt val/labels --images val/images \
  --pred yolo=runs/pred_yolo/labels --pred yolo_vlm=runs/pred_vlm/labels \
  --data data.yaml --out eval_out
```
