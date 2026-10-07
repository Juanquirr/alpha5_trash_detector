# Métricas comunes

Ver `SPEC.md`. Uso:

```bash
# 1) Predicciones en txt YOLO con confianza baja (para YOLO solo y para YOLO + VLM)
yolo predict model=best.pt source=val/images conf=0.001 save_txt save_conf project=runs name=pred_yolo
# 2) Medir y comparar
python detection/eval/eval_detections.py --gt val/labels --images val/images \
  --pred yolo=runs/pred_yolo/labels --pred yolo_vlm=runs/pred_vlm/labels \
  --data data.yaml --out eval_out
```
