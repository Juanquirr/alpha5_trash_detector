# Especificación: script de métricas común (`eval_detections.py`)

**Objetivo.** Medir con la misma regla cualquier sistema de detección (YOLO solo,
YOLO + VLM, …) contra las etiquetas reales, para poder compararlos. No entrena,
no usa GPU, solo lee ficheros.

**Entradas.**
- `--gt DIR`: etiquetas reales, un `.txt` por imagen, formato YOLO `cls cx cy w h`
  (normalizado).
- `--pred NOMBRE=DIR` (repetible): predicciones de un sistema, `.txt` por imagen,
  `cls cx cy w h conf`. Deben guardarse con confianza baja (conf=0.001), igual que
  en validación de Ultralytics, o el mAP saldrá más bajo.
- `--images DIR` (opcional): lista de imágenes evaluadas. Sin él, se usan las
  imágenes con etiqueta; las que no tienen `.txt` cuentan como vacías.
- `--names a,b,c` o `--data data.yaml` (opcional): nombres de clase.
- `--out DIR`: carpeta de salida.

**Salidas.** `metrics.json` y `metrics.csv` (una fila por sistema y clase, más
fila `all`), y una tabla por pantalla; con varios sistemas, diferencia frente al
primero.

**Método.** El mismo que Ultralytics: IoU 0,50 a 0,95 en pasos de 0,05; emparejado
uno a uno por IoU; AP por interpolación de 101 puntos sobre la envolvente de
precisión; P y R en la confianza que maximiza F1 (curva suavizada).

**Criterios de aceptación.**
1. Predicciones idénticas a las etiquetas: mAP@50 = mAP@50-95 = P = R = 1.
2. Sin predicciones: todo 0. Sin etiquetas de una clase: esa clase se omite de la media.
3. Casos con resultado calculado a mano (un falso positivo, un fallo, caja
   desplazada que pasa IoU 0,5 pero no 0,75) dan el valor esperado.
4. Una predicción de otra clase no cuenta como acierto.
5. Con varios `--pred`, las filas salen con el mismo formato y se muestra la diferencia.
6. Solo depende de `numpy` (y `pyyaml` si se usa `--data`).
7. Reproducción de la cifra antigua (mAP@50 0,692 ± 0,01) cuando el usuario la
   ejecute con las predicciones reales de YOLO26x sobre val V10: pendiente, no
   verificable aquí.

## Notas de verificación
- El mAP máximo con perfecto acierto es 0,995 (techo de la interpolación de 101
  puntos, igual que Ultralytics); los tests lo esperan así.
- Tests sintéticos en un PR aparte (`detection/eval/test_eval_detections.py`), para verificar la lógica por separado.
- No comparado todavía con Ultralytics real (no instalado aquí).
