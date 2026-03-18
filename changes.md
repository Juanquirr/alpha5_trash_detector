  Archivos nuevos

  - core/water_detector.py - Nuevo módulo de detección de agua robusto

  Archivos reescritos (todo en inglés)

  - run_test_models.py - Script de comparación de modelos
  - run_fill.py - Pipeline principal de generación
  - config/prompts.csv - Prompts mejorados
  - core/dependencies/.../flux_canny_inpainter.py
  - core/dependencies/.../flux_redux_inpainter.py
  - core/dependencies/.../flux_kontext_inpainter.py
  - core/dependencies/.../flux_local_image_inpainter.py

  ---
  Bug fixes aplicados

  1. Siempre las mismas imágenes (Canny)
  - Antes: sorted(...)[: max_images] -> siempre las primeras N alfabéticamente
  - Ahora: random.shuffle(all_images) antes de seleccionar. Flag --no-shuffle para desactivar.

  2. No detecta el mar / objetos en el cielo
  - Antes: El filtro HSV (70-145 hue) matcheaba tanto agua como cielo azul. Sin rechazo espacial.
  - Ahora: Nuevo módulo core/water_detector.py con:
    - create_water_mask(): máscara binaria pixel a pixel con HSV + exclusión de bordes (Canny dilatado) + varianza local de textura + exclusión del 20%
  superior (cielo) + limpieza morfológica + eliminación de regiones pequeñas
    - find_water_positions(): muestreo aleatorio dentro de la máscara de agua con separación mínima garantizada y verificación de que TODO el footprint del
  objeto cae en agua

  3. Redux no usaba las referencias
  - Antes: references_dir="references" (no existe) + matching fuzzy roto ("plastic bottle" no matcheaba "plastic_bottle")
  - Ahora: references_dir="inputs/references" + mapa explícito _CLASS_FOLDER_MAP que mapea cada class_id directamente al nombre de carpeta correcto (e.g., 1
   -> "glass")

  4. Objetos demasiado grandes / sin perspectiva
  - Antes: 100-380px en imagen de 1024px (hasta 37% del ancho)
  - Ahora: 40-200px (máximo ~20% del ancho). Más realista para la distancia de las cámaras.

  5. Prompts incorrectos para la perspectiva
  - Antes: "seen from above", "overhead view", "aerial perspective", "top-down view"
  - Ahora: Sin indicadores de perspectiva (el modelo infiere del contexto). Añadidos descriptores de integración: "wet", "partially submerged",
  "waterlogged", "matching the lighting"

  ---
  Nueva funcionalidad: Crop-based inpainting (tu idea del zoom)

  Implementado como el modo por defecto. Workflow:

  1. create_water_mask() genera máscara global de agua
  2. find_water_positions() encuentra posiciones dentro del agua
  3. Para cada objeto:
    - Se calcula un crop de 320-640px centrado en la posición (4x el tamaño del objeto)
    - Se extrae el crop de la imagen
    - Se crea la máscara localmente dentro del crop
    - Se inpainta solo el crop (el modelo ve agua de cerca, no la panorámica)
    - Se pega el resultado de vuelta en la imagen completa
    - Las anotaciones YOLO se calculan en coordenadas de imagen completa

  Flag --no-crop para volver al modo antiguo (full-image inpainting).

  ---
  Cómo probar en la máquina objetivo

  # Dentro del contenedor Docker

  # 1. Test rápido de water detection (sin GPU, solo CPU)
  python -c "
  import cv2, numpy as np
  from PIL import Image
  from core.water_detector import create_water_mask, find_water_positions

  SIZES = {0:(50,100,25,50), 1:(50,100,25,50), 2:(40,70,35,65), 3:(80,150,60,120),
           4:(50,100,40,80), 5:(60,110,40,80), 6:(120,200,90,160), 7:(40,80,30,60)}

  import glob
  for p in sorted(glob.glob('inputs/*.jpeg') + glob.glob('inputs/*.jpg'))[:10]:
      img = Image.open(p).convert('RGB')
      w,h = img.size; s = min(1024/max(w,h),1.0)
      img = img.resize((max(16,round(w*s/16)*16), max(16,round(h*s/16)*16)), Image.LANCZOS)
      mask = create_water_mask(np.array(img))
      pos = find_water_positions(mask, 3, SIZES)
      print(f'{p.split(\"/\")[-1][:45]:45s} water={mask.mean()/255:.0%}  positions={len(pos)}')
      # Save water mask for visual inspection
      Image.fromarray(mask).save(f'outputs_test/{p.split(\"/\")[-1]}_water.png')
  "

  # 2. Test un modelo con crop (usa GPU)
  python run_test_models.py --model canny --max-images 3 --num-instances 2

  # 3. Test un modelo sin crop (comparar)
  python run_test_models.py --model canny --max-images 3 --num-instances 2 --no-crop

  # 4. Test Redux (ahora SÍ usará las referencias)
  python run_test_models.py --model redux --max-images 3

  # 5. Test los tres modelos
  python run_test_models.py --model all --max-images 3

  # 6. Generación completa con Fill
  python run_fill.py --num-instances 3

  El test #1 no necesita GPU y te permite verificar visualmente que la detección de agua funciona antes de gastar tiempo con los modelos. Los archivos
  _water_mask.png que se guardan te muestran exactamente qué zonas detectó como agua (blanco = agua, negro = no agua).

  Output de debug por imagen

  Cada imagen genera ahora:
  - {stem}_result.png - Imagen con objetos insertados
  - {stem}_debug.png - Imagen con bounding boxes dibujados
  - {stem}.txt - Anotaciones YOLO
  - {stem}_water_mask.png - NUEVO: Máscara de agua para verificar la detección