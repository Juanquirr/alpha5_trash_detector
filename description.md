  Canny: Basado en bordes/estructura visual

  - Entrada: Extrae un mapa de bordes Canny de la imagen existente
  - Condicionamiento: El modelo ve los bordes del resto de la imagen → respeta la estructura (agua, orillas)
  - Generación: Produce la imagen completa condicionada por esos bordes
  - Compositing: Pega solo la zona enmascarada sobre la original
  - Ventaja: Respeta bien la coherencia estructural del entorno
  - Requiere: Ninguna imagen de referencia adicional

  Imagen original
    ↓
  Extrae bordes (Canny)
    ↓
  Borra bordes en zona de máscara
    ↓
  FLUX.1-Canny-dev genera condicionada por bordes
    ↓
  Pega resultado en zona de máscara

  Redux: Basado en apariencia de referencia real

  - Entrada: Imágenes de referencia de objetos reales (ej: botellas de plástico reales)
  - Condicionamiento: Extrae embeddings visuales de la referencia → sustituye el texto
  - Generación: FLUX Fill usa esos embeddings para generar algo que se vea como la referencia
  - Ventaja: Genera objetos más realistas (porque está entrenado en la apariencia real)
  - Requiere: Carpeta references/{class_name}/ con fotos reales de cada tipo de objeto

  Foto de botella real
    ↓
  Redux extrae embeddings visuales
    ↓
  FLUX.1-Fill-dev genera usando esos embeddings
    ↓
  Resultado tiene apariencia realista como la referencia

  Resumen comparativo:

  ┌───────────────────┬──────────────────────────────┬─────────────────────────────────────┐
  │      Aspecto      │            Canny             │                Redux                │
  ├───────────────────┼──────────────────────────────┼─────────────────────────────────────┤
  │ Fuente de control │ Bordes de la imagen          │ Fotos de referencia real            │
  ├───────────────────┼──────────────────────────────┼─────────────────────────────────────┤
  │ Lo que genera     │ Respeta estructura del fondo │ Respeta apariencia del objeto       │
  ├───────────────────┼──────────────────────────────┼─────────────────────────────────────┤
  │ Requiere          │ Solo la imagen a rellenar    │ Carpeta references/ poblada         │
  ├───────────────────┼──────────────────────────────┼─────────────────────────────────────┤
  │ Realismo          │ Moderado (síntesis genérica) │ Alto (copia características reales) │
  ├───────────────────┼──────────────────────────────┼─────────────────────────────────────┤
  │ Velocidad         │ Más rápido (1 modelo)        │ Más lento (2 modelos)               │
  └───────────────────┴──────────────────────────────┴─────────────────────────────────────┘