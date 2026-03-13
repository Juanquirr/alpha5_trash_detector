# Datagen Orchestrator

![CLI](https://github.com/user-attachments/assets/3cf29f55-6962-4726-8bf0-31a70f9444ff)


**Datagen Orchestrator** es una colección de utilidades y pipelines para generar conjuntos de datos sintéticos utilizando varios servicios de inteligencia artificial. El proyecto incluye una interfaz de línea de comandos (CLI) que te guía paso a paso para generar imágenes, anotaciones y subir los resultados a un dataset en Datagen.

## Instalación

1. **Clona el repositorio**

   ```bash
   git clone <repo-url>
   cd datagen_orchestrator
   ```

2. **Instala las dependencias de Python**

   ```bash
   pip install -r requirements.txt
   ```

   Se recomienda Python 3.12 o superior.

## Ejecutar la CLI

Ejecuta la CLI interactiva con:

```bash
python3 -m apps.cli.main
```

En la primera ejecución, la CLI buscará un archivo `config.json`. Si no existe, se te pedirá que ingreses las URLs de los servicios necesarios y tu clave de API de Datagen. Las claves de configuración requeridas son:

* `api_key`
* `Datagen backend`
* `Diffusers service`
* `Segmentators service`
* `Autodistill service`
* `ComfyUI`

Los valores se guardan en `config.json` para usos posteriores.

### Flujo de trabajo

1. **ID del Dataset** – Se te pedirá que ingreses el ID del dataset objetivo. La CLI validará el ID usando la API de Datagen.
2. **Selección del Pipeline** – Elige qué pipeline deseas ejecutar. Las opciones actualmente disponibles son:

   * **Pipeline 1**
   * **Pipeline 2**
   * **Pipeline 3**
3. **Entrada en CSV** – Proporciona la ruta a un archivo CSV que contenga los datos de condicionamiento requeridos por el pipeline seleccionado.
4. La CLI procesará cada fila del CSV, generará la imagen y las anotaciones correspondientes, y luego las subirá al dataset indicado.

## Estructura del Repositorio

* `apps/cli` – Implementación de la interfaz de línea de comandos interactiva.
* `core/pipelines` – Pipelines de generación de datasets utilizados por la CLI.
* `datagen_sdk` – Cliente mínimo para comunicarse con la API de Datagen.
* `example_csv/` – Archivos CSV de ejemplo que muestran el formato esperado de columnas.

## Docker

Se incluye un `Dockerfile` para ejecutar el proyecto de forma contenedorizada. Puedes construir y ejecutar el contenedor con:

```bash
docker build -t datagen-orchestrator .
docker run --rm -it datagen-orchestrator
```


