# Walmart Batch ETL Pipeline & Dashboard

Este repositorio contiene un pipeline Batch automatizado profesional utilizando **PySpark**, **GitHub Actions** y **GitHub Pages**.

La arquitectura está diseñada bajo los principios de un Data Lake moderno con capas de datos organizadas y un reporte final interactivo publicado automáticamente.

## Arquitectura del Pipeline

```text
                Nuevo Dataset
                      │
                      ▼
        data/bronze/walmart_retail_1M.csv
                      │
                      │ (Al hacer push o diariamente a las 12:00 AM UTC)
                      ▼
             GitHub Actions (Batch ETL)
                      │
          ┌───────────┼─────────────┐
          ▼           ▼             ▼
      Capa Bronze   Capa Silver   Capa Gold (Agregaciones)
      (CSV Raw)    (Parquet)      (Parquet por dimensiones)
          │
          ▼
    Generación de Reporte HTML (docs/index.html & reports/dashboard_report.html)
          │
          ▼
     GitHub Pages
          │
          ▼
 Dashboard corporativo actualizado automáticamente
```

## Estructura del Proyecto

* **`.github/workflows/batch.yml`**: Configuración del pipeline CI/CD en GitHub Actions.
* **`data/bronze/`**: Carpeta destinada al dataset crudo inicial (`walmart_retail_1M.csv`).
* **`data/silver/`**: Capa de datos limpios en formato Parquet (`ventas_limpias.parquet`).
* **`data/gold/`**: Capa de negocio con datasets consolidados y agregados en formato Parquet.
* **`reports/`**: Resguardo del reporte HTML generado.
* **`docs/`**: Carpeta utilizada por GitHub Pages para servir el reporte interactivo.
* **`src/pipeline.py`**: Script de procesamiento ETL en Python que automatiza toda la lógica de PySpark.
* **`requirements.txt`**: Librerías necesarias para el entorno de ejecución.

---

## Ejecución Local

### Prerrequisitos

1. **Java Development Kit (JDK) 11 o superior** (necesario para correr Spark localmente).
2. **Python 3.10 o superior**.

### Pasos

1. Clona este repositorio.
2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
3. Coloca tu archivo de datos en `data/bronze/walmart_retail_1M.csv`.
4. Ejecuta el pipeline:
   ```bash
   python src/pipeline.py
   ```
5. Abre `docs/index.html` en tu navegador para ver el dashboard interactivo generado.

---

## Despliegue en GitHub Pages

Para ver tu dashboard en vivo a través de una URL pública (`https://<tu-usuario>.github.io/<tu-repositorio>/`):

1. Sube tu proyecto a un repositorio de GitHub.
2. Asegúrate de que el pipeline de GitHub Actions corra al menos una vez (esto generará la carpeta `docs/` en tu repositorio).
3. Entra a la configuración de tu repositorio en GitHub (**Settings**).
4. En la barra lateral izquierda, haz clic en **Pages**.
5. En la sección **Build and deployment**, bajo **Source**, selecciona **Deploy from a branch**.
6. En **Branch**, selecciona tu rama principal (`main` o `master`) y cambia la carpeta `/ (root)` a `/docs`.
7. Haz clic en **Save**.

¡Listo! A partir de ese momento, cada vez que se ejecute el pipeline (diariamente a las 12 AM o tras un nuevo push), el dashboard se actualizará automáticamente en tu URL de GitHub Pages.
