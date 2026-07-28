# -*- coding: utf-8 -*-
import os
import sys
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Configuración de Matplotlib/Seaborn
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Configuración de rutas relativas
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_PATH = os.path.join(BASE_PATH, "data", "bronze")
SILVER_PATH = os.path.join(BASE_PATH, "data", "silver")
GOLD_PATH = os.path.join(BASE_PATH, "data", "gold")
REPORTS_PATH = os.path.join(BASE_PATH, "reports")
DOCS_PATH = os.path.join(BASE_PATH, "docs")
DATASET = os.path.join(BRONZE_PATH, "walmart_retail_1M.csv")

# Asegurar la existencia de directorios
os.makedirs(BRONZE_PATH, exist_ok=True)
os.makedirs(SILVER_PATH, exist_ok=True)
os.makedirs(GOLD_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)
os.makedirs(DOCS_PATH, exist_ok=True)

# Verificar si el dataset de origen existe
if not os.path.exists(DATASET):
    print(f"Error: El dataset de origen no existe en {DATASET}")
    print("Por favor, coloca el archivo walmart_retail_1M.csv en 'data/bronze/' para ejecutar el pipeline.")
    sys.exit(1)

# Inicializar Spark Session localmente
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

print("Iniciando sesión de Spark...")
spark = SparkSession.builder \
    .appName("DataLakeRetail") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# ==========================================
# BRONZE LAYER
# ==========================================
print("Cargando capa Bronze...")
df = spark.read.csv(DATASET, header=True, inferSchema=True)
print(f"Número de registros cargados: {df.count()}")

# ==========================================
# SILVER LAYER
# ==========================================
print("Procesando capa Silver...")

# Conversión de fechas
df = df.withColumn("Ship Date", to_date(col("Ship Date")))

# Convertir columnas a enteros y decimales de forma segura
int_cols = ["Number of Records", "Order Quantity"]
double_cols = [
    "Customer Age", "Discount", "Product Base Margin",
    "Profit", "Sales", "Shipping Cost", "Unit Price"
]

for c in int_cols:
    df = df.withColumn(c, expr(f"try_cast(`{c}` as INT)"))

for c in double_cols:
    df = df.withColumn(c, expr(f"try_cast(`{c}` as DOUBLE)"))

df = df.withColumn("Order Date", expr("try_cast(`Order Date` as DATE)"))

# Eliminar nulos en columnas críticas
df = df.dropna(subset=["Customer Name", "Product Name", "Sales", "Order Date"])

# Rellenar nulos
df = df.fillna({
    "Discount": 0.0,
    "Shipping Cost": 0.0,
    "Profit": 0.0,
    "Product Base Margin": 0.0
})

# Reglas de validación
df = df.filter((col("Discount") >= 0) & (col("Discount") <= 0.50))
df = df.filter(col("Order Quantity") > 0)
df = df.filter(col("Sales") > 0)
df = df.filter(col("Profit") >= 0)
df = df.filter(col("Unit Price") > 0)
df = df.filter(col("Shipping Cost") >= 0)

# Columnas temporales adicionales
df = df.withColumn("Year", year("Order Date"))
df = df.withColumn("Month", month("Order Date"))
df = df.withColumn("Quarter", quarter("Order Date"))

SILVER_DATA = os.path.join(SILVER_PATH, "ventas_limpias.parquet")
df.write.mode("overwrite").parquet(SILVER_DATA)
print("Capa Silver guardada exitosamente en formato Parquet.")

# ==========================================
# GOLD LAYER
# ==========================================
print("Generando agregaciones de la capa Gold...")
silver = spark.read.parquet(SILVER_DATA)

# Ventas por categoría
ventas_categoria = silver.groupBy("Product Category").agg(
    round(sum("Sales"), 2).alias("Total Sales"),
    round(sum("Profit"), 2).alias("Total Profit"),
    sum("Order Quantity").alias("Total Quantity")
).orderBy(desc("Total Sales"))
ventas_categoria.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "ventas_categoria.parquet"))

# Ventas por subcategoría
ventas_subcategoria = silver.groupBy("Product Sub-Category").agg(
    round(sum("Sales"), 2).alias("Total Sales"),
    round(sum("Profit"), 2).alias("Total Profit"),
    sum("Order Quantity").alias("Quantity")
).orderBy(desc("Total Sales"))
ventas_subcategoria.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "ventas_subcategoria.parquet"))

# Ventas por estado
ventas_estado = silver.groupBy("State").agg(
    round(sum("Sales"), 2).alias("Sales"),
    round(sum("Profit"), 2).alias("Profit")
).orderBy(desc("Sales"))
ventas_estado.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "ventas_estado.parquet"))

# Ventas por ciudad
ventas_ciudad = silver.groupBy("City").agg(
    round(sum("Sales"), 2).alias("Sales"),
    round(sum("Profit"), 2).alias("Profit")
).orderBy(desc("Sales"))
ventas_ciudad.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "ventas_ciudad.parquet"))

# Ventas por región
ventas_region = silver.groupBy("Region").agg(
    round(sum("Sales"), 2).alias("Sales"),
    round(sum("Profit"), 2).alias("Profit")
).orderBy(desc("Sales"))
ventas_region.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "ventas_region.parquet"))

# Top 10 productos
top_productos = silver.groupBy("Product Name").agg(
    round(sum("Sales"), 2).alias("Sales"),
    sum("Order Quantity").alias("Quantity"),
    round(sum("Profit"), 2).alias("Profit")
).orderBy(desc("Sales"))
top_productos.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "top_productos.parquet"))

# Top 10 clientes
top_clientes = silver.groupBy("Customer Name").agg(
    round(sum("Sales"), 2).alias("Sales"),
    round(sum("Profit"), 2).alias("Profit")
).orderBy(desc("Sales"))
top_clientes.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "top_clientes.parquet"))

# Ventas por segmento
ventas_segmento = silver.groupBy("Customer Segment").agg(
    round(sum("Sales"), 2).alias("Sales"),
    round(sum("Profit"), 2).alias("Profit"),
    count("*").alias("Orders")
).orderBy(desc("Sales"))
ventas_segmento.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "ventas_segmento.parquet"))

# Ventas por año
ventas_anio = silver.groupBy("Year").agg(
    round(sum("Sales"), 2).alias("Sales"),
    round(sum("Profit"), 2).alias("Profit")
).orderBy("Year")
ventas_anio.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "ventas_anio.parquet"))

# Ventas por mes
ventas_mes = silver.groupBy("Year", "Month").agg(
    round(sum("Sales"), 2).alias("Sales"),
    round(sum("Profit"), 2).alias("Profit")
).orderBy("Year", "Month")
ventas_mes.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "ventas_mes.parquet"))

# Ticket promedio
ticket_promedio = silver.agg(
    round(avg("Sales"), 2).alias("Average Ticket")
)
ticket_promedio.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "ticket_promedio.parquet"))

# Resumen ejecutivo
resumen = silver.agg(
    round(sum("Sales"), 2).alias("Total Sales"),
    round(sum("Profit"), 2).alias("Total Profit"),
    sum("Order Quantity").alias("Units Sold"),
    count_distinct("Customer Name").alias("Customers"),
    count_distinct("Order ID").alias("Orders")
)
resumen.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "resumen.parquet"))
print("Capa Gold guardada exitosamente.")

# ==========================================
# GENERACIÓN DE DASHBOARD ANALÍTICO
# ==========================================
print("Generando Dashboard HTML...")

# Carga de datos para visualizaciones en Pandas
ventas_categoria_pd = ventas_categoria.toPandas()
ventas_subcategoria_pd = ventas_subcategoria.toPandas()
ventas_region_pd = ventas_region.toPandas()
ventas_estado_pd = ventas_estado.toPandas()
ventas_mes_pd = ventas_mes.toPandas()
top_productos_pd = top_productos.limit(10).toPandas()
top_clientes_pd = top_clientes.limit(10).toPandas()
ventas_segmento_pd = ventas_segmento.toPandas()
ventas_anio_pd = ventas_anio.toPandas()
resumen_pd = resumen.toPandas()
ticket_promedio_pd = ticket_promedio.toPandas()
top_ciudades = ventas_ciudad.toPandas().head(20)

total_sales = resumen_pd["Total Sales"][0]
total_profit = resumen_pd["Total Profit"][0]
units_sold = resumen_pd["Units Sold"][0]
customers = resumen_pd["Customers"][0]
orders = resumen_pd["Orders"][0]
avg_ticket = ticket_promedio_pd["Average Ticket"][0]

# KPIs Ejecutivos (Plotly)
fig_kpi = make_subplots(
    rows=2, cols=3,
    specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
           [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
)
fig_kpi.add_trace(go.Indicator(mode="number", value=total_sales, number={"valueformat": "$,.2f"}, title={"text": "Ventas Totales"}), row=1, col=1)
fig_kpi.add_trace(go.Indicator(mode="number", value=total_profit, number={"valueformat": "$,.2f"}, title={"text": "Ganancia Total"}), row=1, col=2)
fig_kpi.add_trace(go.Indicator(mode="number", value=units_sold, number={"valueformat": ",d"}, title={"text": "Unidades Vendidas"}), row=1, col=3)
fig_kpi.add_trace(go.Indicator(mode="number", value=customers, number={"valueformat": ",d"}, title={"text": "Clientes"}), row=2, col=1)
fig_kpi.add_trace(go.Indicator(mode="number", value=orders, number={"valueformat": ",d"}, title={"text": "Pedidos"}), row=2, col=2)
fig_kpi.add_trace(go.Indicator(mode="number", value=avg_ticket, number={"valueformat": "$,.2f"}, title={"text": "Ticket Promedio"}), row=2, col=3)
fig_kpi.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))

# Ventas por Región (Pie Plotly)
fig_region = px.pie(
    ventas_region_pd,
    values="Sales",
    names="Region",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_region.update_layout(margin=dict(l=20, r=20, t=30, b=20))

# Ventas Mensuales (Líneas Plotly)
fig_mes = px.line(
    ventas_mes_pd,
    x="Month",
    y="Sales",
    color="Year",
    markers=True
)
fig_mes.update_layout(margin=dict(l=20, r=20, t=30, b=20))

def fig_to_html(fig_obj):
    fig_obj.tight_layout(pad=2.0)
    buf = io.BytesIO()
    fig_obj.savefig(buf, format='png', bbox_inches='tight', dpi=90)
    import base64 as py_base64
    data = py_base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.close(fig_obj)
    return f'<img src="data:image/png;base64,{data}" style="width:100%; max-width:100%; height:auto; border-radius:4px; display: block;"/>'

# Plantilla HTML
html_content = """
<html>
<head>
    <title>Walmart Enterprise Analytics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background-color: #f0f2f5; color: #1c1e21; }
        .header { background: #0071dc; color: white; padding: 25px; text-align: center; border-bottom: 5px solid #ffc220; }
        .container { max-width: 1400px; margin: 20px auto; padding: 0 20px; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 30px; margin-top: 20px; }
        .full-width { grid-column: 1 / -1; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; }
        h2 { color: #0071dc; border-left: 5px solid #ffc220; padding-left: 15px; margin-top: 0; font-size: 1.4em; margin-bottom: 20px; }
        .plotly-wrapper { width: 100%; height: 400px; overflow: hidden; }
        .footer { text-align: center; padding: 40px; color: #65676b; font-size: 0.9em; }
        @media (max-width: 768px) {
            .dashboard-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class='header'>
        <h1>Walmart Retail - Dashboard Analítico</h1>
        <p>Pipeline de datos automatizado</p>
    </div>
    <div class='container'>
"""

# Agregar KPIs
kpi_html = pio.to_html(fig_kpi, full_html=False, include_plotlyjs=False, config={'responsive': True})
html_content += f"<div class='card full-width'><h2>1. Resumen Ejecutivo (KPIs)</h2><div class='plotly-wrapper' style='height:450px;'>{kpi_html}</div></div><div class='dashboard-grid'>"

# 2.1 Ventas por Categoría (Seaborn)
plt.figure(figsize=(9, 6))
sns.barplot(data=ventas_categoria_pd, x="Product Category", y="Total Sales", palette="viridis")
plt.title("Ventas por Categoría")
plt.ylabel("Ventas")
plt.xlabel("")
html_content += f"<div class='card'><h2>Ventas por Categoría</h2>{fig_to_html(plt.gcf())}</div>"

# 2.2 Ventas por Región (Pie Plotly)
region_html = pio.to_html(fig_region, full_html=False, include_plotlyjs=False, config={'responsive': True})
html_content += f"<div class='card'><h2>Distribución Regional de Ventas</h2><div class='plotly-wrapper'>{region_html}</div></div>"

# 2.3 Ventas por Segmento (Seaborn)
plt.figure(figsize=(9, 6))
sns.barplot(data=ventas_segmento_pd, x="Customer Segment", y="Sales", palette="Set2")
plt.title("Ventas por Segmento de Clientes")
plt.ylabel("Ventas")
plt.xlabel("")
html_content += f"<div class='card'><h2>Ventas por Segmento</h2>{fig_to_html(plt.gcf())}</div>"

# 2.4 Ventas por Subcategoría (Seaborn)
plt.figure(figsize=(10, 7))
sns.barplot(data=ventas_subcategoria_pd, x="Product Sub-Category", y="Total Sales", color="#0071dc")
plt.xticks(rotation=45, ha='right')
plt.title("Ventas por Subcategoría de Producto")
plt.ylabel("Ventas")
plt.xlabel("")
html_content += f"<div class='card'><h2>Ventas por Subcategoría</h2>{fig_to_html(plt.gcf())}</div>"

# 2.5 Ventas por Estado (Seaborn)
plt.figure(figsize=(14, 7))
sns.barplot(data=ventas_estado_pd.head(20), x="State", y="Sales", palette="Blues_r")
plt.xticks(rotation=45, ha='right')
plt.title("Top 20 Estados con Mayores Ventas")
plt.ylabel("Ventas")
plt.xlabel("")
html_content += f"<div class='card full-width'><h2>Ventas por Estado (Top 20)</h2>{fig_to_html(plt.gcf())}</div>"

# 2.6 Top 10 Productos (Seaborn)
plt.figure(figsize=(12, 8))
sns.barplot(data=top_productos_pd, y="Product Name", x="Sales", palette="rocket")
plt.title("Top 10 Productos más Vendidos")
plt.xlabel("Ventas")
plt.ylabel("")
html_content += f"<div class='card'><h2>Top 10 Productos</h2>{fig_to_html(plt.gcf())}</div>"

# 2.7 Top 10 Clientes (Seaborn)
plt.figure(figsize=(10, 8))
sns.barplot(data=top_clientes_pd, y="Customer Name", x="Sales", color="#4CAF50")
plt.title("Top 10 Clientes por Ventas")
plt.xlabel("Ventas")
plt.ylabel("")
html_content += f"<div class='card'><h2>Top 10 Clientes</h2>{fig_to_html(plt.gcf())}</div>"

# 2.8 Ventas por Año (Matplotlib)
plt.figure(figsize=(9, 6))
plt.plot(ventas_anio_pd["Year"].astype(str), ventas_anio_pd["Sales"], marker='s', color='#ffc220', linewidth=3, markersize=10)
plt.title("Evolución Anual de Ventas")
plt.ylabel("Ventas")
plt.grid(True, alpha=0.3)
html_content += f"<div class='card'><h2>Ventas por Año</h2>{fig_to_html(plt.gcf())}</div>"

# 2.9 Ventas por Mes (Plotly)
mes_html = pio.to_html(fig_mes, full_html=False, include_plotlyjs=False, config={'responsive': True})
html_content += f"<div class='card full-width'><h2>Estacionalidad Mensual de Ventas</h2><div class='plotly-wrapper' style='height:450px;'>{mes_html}</div></div>"

# 2.10 Ventas por Ciudad (Seaborn)
plt.figure(figsize=(14, 7))
sns.barplot(data=top_ciudades, x="City", y="Sales", palette="magma")
plt.xticks(rotation=45, ha='right')
plt.title("Top 20 Ciudades con Mayores Ventas")
plt.ylabel("Ventas")
plt.xlabel("")
html_content += f"<div class='card full-width'><h2>Ventas por Ciudad (Top 20)</h2>{fig_to_html(plt.gcf())}</div>"

html_content += """
    </div>
    <div class='footer'>
        <p>Walmart Retail | Dashboard Corporativo | Generado Automáticamente por el Pipeline Batch</p>
    </div>
</body>
</html>
"""

OUTPUT_HTML = os.path.join(REPORTS_PATH, "dashboard_report.html")
OUTPUT_DOCS = os.path.join(DOCS_PATH, "index.html")

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

with open(OUTPUT_DOCS, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Dashboard generado exitosamente en: {OUTPUT_HTML} y {OUTPUT_DOCS}")
spark.stop()
print("Sesión de Spark finalizada.")
