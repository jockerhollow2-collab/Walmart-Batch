# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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
from pyspark.sql.window import Window

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

# Top 5 Productos Mensuales
ventana_gold = Window.partitionBy("Year", "Month").orderBy(col("Cantidad Vendida").desc())
top_5_mensual_gold = (silver
    .groupBy("Year", "Month", "Product Name")
    .agg(sum("Order Quantity").alias("Cantidad Vendida"))
    .withColumn("Ranking", row_number().over(ventana_gold))
    .filter(col("Ranking") <= 5)
    .orderBy("Year", "Month", "Ranking")
)
top_5_mensual_gold.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "top_5_productos_mensuales.parquet"))

# Top 10 Productos General (necesario para el dashboard)
top_productos = silver.groupBy("Product Name").agg(
    round(sum("Sales"), 2).alias("Sales"),
    sum("Order Quantity").alias("Quantity"),
    round(sum("Profit"), 2).alias("Profit")
).orderBy(desc("Sales"))
top_productos.write.mode("overwrite").parquet(os.path.join(GOLD_PATH, "top_productos.parquet"))

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
ventas_region_pd = ventas_region.toPandas()
ventas_estado_pd = ventas_estado.toPandas()
ventas_mes_pd = ventas_mes.toPandas()
ventas_mes_pd["Year"] = ventas_mes_pd["Year"].astype(str) # Casteo clave para evitar problemas en Plotly
top_productos_pd = top_productos.limit(10).toPandas()
ventas_segmento_pd = ventas_segmento.toPandas()
resumen_pd = resumen.toPandas()
top_5_mensual_pd = top_5_mensual_gold.toPandas()

# Cargar ticket promedio
ticket_promedio_pd = ticket_promedio.toPandas()
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

# 2.1 Ventas por Categoría (Plotly)
fig_cat = px.bar(
    ventas_categoria_pd,
    x="Product Category",
    y="Total Sales",
    color="Product Category",
    title="Ventas por Categoría",
    labels={"Total Sales": "Ventas (USD)", "Product Category": "Categoría"},
    template="plotly_white"
)

# 2.2 Ventas por Subcategoría (Plotly)
fig_sub = px.bar(
    ventas_subcategoria.toPandas(),
    x="Product Sub-Category",
    y="Total Sales",
    title="Ventas por Subcategoría",
    labels={"Total Sales": "Ventas Totales (USD)", "Product Sub-Category": "Subcategoría"},
    template="plotly_white"
)
fig_sub.update_layout(xaxis_tickangle=-45)

# 2.3 Ventas por Estado (Plotly)
fig_est = px.bar(
    ventas_estado_pd,
    x="State",
    y="Sales",
    title="Ventas por Estado",
    labels={"Sales": "Ventas Totales (USD)", "State": "Estado"},
    template="plotly_white"
)
fig_est.update_layout(xaxis_tickangle=-90)

# 2.4 Ventas por Ciudad (Plotly)
top_ciudades = ventas_ciudad.toPandas().head(20)
fig_ciu = px.bar(
    top_ciudades,
    x="City",
    y="Sales",
    color="Sales",
    title="Top 20 Ciudades por Ventas",
    labels={"Sales": "Ventas (USD)", "City": "Ciudad"},
    color_continuous_scale="magma",
    template="plotly_white"
)
fig_ciu.update_layout(xaxis_tickangle=-45)

# 2.5 Ventas por Región (Pie Plotly)
fig_region = px.pie(
    ventas_region_pd,
    values="Sales",
    names="Region",
    title="Ventas por Región"
)

# 2.6 Ventas por Segmento (Plotly)
fig_seg = px.bar(
    ventas_segmento_pd,
    x="Customer Segment",
    y="Sales",
    color="Customer Segment",
    title="Ventas por Segmento",
    labels={"Sales": "Ventas (USD)", "Customer Segment": "Segmento"},
    template="plotly_white",
    color_discrete_sequence=px.colors.qualitative.Set2
)

# 2.7 Top 5 Productos Mensuales por Año
def plot_top5_monthly_products(df_pandas, year_val):
    df_filtered = df_pandas[df_pandas['Year'] == year_val].copy()
    meses = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
             7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}
    df_filtered["Mes"] = df_filtered["Month"].map(meses)
    df_filtered["Top"] = df_filtered["Ranking"].apply(lambda x: f"Top {x}")
    fig = px.bar(
        df_filtered, x="Mes", y="Cantidad Vendida", color="Top", barmode="group",
        category_orders={"Mes": list(meses.values()), "Top": [f"Top {i}" for i in range(1, 6)]},
        title=f"Top 5 Productos por Mes ({year_val})",
        labels={"Cantidad Vendida": "Cantidad Vendida (Unidades)", "Mes": "Mes"},
        template="plotly_white", height=500
    )
    return fig

fig_2022 = plot_top5_monthly_products(top_5_mensual_pd, 2022)
fig_2023 = plot_top5_monthly_products(top_5_mensual_pd, 2023)
fig_2024 = plot_top5_monthly_products(top_5_mensual_pd, 2024)
fig_2025 = plot_top5_monthly_products(top_5_mensual_pd, 2025)

# 2.8 Ventas por Año (Línea Plotly)
ventas_anio_pd = ventas_anio.toPandas()
ventas_anio_pd["Year"] = ventas_anio_pd["Year"].astype(str)
fig_anio = px.line(
    ventas_anio_pd,
    x="Year",
    y="Sales",
    markers=True,
    title="Tendencia de Ventas por Año",
    labels={"Sales": "Ventas (USD)", "Year": "Año"},
    template="plotly_white"
)
fig_anio.update_traces(
    line_color="#0071dc",
    marker=dict(size=10),
    hovertemplate="<b>Año: %{x}</b><br>Ventas: $%{y:,.2f}<extra></extra>"
)

# 2.9 Ventas Mensuales (Línea Plotly)
fig_mes = px.line(
    ventas_mes_pd,
    x="Month",
    y="Sales",
    color="Year",
    markers=True,
    title="Ventas Mensuales",
    labels={"Sales": "Ventas (USD)", "Month": "Mes del Año", "Year": "Año"}
)
fig_mes.update_xaxes(dtick=1)

# Plantilla HTML con diseño corporativo
html_content = f"""
<html>
<head>
    <title>Walmart Retail Analytics | Executive Dashboard</title>
    <meta charset='utf-8'>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --walmart-blue: #0071dc;
            --walmart-yellow: #ffc220;
            --bg-gray: #f0f2f5;
            --text-dark: #202124;
            --card-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }}
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-gray);
            color: var(--text-dark);
            margin: 0;
            padding: 0;
        }}
        .navbar {{
            background-color: var(--walmart-blue);
            color: white;
            padding: 20px 40px;
            text-align: left;
            border-bottom: 5px solid var(--walmart-yellow);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .container {{
            padding: 30px;
            display: flex;
            flex-direction: column;
            gap: 30px;
            max-width: 1400px;
            margin: auto;
        }}
        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: var(--card-shadow);
            margin-bottom: 10px;
        }}
        .card-header {{
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--walmart-blue);
            margin-bottom: 20px;
            border-left: 6px solid var(--walmart-yellow);
            padding-left: 15px;
        }}
        .footer {{
            text-align: center;
            padding: 40px;
            color: #777;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class='navbar'>
        <div>
            <h1 style='margin:0;'>Walmart Data Lake Analytics</h1>
            <span style='opacity: 0.9;'>Retail Executive Insights Portal</span>
        </div>
    </div>

    <div class='container'>
        <!-- KPIs -->
        <div class='card'>
            <div class='card-header'>Indicadores Clave de Desempeño (KPIs)</div>
            {pio.to_html(fig_kpi, full_html=False, include_plotlyjs=False)}
        </div>

        <!-- Tendencias Temporales -->
        <div class='card'>
            <div class='card-header'>Tendencia de Ventas Anual</div>
            {pio.to_html(fig_anio, full_html=False, include_plotlyjs=False)}
        </div>

        <div class='card'>
            <div class='card-header'>Ventas Mensuales (Comparativa)</div>
            {pio.to_html(fig_mes, full_html=False, include_plotlyjs=False)}
        </div>

        <!-- Desglose por Producto -->
        <div class='card'>
            <div class='card-header'>Distribución por Categoría</div>
            {pio.to_html(fig_cat, full_html=False, include_plotlyjs=False)}
        </div>

        <div class='card'>
            <div class='card-header'>Ventas por Subcategoría</div>
            {pio.to_html(fig_sub, full_html=False, include_plotlyjs=False)}
        </div>

        <!-- Geografía -->
        <div class='card'>
            <div class='card-header'>Rendimiento por Estado</div>
            {pio.to_html(fig_est, full_html=False, include_plotlyjs=False)}
        </div>

        <div class='card'>
            <div class='card-header'>Top 20 Ciudades</div>
            {pio.to_html(fig_ciu, full_html=False, include_plotlyjs=False)}
        </div>

        <div class='card'>
            <div class='card-header'>Participación por Región</div>
            {pio.to_html(fig_region, full_html=False, include_plotlyjs=False)}
        </div>

        <!-- Clientes y Rankings -->
        <div class='card'>
            <div class='card-header'>Ventas por Segmento de Cliente</div>
            {pio.to_html(fig_seg, full_html=False, include_plotlyjs=False)}
        </div>

        <div class='card'>
            <div class='card-header'>Top 5 Productos Mensuales - Histórico</div>
            <div style='display: flex; flex-direction: column; gap: 40px;'>
                <div><h3>2022</h3>{pio.to_html(fig_2022, full_html=False, include_plotlyjs=False)}</div>
                <div><h3>2023</h3>{pio.to_html(fig_2023, full_html=False, include_plotlyjs=False)}</div>
                <div><h3>2024</h3>{pio.to_html(fig_2024, full_html=False, include_plotlyjs=False)}</div>
                <div><h3>2025</h3>{pio.to_html(fig_2025, full_html=False, include_plotlyjs=False)}</div>
            </div>
        </div>
    </div>

    <div class='footer'>
        &copy; 2026 Walmart Retail Analytics Data Lake - Reporte Automatizado
    </div>
</body>
</html>
"""

# Guardar los reportes en localizaciones correspondientes
OUTPUT_HTML = os.path.join(REPORTS_PATH, "dashboard_report.html")
OUTPUT_DOCS = os.path.join(DOCS_PATH, "index.html")

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

with open(OUTPUT_DOCS, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Dashboard generado exitosamente en: {OUTPUT_HTML} y {OUTPUT_DOCS}")
spark.stop()
print("Sesión de Spark finalizada.")
