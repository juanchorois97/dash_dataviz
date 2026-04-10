# 🎓 Dashboard ODS 4 — Tasa de Finalización de Educación Primaria Femenina

> **Indicador WDI:** `SE.PRM.CMPT.FE.ZS` · **Fuente:** Banco Mundial · **Periodo:** 2000–2022  
> **Autores:** Juan Marín · Samuel Chamorro — Científicos de Datos

---

## 📋 Tabla de Contenidos

1. [Contexto del Proyecto](#-contexto-del-proyecto)
2. [¿Qué mide el indicador?](#-qué-mide-el-indicador)
3. [Estructura del Dashboard](#-estructura-del-dashboard)
4. [Pipeline de Datos](#-pipeline-de-datos)
5. [Requisitos del Sistema](#-requisitos-del-sistema)
6. [Instalación](#-instalación)
7. [Ejecución](#-ejecución)
8. [Arquitectura del Código](#-arquitectura-del-código)
9. [Solución de Problemas](#-solución-de-problemas)
10. [Referencias](#-referencias)

---

## 🌍 Contexto del Proyecto

Este dashboard es una aplicación interactiva construida con **Python y Dash** que realiza un **análisis exploratorio de datos (EDA)** completo sobre la tasa de finalización de educación primaria femenina a nivel mundial entre los años **2000 y 2022**.

El análisis está enmarcado en el cumplimiento del **Objetivo de Desarrollo Sostenible 4 (ODS 4)** de las Naciones Unidas: *"Garantizar una educación inclusiva, equitativa y de calidad y promover oportunidades de aprendizaje durante toda la vida para todos"*, específicamente la **meta 4.1**, que busca asegurar que todas las niñas completen la enseñanza primaria y secundaria.

La aplicación replica y extiende una app original desarrollada en R/Shiny, trasladando toda la lógica de visualización y análisis estadístico al ecosistema Python, con acceso directo a la API del Banco Mundial para obtener datos actualizados y reproducibles.

### Relevancia del tema

La brecha educativa de género es uno de los factores más persistentes de desigualdad global. La no finalización de la educación primaria incrementa los riesgos de matrimonio temprano, embarazo adolescente y exclusión del mercado laboral formal. Este indicador permite monitorear el progreso de los países hacia la paridad educativa y detectar rezagos que requieren intervención de política pública.

---

## 📐 ¿Qué mide el indicador?

| Campo | Detalle |
|-------|---------|
| **Código WDI** | `SE.PRM.CMPT.FE.ZS` |
| **Definición** | Porcentaje de niñas que completan el último grado de educación primaria respecto a la población femenina en edad oficial |
| **Valores posibles** | 0% – 115%+ (valores > 100% son metodológicamente válidos) |
| **Cobertura** | ~217 países individuales del Banco Mundial |
| **Observaciones limpias** | **2,992** filas (país × año, sin valores nulos) |
| **Fuente** | World Development Indicators (WDI), Banco Mundial |

> **Nota sobre valores > 100%:** Ocurren cuando el numerador incluye graduadas de cohortes anteriores. El dashboard trabaja con una variable `tasa_fin_cap = clip(tasa_fin, 0, 100)` para análisis gráfico, conservando el valor original para diagnósticos.

---

## 🗂 Estructura del Dashboard

El dashboard está organizado en **7 módulos de análisis** accesibles desde un menú lateral fijo:

### 1. ℹ️ Introducción
Panel de bienvenida con 6 sub-pestañas:

| Sub-pestaña | Contenido |
|-------------|-----------|
| **Introducción** | Definición del indicador, fuente y estado de carga de datos |
| **Justificación** | Importancia del estudio y principales desafíos del dataset |
| **Objetivos** | Objetivo general y 5 objetivos específicos del análisis |
| **Marco Teórico** | Resumen del dataset + 4 pilares teóricos (capital humano, convergencia, series de panel, equidad) |
| **Hipótesis** | 5 hipótesis de investigación formales (H1–H5) |
| **Metodología** | Fuente de datos, pasos de procesamiento y herramientas utilizadas |

---

### 2. 📊 Análisis Univariado
Análisis de la distribución y estructura temporal de la variable principal:

| Visualización | Descripción |
|---------------|-------------|
| **Prueba ADF** | Augmented Dickey-Fuller sobre la serie de medianas anuales; incluye estadístico, p-valor y valores críticos al 1%, 5% y 10% |
| **ACF** | Función de Autocorrelación (hasta 15 rezagos) con bandas de confianza ±1.96/√n |
| **PACF** | Función de Autocorrelación Parcial para identificar el orden AR |
| **Histograma + KDE** | Distribución de `tasa_fin_cap` con curva de densidad kernel superpuesta |
| **Boxplot** | Diagrama de caja con detección de outliers |
| **Estadísticas descriptivas** | Media, mediana, desviación estándar, asimetría de Pearson, P1 y P99 |
| **Tabla > 100%** | Top 30 observaciones con sobrenotificación, ordenadas por exceso |

---

### 3. 📈 Análisis Bivariado
Exploración de relaciones entre la tasa de finalización, el tiempo y los países:

| Visualización | Descripción |
|---------------|-------------|
| **Lollipop + IQR** | Mediana e IQR por país y quinquenio (top N países por cobertura temporal, N ajustable 4–15) |
| **% en el Techo** | Proporción de años en que cada país alcanzó exactamente 100% |
| **Scatter + LOESS** | Nube de puntos país-año con suavizado LOESS (frac=0.55) |
| **Mediana anual + Banda IQR** | Evolución 2000–2022 con banda P25–P75 |
| **Media vs. Mediana** | Comparación anual con área sombreada de brecha |
| **Ribbon multicapa** | Bandas P10–P25–P75–P90 + mediana simultáneamente |
| **Mapa de calor** | `tasa_fin_cap` por país × año (top N países ajustable 10–50) |
| **Pendientes pre-techo** | Top N países con mayor mejora anual (regresión lineal antes de alcanzar 100%) |
| **Trayectorias individuales** | Evolución temporal de países seleccionados vía dropdown |
| **Prueba de hipótesis** | Test comparativo entre dos períodos: Shapiro-Wilk → Levene → t-Student o Mann-Whitney |

---

### 4. 🌍 Mapa Mundial
Tres mapas coropléticos interactivos con `locationmode="ISO-3"`:

| Mapa | Descripción |
|------|-------------|
| **Promedio histórico** | Media 2000–2022 por país; tonos azul oscuro = tasas altas |
| **Evolución animada** | Mapa animado año a año con botón ▶ Play (2000–2022) |
| **Cambio absoluto** | Diferencia entre último año disponible y año 2000; verde = mejora, rojo = retroceso |

---

### 5. 🏁 Conclusiones
Síntesis de los hallazgos en 5 bloques temáticos:
- Hallazgos del análisis univariado
- Prueba ADF, ACF y PACF
- Hallazgos del análisis bivariado
- Hallazgos del análisis multivariado
- Limitaciones y pasos a seguir

---

### 6. 📋 Explorador de Datos
Tabla interactiva del dataset limpio (`df_2`) con:
- Filtro por país (dropdown)
- Filtro por rango de años (slider 2000–2022)
- Ordenamiento y búsqueda por columna
- Resaltado en color de filas con `tasa_fin > 100`

---

### 7. 📚 Referencias
Bibliografía en formato APA 7.ª edición organizada en 4 secciones:
- Fuentes de datos (Banco Mundial, WDI)
- Metodología estadística (ADF, ACF/PACF, ARIMA)
- Marco teórico (Becker, Barro & Lee, ODS, UNESCO, UNICEF)
- Herramientas computacionales (Plotly, Dash)

---

## ⚙️ Pipeline de Datos

El pipeline de carga es **robusto con tres métodos en cascada**. Si el primero falla, se intenta el siguiente automáticamente.

```
┌─────────────────────────────────────────────────────────┐
│                  prepare_data()                          │
├──────────────┬──────────────────┬───────────────────────┤
│  MÉTODO 1    │    MÉTODO 2      │      MÉTODO 3         │
│  wbgapi      │    REST API      │   Datos sintéticos    │
│  (preferido) │    (fallback)    │   (offline fallback)  │
└──────────────┴──────────────────┴───────────────────────┘
                         │
                         ▼
        Columnas: iso3 | pais | anio | tasa_fin
                         │
               drop_duplicates(iso3, anio)
                         │
          ┌──────────────┴──────────────┐
          │                             │
         df_1                         df_2
    (agregados)              (países individuales)
                             dropna(tasa_fin)
                             clip(0, 100) → tasa_fin_cap
                             sort(pais, anio)
                             ─────────────────
                             2,992 observaciones
```

### Método 1 — wbgapi (recomendado)
```python
wb.economy.list()           # obtiene todos los países individuales
wb.data.DataFrame(
    "SE.PRM.CMPT.FE.ZS",
    economy=indiv_iso3,     # solo países (region != NA)
    time=range(2000, 2023),
    skipBlanks=False,       # conserva NAs → df_2 exacto
)
```

### Método 2 — REST API del Banco Mundial
- Llama a `/v2/country` para obtener la lista de países con `region.id != "NA"`
- Descarga `/v2/country/{iso3_list}/indicator/SE.PRM.CMPT.FE.ZS` paginado
- Conserva filas con valor nulo como `float("nan")`

### Método 3 — Dataset sintético ISO-3
- 130 países × 23 años con ~12% de NAs aleatorios
- Misma estructura de columnas que los métodos reales
- Solo para entornos completamente sin acceso a internet

### Separación df_1 / df_2
| DataFrame | Contenido | Uso |
|-----------|-----------|-----|
| `df_raw` | Todos los registros descargados, incluye NAs | Diagnóstico |
| `df_1` | Agregados regionales (iso3 no de 3 letras) | Referencia |
| `df_2` | Países individuales, sin NA, con `tasa_fin_cap` | Todas las visualizaciones |

---

## 💻 Requisitos del Sistema

### Python
- **Versión mínima:** Python 3.10

### Dependencias

```
dash>=2.14.0
dash-bootstrap-components>=1.5.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scipy>=1.11.0
statsmodels>=0.14.0
wbgapi>=1.0.12
requests>=2.31.0
```

### Navegadores compatibles
Chrome 90+, Firefox 88+, Edge 90+, Safari 14+

### Recursos mínimos recomendados
| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| RAM | 512 MB | 1 GB |
| CPU | 1 núcleo | 2 núcleos |
| Disco | 50 MB | 100 MB |
| Conexión | Requerida al inicio* | Banda ancha |

> *La conexión a internet es necesaria en el arranque para descargar los datos de la API del Banco Mundial. Una vez cargados, el dashboard funciona offline.

---

## 🚀 Instalación

### Opción A — Instalación directa (recomendada)

```bash
# 1. Clonar o descargar el repositorio
cd mi_proyecto/

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar el entorno
# En Windows:
venv\Scripts\activate
# En macOS / Linux:
source venv/bin/activate

# 4. Instalar todas las dependencias
pip install dash dash-bootstrap-components pandas numpy plotly scipy statsmodels wbgapi requests
```

### Opción B — Con archivo requirements.txt

Crear el archivo `requirements.txt`:

```txt
dash>=2.14.0
dash-bootstrap-components>=1.5.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scipy>=1.11.0
statsmodels>=0.14.0
wbgapi>=1.0.12
requests>=2.31.0
```

Luego instalar:

```bash
pip install -r requirements.txt
```

### Verificar la instalación

```bash
python -c "import dash, plotly, pandas, wbgapi; print('✅ Todo instalado correctamente')"
```

---

## ▶️ Ejecución

```bash
# Asegurarse de que el entorno virtual esté activo
python app_dash_ods4.py
```

Al arrancar, la consola mostrará el progreso de carga de datos:

```
Cargando datos WDI...
  [wbgapi] 217 países individuales encontrados
  [wbgapi] 4991 filas totales (con NAs), 217 países, 2992 no-nulas
  df_2 final: 2992 filas, 130 países individuales
============================================================
  App Dash — Finalización Primaria Femenina ODS 4
  Abrir en: http://127.0.0.1:8050
============================================================
```

Abrir el navegador en: **http://127.0.0.1:8050**

> El primer inicio puede tardar **15–60 segundos** dependiendo de la velocidad de conexión, ya que descarga los datos de la API del Banco Mundial.

---

## 🏗 Arquitectura del Código

```
app_dash_ods4.py
│
├── CONFIGURACIÓN GLOBAL
│   ├── Paleta de colores (constantes hex)
│   └── pl_layout()          — tema oscuro estándar para todas las figuras
│
├── PIPELINE DE DATOS
│   ├── _get_individual_country_iso3()  — lista ISO-3 desde API /v2/country
│   ├── _FALLBACK_ISO3                  — lista embebida de 217 países
│   ├── prepare_data()                  — 3 métodos en cascada + validaciones
│   └── _synthetic_data_iso3()          — dataset de respaldo offline
│
├── ESTADÍSTICAS GLOBALES
│   └── N_OBS, N_PAISES, MEDIA_GLOBAL, etc. (calculadas al inicio)
│
├── HELPERS DE UI
│   ├── interp_box()    — caja de interpretación azul
│   ├── nota_box()      — caja de advertencia amarilla
│   ├── card_s()        — tarjeta con borde de color
│   ├── page_header()   — encabezado de sección con gradiente
│   ├── kpi_card()      — tarjeta de métrica
│   └── _empty_fig()    — figura vacía con mensaje de error
│
├── LAYOUT
│   ├── sidebar         — menú lateral fijo (7 botones de navegación)
│   ├── header          — barra superior con título
│   ├── tab_intro       — contenido Introducción
│   ├── tab_univariado  — contenido Análisis Univariado
│   ├── tab_bivariado   — contenido Análisis Bivariado
│   ├── tab_mapa        — contenido Mapas Mundiales
│   ├── tab_conclusiones— contenido Conclusiones
│   ├── tab_datos       — contenido Explorador de Datos
│   └── tab_referencias — contenido Referencias
│
└── CALLBACKS (Dash reactive)
    ├── navigate()           — enruta entre las 7 secciones principales
    ├── render_intro_tab()   — renderiza sub-pestañas de Introducción
    ├── render_adf()         — ADF + ACF + PACF
    ├── render_hist_box()    — Histograma + Boxplot + estadísticas
    ├── render_tabla_gt100() — Tabla de observaciones > 100%
    ├── render_lollipop()    — Lollipop IQR por quinquenio
    ├── render_techo()       — % de años en el techo
    ├── render_scatter_loess()  — Scatter + LOESS
    ├── render_mediana_anual()  — Mediana + banda IQR anual
    ├── render_media_mediana()  — Media vs. Mediana
    ├── render_ribbon()      — Ribbon multicapa P10–P90
    ├── render_heatmap()     — Mapa de calor país × año
    ├── render_slopes()      — Pendientes pre-techo
    ├── render_evol_paises() — Trayectorias individuales
    ├── render_hipotesis()   — Prueba estadística comparativa
    ├── render_mapas()       — 3 mapas coropléticos ISO-3
    └── render_tabla_datos() — Tabla de datos interactiva
```

### Patrón de navegación

Cada callback de visualización recibe `Input("current-tab", "data")` y aplica un **guard de pestaña activa** antes de computar:

```python
def render_adf(tab):
    if tab != "univariado":
        return html.Div(), html.Div(), go.Figure(), go.Figure()
    # ... lógica real
```

Esto evita que Dash intente actualizar componentes que no existen en el DOM cuando el usuario está en otra sección.

---

## 📚 Referencias

| # | Referencia |
|---|-----------|
| 1 | Banco Mundial. (2023). *Primary completion rate, female (% of relevant age group) — SE.PRM.CMPT.FE.ZS*. World Development Indicators. https://data.worldbank.org/indicator/SE.PRM.CMPT.FE.ZS |
| 2 | Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *JASA, 74*(366), 427–431. |
| 3 | Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley. |
| 4 | Becker, G. S. (1964). *Human Capital: A Theoretical and Empirical Analysis*. University of Chicago Press. |
| 5 | Barro, R. J., & Lee, J. W. (2013). A new data set of educational attainment in the world, 1950–2010. *Journal of Development Economics, 104*, 184–198. |
| 6 | Naciones Unidas. (2015). *Transforming our world: the 2030 Agenda for Sustainable Development*. A/RES/70/1. https://sdgs.un.org/goals/goal4 |
| 7 | UNESCO. (2023). *Global Education Monitoring Report 2023*. UNESCO Publishing. |
| 8 | Plotly. (2023). *Dash: Analytical Web Apps for Python*. https://dash.plotly.com |

---

<div align="center">

**ODS 4 · Educación de Calidad · Banco Mundial WDI · 2000–2022**

*Juan Marín · Samuel Chamorro — Científicos de Datos*

</div>
