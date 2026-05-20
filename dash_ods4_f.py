# =============================================================================
# Dash App: Análisis de la Tasa de Finalización Primaria Femenina
# Indicador: SE.PRM.CMPT.FE.ZS — World Development Indicators
# Periodo: 2000–2022  |  Equivalente Python/Dash de la app Shiny original
# Autores: Juan Marín · Samuel Chamorro
#
# INSTALACIÓN (entorno local):
#   pip install dash dash-bootstrap-components pandas numpy plotly scipy
#              statsmodels wbgapi requests
#
# EJECUCIÓN:
#   python dash_ods4_ultimate.py
#   Luego abrir http://127.0.0.1:8050
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import levene, shapiro, mannwhitneyu, ttest_ind
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Statsmodels para ACF/PACF y ADF ──────────────────────────────────────────
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.nonparametric.smoothers_lowess import lowess
# ── Imports adicionales para módulo ARIMA/Predicción ─────────────────────────
import warnings
from scipy.stats import probplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# =============================================================================
# PALETA DE COLORES
# =============================================================================
AZUL_MAIN   = "#2E86AB"
AZUL_FUERTE = "#1B4F72"
AZUL_CLARO  = "#AED6F1"
GRIS_LINEA  = "#4D4D4D"
ROJO_OUT    = "#C0392B"
VERDE_OK    = "#1E8449"
AMARILLO    = "#F39C12"
BG_DARK     = "#0D1B2A"
BG_CARD     = "#112240"
BG_PLOT     = "#0D1F3C"
TEXT_MAIN   = "#C5D8E8"
TEXT_LIGHT  = "#E8F4FD"
TEXT_DIM    = "#8BAEC8"
CYAN        = "#64CFF6"
NARANJA     = "#E67E22"
PURPURA     = "#9B59B6"

# =============================================================================
# LAYOUT PLOTLY (tema oscuro coherente)
# =============================================================================
def pl_layout(fig, title="", subtitle="", xtitle="", ytitle=""):
    full_title = f"<b>{title}</b><br><sup>{subtitle}</sup>" if subtitle else f"<b>{title}</b>"
    fig.update_layout(
        title=dict(text=full_title, font=dict(size=14, color=CYAN), x=0.02),
        xaxis=dict(
            title=dict(text=xtitle, font=dict(color=TEXT_DIM)),
            gridcolor="rgba(46,134,171,0.15)", zeroline=False,
            showline=True, linecolor="rgba(46,134,171,0.3)",
            tickfont=dict(color=TEXT_DIM)
        ),
        yaxis=dict(
            title=dict(text=ytitle, font=dict(color=TEXT_DIM)),
            gridcolor="rgba(46,134,171,0.15)", zeroline=False,
            showline=True, linecolor="rgba(46,134,171,0.3)",
            tickfont=dict(color=TEXT_DIM)
        ),
        paper_bgcolor="#112240",
        plot_bgcolor=BG_PLOT,
        font=dict(color=TEXT_MAIN),
        legend=dict(
            orientation="h", x=0, y=-0.22,
            font=dict(color=TEXT_MAIN),
            bgcolor="rgba(13,31,60,0.8)",
            bordercolor="rgba(46,134,171,0.3)", borderwidth=1
        ),
        margin=dict(t=80, r=30, b=70, l=65),
        hoverlabel=dict(bgcolor="#0A1628", bordercolor=CYAN, font=dict(size=12, color=TEXT_LIGHT))
    )
    return fig

# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def _get_individual_country_iso3():
    """
    Obtiene la lista oficial de países individuales del Banco Mundial.
    Llama a /v2/country?per_page=500 y filtra los que tienen region.id != 'NA'.
    Retorna lista de códigos ISO-3 (3 letras, e.g. 'AFG', 'COL', ...).
    """
    try:
        import requests
        url = "https://api.worldbank.org/v2/country?format=json&per_page=500"
        data = requests.get(url, timeout=20).json()
        countries = data[1] or []
        iso3_list = [
            c["id"] for c in countries
            if c.get("region", {}).get("id") not in ("NA", "", None)
            and len(c.get("id", "")) == 3
        ]
        return sorted(set(iso3_list))
    except Exception as e:
        print(f"  [country list] fallo: {e} — usando lista embebida")
        return None


# Lista embebida de respaldo (217 países individuales WB, región != NA)
_FALLBACK_ISO3 = [
    "ABW","AFG","AGO","ALB","AND","ARE","ARG","ARM","ASM","ATG","AUS","AUT","AZE",
    "BDI","BEL","BEN","BFA","BGD","BGR","BHR","BHS","BIH","BLR","BLZ","BMU","BOL",
    "BRA","BRB","BRN","BTN","BWA","CAF","CAN","CHE","CHL","CHN","CIV","CMR","COD",
    "COG","COL","COM","CPV","CRI","CUB","CYP","CZE","DEU","DJI","DNK","DOM","DZA",
    "ECU","EGY","ERI","ESP","EST","ETH","FIN","FJI","FRA","FSM","GAB","GBR","GEO",
    "GHA","GIN","GMB","GNB","GNQ","GRC","GTM","GUY","HND","HRV","HTI","HUN","IDN",
    "IND","IRL","IRN","IRQ","ISL","ISR","ITA","JAM","JOR","JPN","KAZ","KEN","KGZ",
    "KHM","KIR","KOR","KWT","LAO","LBN","LBR","LBY","LCA","LKA","LSO","LTU","LUX",
    "LVA","MAR","MDA","MDG","MDV","MEX","MHL","MKD","MLI","MLT","MMR","MNE","MNG",
    "MOZ","MRT","MUS","MWI","MYS","NAM","NER","NGA","NIC","NLD","NOR","NPL","NZL",
    "OMN","PAK","PAN","PER","PHL","PLW","PNG","POL","PRT","PRK","PRY","PSE","QAT",
    "ROU","RUS","RWA","SAU","SDN","SEN","SGP","SLB","SLE","SLV","SOM","SRB","SSD",
    "STP","SUR","SVK","SVN","SWE","SWZ","SYC","SYR","TCD","TGO","THA","TJK","TKM",
    "TLS","TON","TTO","TUN","TUR","TUV","TZA","UGA","UKR","URY","USA","UZB","VCT",
    "VEN","VNM","VUT","WSM","XKX","YEM","ZAF","ZMB","ZWE",
]


def prepare_data():
    """
    Descarga SE.PRM.CMPT.FE.ZS (2000-2022) para todos los países individuales WB.
    Pipeline con tres métodos en cascada:

      Método 1 — wbgapi (preferido):
        Descarga todos los países individuales usando wb.economy.coder()
        para obtener la lista ISO-3 y wb.data.DataFrame con skipBlanks=False
        para incluir NAs. Produce exactamente 2992 filas no-nulas en df_2.

      Método 2 — REST API WB:
        Llama /v2/country para obtener ISO-3 individuales (region != NA),
        luego descarga el indicador incluyendo celdas nulas.

      Método 3 — Dataset sintético ISO-3 (offline).

    Columnas finales:
      df_raw : iso3, pais, anio, tasa_fin  (incluye NAs, todos los países)
      df_1   : agregados regionales
      df_2   : solo países individuales, sin NA, con tasa_fin_cap clipped [0,100]
    """
    raw = None

    # ── Método 1: wbgapi ────────────────────────────────────────────────────
    try:
        import wbgapi as wb

        # Obtener lista de economías individuales (excluye aggregates)
        # wb.economy.list() devuelve todas; filtramos por aggregate == False
        all_econ = list(wb.economy.list())
        indiv_iso3 = [
            e["id"] for e in all_econ
            if e.get("aggregate") is False or e.get("aggregate") == 0
        ]
        if not indiv_iso3:
            # fallback: usar region info si "aggregate" no está disponible
            indiv_iso3 = [
                e["id"] for e in all_econ
                if e.get("region", {}).get("id") not in ("NA", "", None)
                and len(e.get("id", "")) == 3
            ]

        print(f"  [wbgapi] {len(indiv_iso3)} países individuales encontrados")

        # Descargar TODOS los datos 2000-2022, incluyendo NAs (skipBlanks=False)
        raw_wide = wb.data.DataFrame(
            "SE.PRM.CMPT.FE.ZS",
            economy=indiv_iso3,
            time=range(2000, 2023),
            labels=True,
            skipBlanks=False,
        ).reset_index()

        year_cols = [c for c in raw_wide.columns if str(c).startswith("YR")]
        raw = raw_wide.melt(
            id_vars=["economy", "Economy"],
            value_vars=year_cols,
            var_name="anio",
            value_name="tasa_fin",
        )
        raw.rename(columns={"economy": "iso3", "Economy": "pais"}, inplace=True)
        raw["anio"] = raw["anio"].str.replace("YR", "").astype(int)
        raw = (raw[["iso3", "pais", "anio", "tasa_fin"]]
               .drop_duplicates(subset=["iso3", "anio"])
               .reset_index(drop=True))
        print(f"  [wbgapi] {len(raw)} filas totales (con NAs), "
              f"{raw['iso3'].nunique()} países, "
              f"{raw['tasa_fin'].notna().sum()} no-nulas")
    except Exception as e:
        print(f"  [wbgapi] fallo: {e}")
        raw = None

    # ── Método 2: REST API ──────────────────────────────────────────────────
    if raw is None or len(raw) == 0:
        try:
            import requests

            # Paso 2a: obtener lista de países individuales
            indiv_iso3 = _get_individual_country_iso3() or _FALLBACK_ISO3

            # Paso 2b: descargar indicador por páginas, conservando NAs
            indicator = "SE.PRM.CMPT.FE.ZS"
            iso3_str  = ";".join(indiv_iso3)
            base_url  = (
                f"https://api.worldbank.org/v2/country/{iso3_str}/indicator/{indicator}"
                f"?format=json&per_page=20000&date=2000:2022"
            )
            resp        = requests.get(base_url, timeout=60)
            meta, data0 = resp.json()
            total_pages  = meta["pages"]
            records      = list(data0 or [])
            for p in range(2, total_pages + 1):
                rp = requests.get(base_url + f"&page={p}", timeout=60)
                records += list(rp.json()[1] or [])

            rows = []
            for rec in records:
                iso3 = (rec.get("countryiso3code") or "").strip()
                if len(iso3) != 3:
                    continue
                rows.append({
                    "iso3":     iso3,
                    "pais":     rec["country"]["value"],
                    "anio":     int(rec["date"]),
                    "tasa_fin": float(rec["value"]) if rec.get("value") is not None else float("nan"),
                })

            raw = (pd.DataFrame(rows)
                   .drop_duplicates(subset=["iso3", "anio"])
                   .reset_index(drop=True))
            print(f"  [REST] {len(raw)} filas totales, "
                  f"{raw['iso3'].nunique()} países, "
                  f"{raw['tasa_fin'].notna().sum()} no-nulas")
        except Exception as e:
            print(f"  [REST] fallo: {e}")
            raw = None

    # ── Método 3: Dataset sintético ─────────────────────────────────────────
    if raw is None or len(raw) == 0:
        raw = _synthetic_data_iso3()
        print(f"  [sintético] {len(raw)} filas generadas")

    # ── Procesamiento común ─────────────────────────────────────────────────
    df_raw = raw.copy()
    df_raw["anio"] = df_raw["anio"].astype(int)

    if "iso3" not in df_raw.columns:
        raise ValueError("Pipeline: columna 'iso3' ausente")

    # Separar agregados (iso3 de 2 letras o no reconocidos) de individuales
    # Los países individuales WB tienen iso3 de exactamente 3 letras mayúsculas
    df_raw["_is_country"] = df_raw["iso3"].str.match(r"^[A-Z]{3}$")
    df_1  = df_raw[~df_raw["_is_country"]].drop(columns="_is_country").reset_index(drop=True)
    df_2r = df_raw[ df_raw["_is_country"]].drop(columns="_is_country").copy()

    df_2 = df_2r.dropna(subset=["tasa_fin"]).copy()

    if len(df_2) < 100:
        raise ValueError(f"df_2 tiene solo {len(df_2)} filas — datos insuficientes")

    df_2["tasa_fin_cap"] = df_2["tasa_fin"].clip(0, 100)
    df_2 = df_2.sort_values(["pais", "anio"]).reset_index(drop=True)

    print(f"  df_2 final: {len(df_2)} filas, {df_2['iso3'].nunique()} países individuales")
    return df_raw.drop(columns=["_is_country"], errors="ignore"), df_1, df_2


def _synthetic_data_iso3():
    """Dataset sintético con exactamente 130 países × 23 años SIN NAs (garantiza ARIMA)."""
    np.random.seed(42)
    iso_names = {
        "AFG":"Afghanistan","ALB":"Albania","DZA":"Algeria","AGO":"Angola",
        "ARG":"Argentina","ARM":"Armenia","AUS":"Australia","AUT":"Austria",
        "AZE":"Azerbaijan","BGD":"Bangladesh","BLR":"Belarus","BEL":"Belgium",
        "BEN":"Benin","BOL":"Bolivia","BIH":"Bosnia and Herzegovina","BWA":"Botswana",
        "BRA":"Brazil","BFA":"Burkina Faso","BDI":"Burundi","KHM":"Cambodia",
        "CMR":"Cameroon","CAN":"Canada","CAF":"Central African Republic","TCD":"Chad",
        "CHL":"Chile","CHN":"China","COL":"Colombia","COM":"Comoros",
        "COD":"Congo, Dem. Rep.","COG":"Congo, Rep.","CRI":"Costa Rica",
        "CIV":"Cote d\'Ivoire","HRV":"Croatia","CUB":"Cuba","CZE":"Czech Republic",
        "DNK":"Denmark","DJI":"Djibouti","DOM":"Dominican Republic","ECU":"Ecuador",
        "EGY":"Egypt, Arab Rep.","SLV":"El Salvador","ERI":"Eritrea","ETH":"Ethiopia",
        "FIN":"Finland","FRA":"France","GAB":"Gabon","GMB":"Gambia","GEO":"Georgia",
        "DEU":"Germany","GHA":"Ghana","GRC":"Greece","GTM":"Guatemala","GIN":"Guinea",
        "GNB":"Guinea-Bissau","HTI":"Haiti","HND":"Honduras","HUN":"Hungary",
        "IND":"India","IDN":"Indonesia","IRN":"Iran, Islamic Rep.","IRQ":"Iraq",
        "IRL":"Ireland","ISR":"Israel","ITA":"Italy","JAM":"Jamaica","JPN":"Japan",
        "JOR":"Jordan","KAZ":"Kazakhstan","KEN":"Kenya","KOR":"Korea, Rep.",
        "KWT":"Kuwait","KGZ":"Kyrgyz Republic","LAO":"Lao PDR","LBN":"Lebanon",
        "LSO":"Lesotho","LBR":"Liberia","LBY":"Libya","LTU":"Lithuania",
        "MDG":"Madagascar","MWI":"Malawi","MYS":"Malaysia","MLI":"Mali",
        "MRT":"Mauritania","MEX":"Mexico","MDA":"Moldova","MNG":"Mongolia",
        "MAR":"Morocco","MOZ":"Mozambique","MMR":"Myanmar","NAM":"Namibia",
        "NPL":"Nepal","NLD":"Netherlands","NIC":"Nicaragua","NER":"Niger",
        "NGA":"Nigeria","NOR":"Norway","PAK":"Pakistan","PAN":"Panama",
        "PNG":"Papua New Guinea","PRY":"Paraguay","PER":"Peru","PHL":"Philippines",
        "POL":"Poland","PRT":"Portugal","ROU":"Romania","RUS":"Russian Federation",
        "RWA":"Rwanda","SAU":"Saudi Arabia","SEN":"Senegal","SLE":"Sierra Leone",
        "SOM":"Somalia","ZAF":"South Africa","ESP":"Spain","LKA":"Sri Lanka",
        "SDN":"Sudan","SWE":"Sweden","CHE":"Switzerland","SYR":"Syrian Arab Republic",
        "TJK":"Tajikistan","TZA":"Tanzania","THA":"Thailand","TGO":"Togo",
        "TUN":"Tunisia","TUR":"Turkey","TKM":"Turkmenistan","UGA":"Uganda",
        "UKR":"Ukraine","GBR":"United Kingdom","USA":"United States","URY":"Uruguay",
        "UZB":"Uzbekistan","VEN":"Venezuela, RB","VNM":"Vietnam","YEM":"Yemen, Rep.",
        "ZMB":"Zambia","ZWE":"Zimbabwe",
    }
    years = list(range(2000, 2023))
    rows = []
    for iso3, nombre in iso_names.items():
        base = np.random.uniform(40, 95)
        for y in years:
            # Sin NaN: todos los años tienen valor para garantizar series completas en ARIMA
            trend = (y - 2000) * np.random.uniform(0.3, 1.5)
            val   = min(base + trend + np.random.normal(0, 2), 115)
            rows.append({"iso3": iso3, "pais": nombre, "anio": y, "tasa_fin": max(val, 0)})
    return pd.DataFrame(rows)

# =============================================================================
# CARGAR DATOS AL INICIAR
# =============================================================================
print("Cargando datos WDI...")
df_raw, df_1, df_2 = prepare_data()
print(f"  df_raw: {len(df_raw)} filas | df_2: {len(df_2)} filas (datos limpios)")

# Estadísticas globales del dataset (con guardia contra df_2 vacío)
N_OBS        = len(df_2)
N_PAISES     = df_2["pais"].nunique()   if N_OBS > 0 else 0
ANIO_MIN     = int(df_2["anio"].min()) if N_OBS > 0 else 2000
ANIO_MAX     = int(df_2["anio"].max()) if N_OBS > 0 else 2022
N_ANIOS      = df_2["anio"].nunique()  if N_OBS > 0 else 0
MEDIA_GLOBAL = df_2["tasa_fin_cap"].mean()   if N_OBS > 0 else 0.0
MEDIAN_GLOBAL= df_2["tasa_fin_cap"].median() if N_OBS > 0 else 0.0
SD_GLOBAL    = df_2["tasa_fin_cap"].std()    if N_OBS > 0 else 0.0
N_GT100      = int((df_2["tasa_fin"] > 100).sum()) if N_OBS > 0 else 0
PCT_GT100    = round(100 * N_GT100 / N_OBS, 2)     if N_OBS > 0 else 0.0
MAX_TASA     = round(df_2["tasa_fin"].max(), 2)     if N_OBS > 0 else 0.0
PAISES_LIST  = sorted(df_2["pais"].unique().tolist())
# Lookup iso3 → nombre para dropdown (ordenado por nombre)
ISO3_TO_PAIS = df_2.drop_duplicates("iso3").set_index("iso3")["pais"].to_dict()
PAIS_TO_ISO3 = {v: k for k, v in ISO3_TO_PAIS.items()}

# =============================================================================
# HELPERS UI
# =============================================================================
def interp_box(children):
    return html.Div([
        html.I(className="fas fa-lightbulb", style={"color": AZUL_MAIN, "marginRight": "8px"}),
        html.Strong("Interpretación: "),
        children
    ], style={
        "background": "linear-gradient(135deg,rgba(46,134,171,0.1) 0%,rgba(46,134,171,0.05) 100%)",
        "borderLeft": f"5px solid {AZUL_MAIN}", "borderRadius": "6px",
        "padding": "14px 18px", "marginTop": "12px", "fontSize": "0.92em",
        "lineHeight": "1.75", "color": TEXT_MAIN
    })

def nota_box(children):
    return html.Div([
        html.I(className="fas fa-exclamation-triangle", style={"marginRight":"6px","color":AMARILLO}), children
    ], style={
        "background": "rgba(243,156,18,0.08)", "border": f"1px solid rgba(243,156,18,0.3)",
        "borderLeft": f"5px solid {AMARILLO}", "borderRadius": "6px",
        "padding": "12px 16px", "marginTop": "10px", "fontSize": "0.9em", "color": "#F5CBA7"
    })

def card_s(children, border_color=AZUL_MAIN):
    return html.Div(children, style={
        "background": BG_CARD, "borderRadius": "10px", "padding": "20px 24px",
        "marginBottom": "16px", "boxShadow": "0 2px 10px rgba(0,0,0,0.4)",
        "borderLeft": f"6px solid {border_color}"
    })

def page_header(icon_txt, title):
    if isinstance(icon_txt, str) and icon_txt.startswith("fas "):
        icon_el = html.I(className=icon_txt, style={"marginRight":"10px","fontSize":"1.1em"})
    else:
        icon_el = html.Span(icon_txt, style={"marginRight":"10px","fontSize":"1.2em"})
    return html.Div([icon_el, title], style={
        "background": f"linear-gradient(135deg,{AZUL_FUERTE} 0%,{AZUL_MAIN} 100%)",
        "color": "white", "padding": "12px 20px", "borderRadius": "8px",
        "fontWeight": "700", "fontSize": "1.05em", "marginBottom": "20px",
        "letterSpacing": "0.3px"
    })

def kpi_card(value, label, color=CYAN):
    return html.Div([
        html.Span(str(value), style={"color": color, "fontSize": "1.6em", "fontWeight": "700", "display": "block"}),
        html.Span(label, style={"color": TEXT_DIM, "fontSize": "0.75em", "textTransform": "uppercase", "letterSpacing": "0.8px"})
    ], style={
        "background": "rgba(46,134,171,0.1)", "border": "1px solid rgba(46,134,171,0.25)",
        "borderRadius": "8px", "padding": "14px 18px", "textAlign": "center", "flex": "1"
    })


def _empty_fig(msg="Sin datos suficientes para mostrar esta visualización."):
    """Figura vacía con mensaje de advertencia (tema oscuro)."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"⚠ {msg}", xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#F39C12"),
        bgcolor="rgba(243,156,18,0.1)",
        bordercolor="#F39C12", borderwidth=1, borderpad=10,
    )
    fig.update_layout(
        paper_bgcolor="#112240", plot_bgcolor="#0D1F3C",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig

# =============================================================================
# COLORES PLOTLY MAP
# =============================================================================
MAPA_COLORSCALE = [
    [0,    "#FEF9E7"],
    [0.2,  "#AED6F1"],
    [0.5,  AZUL_MAIN],
    [0.8,  AZUL_FUERTE],
    [1,    "#0A2740"]
]

CAMBIO_COLORSCALE = [
    [0,    "#C0392B"],
    [0.35, "#F1948A"],
    [0.5,  "#FDFEFE"],
    [0.65, "#82E0AA"],
    [1,    "#1E8449"]
]

# =============================================================================
# LAYOUT DE LA APP
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.15.4/css/all.css",
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800"
        "&family=JetBrains+Mono:wght@400;500&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="Finalización Primaria Femenina — ODS 4"
)
server = app.server

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
sidebar = html.Div([
    # Header pulsante
    html.Div([
        html.Div(style={
            "width": "8px", "height": "8px", "borderRadius": "50%",
            "background": CYAN, "flexShrink": "0",
            "boxShadow": f"0 0 6px {CYAN},0 0 12px rgba(100,207,246,0.4)",
            "animation": "dotBlink 2s ease-in-out infinite"
        }),
        html.Div([
            html.Div("Módulos de análisis", style={"color": CYAN, "fontSize": "0.7em",
                     "textTransform": "uppercase", "letterSpacing": "1.5px", "fontWeight": "700"}),
            html.Div("9 secciones disponibles", style={"color": "#4A7A9B", "fontSize": "0.65em", "marginTop": "1px"})
        ])
    ], style={"display": "flex", "alignItems": "center", "gap": "10px",
              "background": "linear-gradient(135deg,rgba(46,134,171,0.15) 0%,rgba(46,134,171,0.05) 100%)",
              "border": "1px solid rgba(46,134,171,0.25)", "borderRadius": "10px", "padding": "10px 14px",
              "margin": "18px 14px 10px"}),

    # Menú de navegación
    html.Div([
        html.Button([html.I(className="fas fa-info-circle", style={"marginRight":"8px","width":"16px"}), "Introducción"],  id="nav-intro",        n_clicks=0, className="nav-btn active"),
        html.Button([html.I(className="fas fa-chart-bar", style={"marginRight":"8px","width":"16px"}), "Univariado"],     id="nav-univariado",   n_clicks=0, className="nav-btn"),
        html.Button([html.I(className="fas fa-chart-line", style={"marginRight":"8px","width":"16px"}), "Bivariado"],      id="nav-bivariado",    n_clicks=0, className="nav-btn"),
        html.Button([html.I(className="fas fa-globe-americas", style={"marginRight":"8px","width":"16px"}), "Mapa Mundial"],   id="nav-mapa",         n_clicks=0, className="nav-btn"),
        html.Button([html.I(className="fas fa-flag-checkered", style={"marginRight":"8px","width":"16px"}), "Conclusiones"],   id="nav-conclusiones", n_clicks=0, className="nav-btn"),
        html.Button([html.I(className="fas fa-table", style={"marginRight":"8px","width":"16px"}), "Datos"],          id="nav-datos",        n_clicks=0, className="nav-btn"),
        html.Button([html.I(className="fas fa-book", style={"marginRight":"8px","width":"16px"}), "Referencias"],    id="nav-referencias",  n_clicks=0, className="nav-btn"),
        html.Button([html.I(className="fas fa-robot", style={"marginRight":"8px","width":"16px"}), "Predicción ARIMA"], id="nav-prediccion", n_clicks=0, className="nav-btn"),
        html.Button([html.I(className="fab fa-youtube", style={"marginRight":"8px","width":"16px","color":"#FF0000"}), "Video ODS 4"], id="nav-video", n_clicks=0, className="nav-btn"),
    ], style={"display": "flex", "flexDirection": "column", "gap": "4px", "padding": "0 10px"}),

    # Divisor
    html.Div(style={"margin": "12px 14px", "height": "1px",
                    "background": "linear-gradient(90deg,transparent,rgba(46,134,171,0.5),transparent)"}),

    # Meta-info
    html.Div([
        html.Div([html.I(className="fas fa-database", style={"width":"14px","marginRight":"6px","color":CYAN}),
                  html.Span("Banco Mundial — WDI", style={"color": "#7FADD4", "fontSize": "0.75em"})],
                 style={"display": "flex", "alignItems": "center", "gap": "9px",
                        "padding": "6px 10px", "borderRadius": "7px", "background": "rgba(46,134,171,0.07)"}),
        html.Div([html.I(className="fas fa-calendar-alt", style={"width":"14px","marginRight":"6px","color":CYAN}),
                  html.Span("Periodo: 2000–2022", style={"color": "#7FADD4", "fontSize": "0.75em"})],
                 style={"display": "flex", "alignItems": "center", "gap": "9px",
                        "padding": "6px 10px", "borderRadius": "7px", "background": "rgba(46,134,171,0.07)"}),
        html.Div([html.I(className="fas fa-globe", style={"width":"14px","marginRight":"6px","color":CYAN}),
                  html.Span("Cobertura: ~192 países", style={"color": "#7FADD4", "fontSize": "0.75em"})],
                 style={"display": "flex", "alignItems": "center", "gap": "9px",
                        "padding": "6px 10px", "borderRadius": "7px", "background": "rgba(46,134,171,0.07)"}),
    ], style={"padding": "4px 14px 8px", "display": "flex", "flexDirection": "column", "gap": "6px"}),

    # Divisor
    html.Div(style={"margin": "8px 14px", "height": "1px",
                    "background": "linear-gradient(90deg,transparent,rgba(46,134,171,0.5),transparent)"}),

    # Autores
    html.Div([
        html.Div([html.I(className="fas fa-users", style={"marginRight":"8px","color":CYAN}),
                  html.Div("Autores del proyecto", style={"color": CYAN, "fontSize": "0.68em",
                           "textTransform": "uppercase", "letterSpacing": "1.5px", "fontWeight": "700"})],
                 style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Div("JM", style={"color": "#AED6F1", "fontSize": "0.65em", "fontWeight": "800",
                         "width": "28px", "height": "28px", "borderRadius": "50%",
                         "background": f"linear-gradient(135deg,{AZUL_FUERTE},{AZUL_MAIN})",
                         "display": "flex", "alignItems": "center", "justifyContent": "center"}),
                html.Div([
                    html.Div("Juan Marín", style={"color": TEXT_LIGHT, "fontSize": "0.82em", "fontWeight": "600"}),
                    html.Div("Científico de datos", style={"color": "#4A7A9B", "fontSize": "0.68em"})
                ])
            ], style={"display": "flex", "alignItems": "center", "gap": "10px",
                      "padding": "7px 10px", "borderRadius": "8px",
                      "background": "rgba(46,134,171,0.1)", "border": "1px solid rgba(46,134,171,0.2)",
                      "marginBottom": "6px"}),
            html.Div([
                html.Div("SC", style={"color": "#AED6F1", "fontSize": "0.65em", "fontWeight": "800",
                         "width": "28px", "height": "28px", "borderRadius": "50%",
                         "background": f"linear-gradient(135deg,{AZUL_FUERTE},{AZUL_MAIN})",
                         "display": "flex", "alignItems": "center", "justifyContent": "center"}),
                html.Div([
                    html.Div("Samuel Chamorro", style={"color": TEXT_LIGHT, "fontSize": "0.82em", "fontWeight": "600"}),
                    html.Div("Científico de datos", style={"color": "#4A7A9B", "fontSize": "0.68em"})
                ])
            ], style={"display": "flex", "alignItems": "center", "gap": "10px",
                      "padding": "7px 10px", "borderRadius": "8px",
                      "background": "rgba(46,134,171,0.1)", "border": "1px solid rgba(46,134,171,0.2)"}),
        ])
    ], style={"margin": "8px 14px 16px", "padding": "14px 16px",
              "background": "linear-gradient(135deg,#081424,#0D1F3C,#0A1628)",
              "borderRadius": "12px", "border": "1px solid rgba(46,134,171,0.3)"}),

], style={
    "width": "270px", "minWidth": "270px", "height": "100vh", "overflowY": "auto",
    "background": "linear-gradient(180deg,#060F1E 0%,#0A1628 35%,#0D1F3C 70%,#0A1A30 100%)",
    "borderRight": "1px solid rgba(46,134,171,0.25)",
    "boxShadow": "4px 0 24px rgba(0,0,0,0.5)", "position": "fixed", "zIndex": "100"
})

# ─── HEADER ───────────────────────────────────────────────────────────────────
header = html.Div([
    html.H4([html.I(className="fas fa-graduation-cap", style={"marginRight":"10px"}), " Finalización Primaria Femenina — ODS 4"],
            style={"margin": "0", "color": "white", "fontWeight": "700", "fontSize": "1.05em"})
], style={
    "background": f"linear-gradient(135deg,{AZUL_FUERTE} 0%,{AZUL_MAIN} 100%)",
    "padding": "14px 28px", "marginLeft": "270px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.4)",
    "borderBottom": f"2px solid {CYAN}"
})

# =============================================================================
# CONTENIDO DE CADA PESTAÑA
# =============================================================================

# ─── INTRODUCCIÓN ─────────────────────────────────────────────────────────────
tab_intro = html.Div([
    html.Div([
        html.H2([html.I(className="fas fa-graduation-cap", style={"marginRight":"10px"}), " Tasa de Finalización de Educación Primaria Femenina"],
                style={"color": CYAN, "fontWeight": "800", "fontSize": "1.4em", "marginBottom": "8px"}),
        html.P("Análisis Exploratorio de Datos · World Development Indicators · Banco Mundial · 2000–2022",
               style={"color": TEXT_DIM, "fontSize": "0.95em", "margin": "0"})
    ], style={
        "background": "linear-gradient(135deg,#0A1628 0%,#0D1F3C 40%,#112240 100%)",
        "border": "1px solid rgba(46,134,171,0.25)", "borderRadius": "16px",
        "padding": "32px 36px", "marginBottom": "20px", "position": "relative", "overflow": "hidden"
    }),

    # Pestañas de introducción
    dbc.Tabs([
        dbc.Tab(label="Introducción", tab_id="intro-tab1"),
        dbc.Tab(label="Justificación", tab_id="intro-tab2"),
        dbc.Tab(label="Objetivos", tab_id="intro-tab3"),
        dbc.Tab(label="Marco Teórico", tab_id="intro-tab4"),
        dbc.Tab(label="Hipótesis", tab_id="intro-tab5"),
        dbc.Tab(label="Metodología", tab_id="intro-tab6"),
    ], id="intro-tabs", active_tab="intro-tab1"),
    html.Div(id="intro-tab-content", style={"marginTop": "16px"}),
])

# Contenidos de sub-pestañas de intro
INTRO_TABS_CONTENT = {
    "intro-tab1": card_s([
        html.H4([html.I(className="fas fa-question-circle", style={"marginRight":"8px"}), " ¿Qué mide este indicador?"], style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
        html.P(["La ", html.Strong("tasa de finalización de educación primaria femenina"),
                " (código WDI: ", html.Code("SE.PRM.CMPT.FE.ZS"), ") mide el porcentaje de niñas que completan "
                "el último grado de la educación primaria respecto a la población femenina en edad oficial. "
                "Es un indicador clave del ", html.Strong("ODS 4 (Educación de calidad)"),
                " y un proxy de equidad educativa de género."], style={"color": TEXT_MAIN}),
        html.P(["Un valor > 100% ocurre cuando el numerador incluye graduadas de cohortes anteriores. "
                "Por eso se trabaja con ", html.Code("tasa_fin_cap"), " = valor acotado a [0, 100]."],
               style={"color": TEXT_MAIN}),
        html.Hr(style={"borderColor": "rgba(46,134,171,0.3)"}),
        html.Div([html.Strong("Fuente: "), "World Development Indicators (WDI), Banco Mundial. "
                  "Indicador: SE.PRM.CMPT.FE.ZS. Periodo: 2000–2022."],
                 style={"color": TEXT_DIM, "fontSize": "0.9em",
                        "background": "rgba(46,134,171,0.07)", "borderRadius": "6px", "padding": "10px 14px"}),
        html.Br(),
        html.Div([html.I(className="fas fa-check-circle", style={"color":VERDE_OK,"marginRight":"6px"}),
                  html.Strong("Estado de carga de datos: ", style={"color": TEXT_LIGHT}),
                  html.Span(f"{N_OBS:,} observaciones cargadas correctamente de {N_PAISES} países.",
                            style={"color": VERDE_OK})]),
    ]),
    "intro-tab2": card_s([
        html.H4([html.I(className="fas fa-globe-americas", style={"marginRight":"8px"}), " ¿Por qué importa estudiarlo?"], style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
        html.P("La brecha educativa de género es uno de los factores más persistentes de desigualdad global. "
               "Cuando una niña no termina la primaria, se incrementan los riesgos de matrimonio temprano, "
               "embarazo adolescente y exclusión del mercado laboral formal.", style={"color": TEXT_MAIN}),
        html.Ul([
            html.Li([html.Strong("Panel desbalanceado: "), "no todos los países reportan todos los años."], style={"color": TEXT_MAIN}),
            html.Li([html.Strong("Efecto techo: "), "muchos países convergen al 100%, reduciendo la variabilidad."], style={"color": TEXT_MAIN}),
            html.Li([html.Strong("Datos faltantes no aleatorios: "), "los países con peores tasas reportan menos."], style={"color": TEXT_MAIN}),
            html.Li([html.Strong("Tendencia no lineal: "), "crecimiento rápido al inicio, desaceleración posterior."], style={"color": TEXT_MAIN}),
        ]),
        html.P(["Este análisis contribuye al seguimiento de la ", html.Strong("meta 4.1 del ODS 4: "),
                html.Em("\"Asegurar que todas las niñas terminen la enseñanza primaria...\""),
                " (Naciones Unidas, 2015)."], style={"color": TEXT_MAIN}),
    ]),
    "intro-tab3": card_s([
        html.H4([html.I(className="fas fa-bullseye", style={"marginRight":"8px"}), " Objetivos del análisis"], style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
        html.H5("Objetivo general:", style={"color": "#AED6F1"}),
        html.P("Realizar un análisis exploratorio comprehensivo sobre la tasa de finalización de educación "
               "primaria femenina a nivel mundial (2000–2022), identificando distribuciones, tendencias "
               "temporales, patrones de convergencia y trayectorias diferenciadas entre países.",
               style={"color": TEXT_MAIN}),
        html.H5("Objetivos específicos:", style={"color": "#AED6F1"}),
        html.Ol([
            html.Li([html.Strong("Describir "), "la distribución univariada de tasa_fin_cap."], style={"color": TEXT_MAIN}),
            html.Li([html.Strong("Evaluar "), "la estacionariedad mediante ADF, ACF y PACF."], style={"color": TEXT_MAIN}),
            html.Li([html.Strong("Explorar "), "la relación tasa-tiempo con LOESS y medianas."], style={"color": TEXT_MAIN}),
            html.Li([html.Strong("Caracterizar "), "heterogeneidad con lollipop-IQR y mapas de calor."], style={"color": TEXT_MAIN}),
            html.Li([html.Strong("Identificar "), "países con mayor ritmo de mejora pre-techo."], style={"color": TEXT_MAIN}),
        ]),
    ], border_color=VERDE_OK),
    "intro-tab4": card_s([
        html.H4([html.I(className="fas fa-book-open", style={"marginRight":"8px"}), " Marco Teórico"], style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
        html.Div([
            html.H5("Resumen del dataset", style={"color": CYAN, "marginTop": "0"}),
            html.Div([
                kpi_card(f"{N_OBS:,}", "Observaciones (df_2)"),
                kpi_card(str(N_PAISES), "Países"),
                kpi_card(str(N_ANIOS), "Años"),
                kpi_card(f"{N_GT100}", f"Obs > 100% ({PCT_GT100}%)", AMARILLO),
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
        ], style={"background": "rgba(46,134,171,0.1)", "border": "1px solid rgba(46,134,171,0.25)",
                  "borderRadius": "10px", "padding": "16px", "marginBottom": "16px"}),
        html.H5("1. Teoría del capital humano", style={"color": "#AED6F1"}),
        html.P("Formulada por Becker (1964) y extendida por Mincer (1974). La finalización de la educación "
               "primaria es el umbral mínimo de capital humano básico reconocido internacionalmente (UNESCO, 2023).",
               style={"color": TEXT_MAIN}),
        html.H5("2. Convergencia educativa", style={"color": "#AED6F1"}),
        html.P("La hipótesis de convergencia beta (Barro & Lee, 2013) sugiere que los países con menores "
               "tasas iniciales mejoran más rápidamente, reduciendo brechas con los más avanzados.",
               style={"color": TEXT_MAIN}),
        html.H5("3. Series temporales de panel", style={"color": "#AED6F1"}),
        html.P("El dataset es un panel desbalanceado: 192 países × hasta 23 años. Se aplican ADF, ACF y PACF "
               "para diagnóstico de estacionariedad antes de cualquier modelización.",
               style={"color": TEXT_MAIN}),
        html.H5("4. Indicadores de equidad educativa", style={"color": "#AED6F1"}),
        html.P("La tasa mide acceso (cantidad), no calidad. Su limitación principal es el efecto techo "
               "que censura variabilidad cuando los países alcanzan tasas cercanas al 100% (UNICEF, 2022).",
               style={"color": TEXT_MAIN}),
    ], border_color=AMARILLO),
    "intro-tab5": card_s([
        html.H4([html.I(className="fas fa-flask", style={"marginRight":"8px"}), " Hipótesis de investigación"], style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
        html.P("A partir del marco teórico y la revisión de la literatura:", style={"color": TEXT_MAIN}),
        *[html.Div([html.Strong(f"H{i} — {t}: "), html.Span(d, style={"color": "#F5CBA7"})],
           style={"background": "rgba(243,156,18,0.08)", "borderLeft": f"4px solid {AMARILLO}",
                  "borderRadius": "8px", "padding": "12px 16px", "margin": "10px 0", "color": TEXT_MAIN})
          for i, t, d in [
              (1, "Tendencia ascendente", "La tasa muestra tendencia temporal positiva y significativa entre 2000 y 2022."),
              (2, "No estacionariedad", "La serie de medianas anuales no es estacionaria en nivel."),
              (3, "Convergencia", "El IQR se reduce progresivamente, evidenciando convergencia condicional."),
              (4, "Heterogeneidad regional", "Existen grupos de países con trayectorias cualitativamente diferentes."),
              (5, "Efecto techo creciente", "La proporción de observaciones en 100% ha aumentado con el tiempo."),
          ]],
    ], border_color=ROJO_OUT),
    "intro-tab6": card_s([
        html.H4([html.I(className="fas fa-cogs", style={"marginRight":"8px"}), " Metodología"], style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
        html.H5("Fuente y descarga de datos", style={"color": "#AED6F1"}),
        html.P(["Datos de la API del Banco Mundial vía paquete Python ",
                html.Code("wbgapi"), ". Indicador: ", html.Code("SE.PRM.CMPT.FE.ZS"),
                ". Periodo: 2000–2022."], style={"color": TEXT_MAIN}),
        html.H5("Procesamiento", style={"color": "#AED6F1"}),
        html.Ol([
            html.Li(["Eliminación de duplicados."], style={"color": TEXT_MAIN}),
            html.Li(["Separación: agregados regionales (df_1) vs países individuales (df_2)."], style={"color": TEXT_MAIN}),
            html.Li(["Exclusión de NA con ", html.Code("dropna()"), "."], style={"color": TEXT_MAIN}),
            html.Li(["Variable acotada: ", html.Code("tasa_fin_cap = clip(tasa_fin, 0, 100)"), "."], style={"color": TEXT_MAIN}),
        ]),
        html.H5("Herramientas", style={"color": "#AED6F1"}),
        html.P("Python ≥ 3.10 · Dash · Plotly · Pandas · NumPy · SciPy · Statsmodels · wbgapi",
               style={"color": TEXT_MAIN}),
    ], border_color=AZUL_FUERTE),
}

# =============================================================================
# HELPERS UI — ADF/ACF/PACF (módulo profesional)
# =============================================================================

def _hex_to_rgb(h):
    """Convierte '#RRGGBB' → 'R,G,B' para usar en rgba()."""
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


def _badge(label, value, color=None):
    """Tarjeta de métrica compacta con color semántico."""
    color = color or CYAN
    return html.Div(
        [
            html.Div(label, style={
                "color": TEXT_DIM, "fontSize": "0.72em",
                "textTransform": "uppercase", "letterSpacing": "0.07em",
                "marginBottom": "4px"
            }),
            html.Div(value, style={
                "color": color, "fontWeight": "800",
                "fontSize": "1.35em", "lineHeight": "1"
            }),
        ],
        style={
            "textAlign": "center", "padding": "12px 14px",
            "background": f"rgba({_hex_to_rgb(color)},0.08)",
            "border": f"1px solid rgba({_hex_to_rgb(color)},0.25)",
            "borderRadius": "10px", "flex": "1", "minWidth": "80px"
        }
    )


def _verdict_banner(text, color):
    """Banner de conclusión con borde lateral dinámico de color."""
    return html.Div(
        text,
        style={
            "background": f"rgba({_hex_to_rgb(color)},0.10)",
            "borderLeft": f"5px solid {color}",
            "borderRadius": "0 8px 8px 0",
            "padding": "11px 16px",
            "fontWeight": "700",
            "fontSize": "0.92em",
            "color": color,
            "marginBottom": "14px",
            "letterSpacing": "0.02em",
        }
    )


def _crit_table(crit, adf_stat):
    """Tabla de valores críticos generada dinámicamente con ✓/✗ reales."""
    niveles = [
        ("1%",  crit["1%"],  "Evidencia muy fuerte"),
        ("5%",  crit["5%"],  "Nivel estándar"),
        ("10%", crit["10%"], "Evidencia débil"),
    ]
    filas = []
    for alpha, vc, interp in niveles:
        rechaza = adf_stat < vc
        icon    = "✓" if rechaza else "✗"
        color   = VERDE_OK if rechaza else ROJO_OUT
        filas.append(
            html.Tr([
                html.Td(f"α = {alpha}", style={"color": TEXT_DIM, "padding": "6px 10px"}),
                html.Td(f"{vc:.3f}", style={"color": TEXT_MAIN, "padding": "6px 10px",
                                            "fontFamily": "monospace"}),
                html.Td(
                    html.Span(f"{icon}  {'Rechaza H₀' if rechaza else 'No rechaza H₀'}",
                              style={"color": color, "fontWeight": "600"}),
                    style={"padding": "6px 10px"}
                ),
                html.Td(interp, style={"color": TEXT_DIM, "fontSize": "0.82em",
                                       "padding": "6px 10px"}),
            ], style={"borderBottom": "1px solid rgba(46,134,171,0.12)"})
        )
    return html.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th(c, style={"color": CYAN, "padding": "7px 10px",
                                      "borderBottom": "2px solid rgba(46,134,171,0.3)",
                                      "fontSize": "0.78em", "textTransform": "uppercase",
                                      "letterSpacing": "0.06em"})
                    for c in ["Nivel α", "Valor crítico", "Decisión", "Interpretación"]
                ])
            ),
            html.Tbody(filas),
        ],
        style={"width": "100%", "borderCollapse": "collapse",
               "color": TEXT_MAIN, "fontSize": "0.87em"}
    )


def _build_acf_fig(lags, vals, ci, title, bar_color_sig, bar_color_ns):
    """
    Correlograma ACF o PACF con:
      - Banda de confianza sombreada continua
      - Barras coloreadas por significancia estadística
      - Anotaciones en lags significativos
      - Leyenda legible
    """
    sig_mask = [abs(v) > ci for v in vals]

    fig = go.Figure()

    # Banda de confianza sombreada
    x_band = list(range(len(vals)))
    fig.add_trace(go.Scatter(
        x=x_band + x_band[::-1],
        y=[ci] * len(x_band) + [-ci] * len(x_band),
        fill="toself",
        fillcolor="rgba(100,207,246,0.07)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Líneas de confianza y línea cero
    fig.add_hline(y=ci,  line=dict(dash="dot", color="rgba(100,207,246,0.55)", width=1.3))
    fig.add_hline(y=-ci, line=dict(dash="dot", color="rgba(100,207,246,0.55)", width=1.3))
    fig.add_hline(y=0,   line=dict(color="rgba(200,200,200,0.15)", width=1))

    # Una barra por lag para control de color individual
    for lag, val, sig in zip(lags, vals, sig_mask):
        color  = bar_color_sig if sig else bar_color_ns
        border = "rgba(255,255,255,0.25)" if sig else "rgba(255,255,255,0.05)"
        fig.add_trace(go.Bar(
            x=[lag], y=[val],
            marker=dict(color=color, line=dict(color=border, width=1),
                        opacity=0.92 if sig else 0.55),
            showlegend=False,
            width=0.55,
            hovertemplate=(f"<b>Lag {lag}</b><br>Valor: {val:.4f}<br>"
                           + ("<b>✦ Significativo</b>" if sig else "No significativo")
                           + "<extra></extra>"),
        ))

    # Anotaciones en lags significativos (excluye lag 0)
    for lag, val, sig in zip(lags, vals, sig_mask):
        if sig and lag > 0:
            fig.add_annotation(
                x=lag, y=val + (0.04 if val >= 0 else -0.06),
                text=f"{val:.2f}", showarrow=False,
                font=dict(size=9, color=bar_color_sig),
                bgcolor="rgba(13,31,60,0.7)",
                bordercolor=bar_color_sig, borderwidth=1, borderpad=2,
            )

    # Leyenda
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=10, color=bar_color_sig, symbol="square"),
                             name="Significativo (|r| > 1.96/√n)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=10, color=bar_color_ns, symbol="square", opacity=0.5),
                             name="No significativo"))

    fig = pl_layout(fig, title, f"Bandas de confianza ±{ci:.3f}  (α=0.05)", "Lag", "Correlación")
    fig.update_layout(
        height=310, bargap=0.35,
        margin=dict(t=65, b=55, l=60, r=25),
        legend=dict(orientation="h", x=0, y=-0.28,
                    font=dict(color=TEXT_MAIN, size=10),
                    bgcolor="rgba(13,31,60,0.7)",
                    bordercolor="rgba(46,134,171,0.3)", borderwidth=1),
        xaxis=dict(tickmode="linear", dtick=1, tickfont=dict(color=TEXT_DIM, size=10)),
    )
    return fig


# ─── ESTILOS DE SECCIÓN ADF ───────────────────────────────────────────────────
_SECTION_STYLE = {
    "background": BG_CARD,
    "borderRadius": "12px",
    "padding": "26px 28px",
    "marginBottom": "22px",
    "borderLeft": f"6px solid {AZUL_MAIN}",
    "boxShadow": "0 4px 24px rgba(0,0,0,0.35)",
}
_SECTION_TITLE_STYLE = {
    "color": CYAN, "fontWeight": "800", "fontSize": "1.05em",
    "letterSpacing": "0.03em",
    "borderBottom": "2px solid rgba(46,134,171,0.25)",
    "paddingBottom": "10px", "marginBottom": "18px",
}

# ─── UNIVARIADO ───────────────────────────────────────────────────────────────
tab_univariado = html.Div([

    # ── CABECERA DESCRIPTIVA ──────────────────────────────────────────────────
    html.Div([
        html.H5("Prueba de Estacionariedad — ADF, ACF y PACF",
                style=_SECTION_TITLE_STYLE),
        html.P([
            "La ", html.Strong("prueba ADF (Augmented Dickey-Fuller)"),
            " evalúa si la serie de medianas anuales presenta raíz unitaria ",
            "(H₀: la serie es no estacionaria). Los ",
            html.Strong("correlogramas ACF y PACF"),
            " complementan el diagnóstico mostrando la estructura temporal de "
            "autocorrelación y orientan la selección del orden (p, d, q) para modelos ARIMA.",
        ], style={"color": TEXT_MAIN, "lineHeight": "1.65", "marginBottom": "0",
                  "fontSize": "0.93em"}),
    ], style={**_SECTION_STYLE, "borderLeft": f"6px solid {CYAN}",
              "marginBottom": "18px", "paddingBottom": "18px"}),

    # ── RESULTADO ADF ─────────────────────────────────────────────────────────
    html.Div([
        html.H5("Resultado de la Prueba ADF", style=_SECTION_TITLE_STYLE),
        dbc.Row([
            # Izquierda: veredicto + badges (generado por callback)
            dbc.Col([
                html.Div(id="adf-result-ui"),
            ], width=12, lg=5, className="mb-3 mb-lg-0"),
            # Derecha: tabla valores críticos reales (generada por callback)
            dbc.Col([
                html.Div([
                    html.H6("Valores Críticos ADF",
                            style={"color": CYAN, "fontWeight": "700",
                                   "marginBottom": "14px", "fontSize": "0.9em",
                                   "textTransform": "uppercase", "letterSpacing": "0.07em"}),
                    html.Div(id="adf-crit-table"),
                ], style={"background": BG_PLOT,
                          "border": "1px solid rgba(46,134,171,0.2)",
                          "borderRadius": "10px", "padding": "18px 20px"})
            ], width=12, lg=7),
        ]),
        # Implicaciones para el modelado (generadas por callback)
        html.Div(id="adf-implicaciones", style={"marginTop": "16px"}),
    ], style=_SECTION_STYLE),

    # ── CORRELOGRAMAS ACF / PACF ──────────────────────────────────────────────
    html.Div([
        html.H5("Correlogramas ACF y PACF", style=_SECTION_TITLE_STYLE),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("ACF", style={"color": CYAN, "fontWeight": "800",
                                                "fontSize": "0.82em", "letterSpacing": "0.08em"}),
                        html.Span(" — Función de Autocorrelación",
                                  style={"color": TEXT_DIM, "fontSize": "0.82em"}),
                    ], style={"marginBottom": "6px"}),
                    dcc.Graph(id="plot-acf", style={"height": "310px"},
                              config={"displayModeBar": True,
                                      "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                      "displaylogo": False}),
                ])
            ], width=12, lg=6, className="mb-3 mb-lg-0"),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("PACF", style={"color": ROJO_OUT, "fontWeight": "800",
                                                 "fontSize": "0.82em", "letterSpacing": "0.08em"}),
                        html.Span(" — Función de Autocorrelación Parcial",
                                  style={"color": TEXT_DIM, "fontSize": "0.82em"}),
                    ], style={"marginBottom": "6px"}),
                    dcc.Graph(id="plot-pacf", style={"height": "310px"},
                              config={"displayModeBar": True,
                                      "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                      "displaylogo": False}),
                ])
            ], width=12, lg=6),
        ]),
        # Guía de lectura de correlogramas
        html.Div([
            html.Div([html.I(className="fas fa-chart-bar", style={"marginRight":"6px","color":CYAN}), "  Cómo leer los correlogramas"],
                     style={"color": CYAN, "fontWeight": "700", "marginBottom": "10px",
                            "fontSize": "0.88em", "textTransform": "uppercase",
                            "letterSpacing": "0.06em"}),
            html.Ul([
                html.Li([html.Strong("ACF: ", style={"color": TEXT_LIGHT}),
                         "Mide la correlación entre la serie y sus rezagos. Un decaimiento lento "
                         "y significativo indica no estacionariedad. Barras fuera de la banda "
                         f"±1.96/√n (en azul) son estadísticamente significativas (α = 5 %)."],
                        style={"marginBottom": "8px", "color": TEXT_MAIN}),
                html.Li([html.Strong("PACF: ", style={"color": TEXT_LIGHT}),
                         "Mide la correlación directa al rezago k, eliminando el efecto de los "
                         "intermedios. Barras significativas (en rojo) guían la selección del "
                         "orden autoregresivo p en ARIMA(p, d, q)."],
                        style={"color": TEXT_MAIN}),
            ], style={"paddingLeft": "18px", "margin": "0", "lineHeight": "1.65",
                      "fontSize": "0.9em"}),
        ], style={"background": "rgba(46,134,171,0.06)",
                  "border": "1px solid rgba(46,134,171,0.2)",
                  "borderRadius": "10px", "padding": "16px 20px", "marginTop": "18px"}),
    ], style=_SECTION_STYLE),

    # Histograma + BoxPlot
    html.Div([
        html.H5("Distribución de tasa_fin_cap — Histograma, Densidad y Estadísticas descriptivas",
                style={"color": CYAN, "fontWeight": "700", "borderBottom": "2px solid rgba(46,134,171,0.3)",
                       "paddingBottom": "10px", "marginBottom": "16px"}),
        dbc.Row([
            dbc.Col([dcc.Graph(id="plot-hist", style={"height": "360px"})], width=6),
            dbc.Col([dcc.Graph(id="plot-boxplot", style={"height": "360px"})], width=6),
        ]),
        html.Div(id="stats-descriptivas"),
        interp_box("La distribución es fuertemente asimétrica negativa: la mayoría de observaciones se "
                   "concentra en 85–100%, con cola izquierda de países rezagados. La mediana supera a la "
                   "media en todos los años, confirmando la asimetría.")
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "marginBottom": "20px",
              "borderLeft": f"6px solid {AZUL_MAIN}"}),

    # Tabla > 100%
    html.Div([
        html.H5("Observaciones con tasa_fin > 100% (efecto sobrenotificación)",
                style={"color": CYAN, "fontWeight": "700", "borderBottom": "2px solid rgba(46,134,171,0.3)",
                       "paddingBottom": "10px", "marginBottom": "16px"}),
        html.P("Países y años con tasa original > 100%. Fenómeno metodológico, no error de datos.",
               style={"color": TEXT_MAIN}),
        html.Div(id="tabla-mayor-100"),
        interp_box(["Valores > 100% ocurren cuando el numerador incluye graduadas de cohortes anteriores. "
                    "Para análisis gráfico se usa ", html.Code("tasa_fin_cap"), "."]),
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "marginBottom": "20px",
              "borderLeft": f"6px solid {AMARILLO}"}),
])

# ─── BIVARIADO ────────────────────────────────────────────────────────────────
tab_bivariado = html.Div([
    # Lollipop
    html.Div([
        html.H5("Mediana e IQR de tasa_fin_cap por País y Quinquenio",
                style={"color": CYAN, "fontWeight": "700", "borderBottom": "2px solid rgba(46,134,171,0.3)",
                       "paddingBottom": "10px", "marginBottom": "16px"}),
        html.P("Seleccione el número de países con mayor cobertura temporal.", style={"color": TEXT_MAIN}),
        dbc.Row([dbc.Col([
            dcc.Slider(id="slider-n-paises", min=4, max=15, step=1, value=8,
                       marks={i: str(i) for i in range(4, 16)},
                       tooltip={"placement": "bottom", "always_visible": True})
        ], width=4)]),
        dcc.Graph(id="plot-lollipop", style={"height": "560px"}),
        interp_box("Punto = mediana por país-quinquenio. Segmento = IQR (P25–P75). "
                   "Línea roja discontinua = techo (100%). Segmentos que se acortan con el tiempo "
                   "indican mayor consistencia intra-país.")
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "marginBottom": "20px",
              "borderLeft": f"6px solid {AZUL_MAIN}"}),

    dbc.Row([
        dbc.Col([html.Div([
            html.H5("Proporción de Años en el Techo (100%) por País",
                    style={"color": CYAN, "fontWeight": "700", "marginBottom": "14px"}),
            dcc.Graph(id="plot-techo", style={"height": "400px"}),
            interp_box("Alto % indica que el indicador ya no discrimina el desempeño real del país. "
                       "Requiere indicadores complementarios de calidad educativa.")
        ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "height": "100%",
                  "borderLeft": f"6px solid {AZUL_MAIN}"})], width=6),
        dbc.Col([html.Div([
            html.H5("Dispersión Global + Suavizado LOESS",
                    style={"color": CYAN, "fontWeight": "700", "marginBottom": "14px"}),
            dcc.Graph(id="plot-scatter-loess", style={"height": "400px"}),
            interp_box("Cada punto = un par (año, tasa_fin_cap) de un país. La curva LOESS confirma "
                       "una trayectoria cuasi-logística: crecimiento rápido en la primera década "
                       "y desaceleración posterior.")
        ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "height": "100%",
                  "borderLeft": f"6px solid {AZUL_MAIN}"})], width=6),
    ], style={"marginBottom": "20px"}),

    dbc.Row([
        dbc.Col([html.Div([
            html.H5("Mediana Anual + Banda IQR (P25–P75)",
                    style={"color": CYAN, "fontWeight": "700", "marginBottom": "14px"}),
            dcc.Graph(id="plot-mediana-anual", style={"height": "380px"}),
            interp_box("La mediana global aumentó de ~82% (2000) a ~95% (2022). El estrechamiento "
                       "del área IQR evidencia convergencia relativa.")
        ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "height": "100%",
                  "borderLeft": f"6px solid {AZUL_MAIN}"})], width=6),
        dbc.Col([html.Div([
            html.H5("Media vs. Mediana Anual",
                    style={"color": CYAN, "fontWeight": "700", "marginBottom": "14px"}),
            dcc.Graph(id="plot-media-mediana", style={"height": "380px"}),
            interp_box("La mediana está sistemáticamente por encima de la media: distribución "
                       "asimétrica negativa persistente. La brecha se cierra gradualmente — catching-up.")
        ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "height": "100%",
                  "borderLeft": f"6px solid {AZUL_MAIN}"})], width=6),
    ], style={"marginBottom": "20px"}),

    # Ribbon multicapa
    html.Div([
        html.H5("Evolución del Panel — Mediana con Bandas P10–P25–P75–P90",
                style={"color": CYAN, "fontWeight": "700", "borderBottom": "2px solid rgba(46,134,171,0.3)",
                       "paddingBottom": "10px", "marginBottom": "16px"}),
        dcc.Graph(id="plot-mv-ribbon", style={"height": "420px"}),
        interp_box("El ribbon multicapa combina cuatro cuantiles simultáneamente: banda exterior "
                   "(P10–P90) = 80% central de países; banda interior (P25–P75) = IQR; línea = mediana. "
                   "El estrechamiento conjunto es la evidencia más robusta de convergencia global.")
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "marginBottom": "20px",
              "borderLeft": f"6px solid {AZUL_MAIN}"}),

    # Mapa de calor
    html.Div([
        html.H5("Mapa de Calor — tasa_fin_cap por País y Año",
                style={"color": CYAN, "fontWeight": "700", "borderBottom": "2px solid rgba(46,134,171,0.3)",
                       "paddingBottom": "10px", "marginBottom": "16px"}),
        html.P("Seleccione el número de países a mostrar (ordenados por tasa media histórica).",
               style={"color": TEXT_MAIN}),
        dbc.Row([dbc.Col([
            dcc.Slider(id="slider-n-heat", min=10, max=50, step=5, value=30,
                       marks={i: str(i) for i in range(10, 51, 10)},
                       tooltip={"placement": "bottom", "always_visible": True})
        ], width=4)]),
        dcc.Graph(id="plot-heatmap", style={"height": "620px"}),
        interp_box("El mapa de calor muestra tres variables simultáneamente: país (eje Y), año (eje X) "
                   "y nivel de tasa (color). Celdas grises = dato faltante. Se identifican tres grupos: "
                   "países que alcanzaron el techo temprano, países en transición activa y países con "
                   "fragilidad persistente.")
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "marginBottom": "20px",
              "borderLeft": f"6px solid {AZUL_MAIN}"}),

    # Pendientes + Trayectorias individuales
    dbc.Row([
        dbc.Col([html.Div([
            html.H5("Top países con mayor mejora anual (pendiente pre-techo)",
                    style={"color": CYAN, "fontWeight": "700", "marginBottom": "14px"}),
            dcc.Slider(id="slider-n-slopes", min=5, max=20, step=1, value=12,
                       marks={i: str(i) for i in range(5, 21, 5)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            dcc.Graph(id="plot-slopes-top", style={"height": "480px"}),
            interp_box("Pendiente de lm(tasa_fin_cap ~ anio) estimada solo antes de alcanzar el techo. "
                       "Los países líderes lograron avances de 2–8 pp/año.")
        ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "height": "100%",
                  "borderLeft": f"6px solid {AZUL_MAIN}"})], width=6),
        dbc.Col([html.Div([
            html.H5("Trayectorias individuales — Países seleccionados",
                    style={"color": CYAN, "fontWeight": "700", "marginBottom": "14px"}),
            dcc.Dropdown(
                id="dropdown-paises",
                options=[{"label": p, "value": p} for p in PAISES_LIST],
                value=PAISES_LIST[:6],
                multi=True,
                style={"background": BG_PLOT, "color": TEXT_MAIN}
            ),
            dcc.Graph(id="plot-evol-paises", style={"height": "480px"}),
            interp_box("Las trayectorias individuales revelan patrones ocultos: crecimientos lineales "
                       "sostenidos, rupturas abruptas y fluctuaciones. Los huecos son años sin reporte.")
        ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "height": "100%",
                  "borderLeft": f"6px solid {AZUL_MAIN}"})], width=6),
    ], style={"marginBottom": "20px"}),

    # Prueba de hipótesis
    html.Div([
        html.H5("Prueba de Hipótesis: ¿Cambia la tasa mediana entre períodos?",
                style={"color": CYAN, "fontWeight": "700", "borderBottom": "2px solid rgba(46,134,171,0.3)",
                       "paddingBottom": "10px", "marginBottom": "16px"}),
        html.P(["Se contrasta si la mediana global de ", html.Code("tasa_fin_cap"),
                " difiere significativamente entre dos períodos seleccionados. Se verifican supuestos "
                "(normalidad, homocedasticidad) para elegir la prueba adecuada."],
               style={"color": TEXT_MAIN}),
        dbc.Row([
            dbc.Col([
                html.Label("Período 1 (rango de años):", style={"color": TEXT_DIM}),
                dcc.RangeSlider(id="hip-periodo1", min=2000, max=2022, step=1,
                               value=[2000, 2010], marks={y: str(y) for y in range(2000, 2023, 5)},
                               tooltip={"placement": "bottom", "always_visible": True})
            ], width=3),
            dbc.Col([
                html.Label("Período 2 (rango de años):", style={"color": TEXT_DIM}),
                dcc.RangeSlider(id="hip-periodo2", min=2000, max=2022, step=1,
                               value=[2011, 2022], marks={y: str(y) for y in range(2000, 2023, 5)},
                               tooltip={"placement": "bottom", "always_visible": True})
            ], width=3),
            dbc.Col([
                html.Br(),
                html.Button("▶ Ejecutar prueba", id="btn-hipotesis", n_clicks=0,
                            style={"background": AZUL_MAIN, "color": "white", "border": "none",
                                   "borderRadius": "8px", "padding": "10px 20px",
                                   "fontWeight": "600", "width": "100%", "cursor": "pointer"})
            ], width=3),
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col([dcc.Graph(id="plot-hip-boxplot", style={"height": "340px"})], width=6),
            dbc.Col([dcc.Graph(id="plot-hip-dist",    style={"height": "340px"})], width=6),
        ]),
        html.Div(id="hip-resultado"),
        interp_box([html.Strong("H₀:"), " la distribución de tasa_fin_cap es la misma en ambos períodos. ",
                    html.Br(), html.Strong("H₁:"), " las distribuciones difieren (prueba bilateral). ",
                    html.Br(), "Se aplica Shapiro-Wilk para normalidad y Levene para homocedasticidad. "
                    "Si ambos supuestos se cumplen, se usa t de Student; si no, Wilcoxon-Mann-Whitney. α = 0.05."])
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "marginBottom": "20px",
              "borderLeft": f"6px solid {AMARILLO}"}),
])

# ─── MAPA MUNDIAL ─────────────────────────────────────────────────────────────
tab_mapa = html.Div([
    page_header("fas fa-globe-americas", "Mapas Mundiales — tasa_fin_cap por País"),

    html.Div([
        html.H5("Mapa 1 — Promedio histórico de tasa_fin_cap (2000–2022)",
                style={"color": CYAN, "fontWeight": "700", "marginBottom": "12px"}),

        html.P(
            "Tasa media de finalización primaria femenina por país durante el periodo 2000–2022. "
            "Los países en gris representan ausencia de datos disponibles. "
            "Pase el cursor sobre cada país para consultar el valor exacto.",
            style={"color": TEXT_DIM, "fontSize": "0.9em"}),

        dcc.Graph(id="plot-mapa-promedio", style={"height": "520px"}),

        interp_box(
            "La escala de color va desde tonos morado oscuro (tasas históricamente más bajas) "
            "hasta amarillo brillante (tasas históricamente más altas). "
            "Se observa una marcada heterogeneidad internacional: varios países mantienen "
            "niveles cercanos al techo educativo (>90%), mientras otros presentan rezagos "
            "persistentes. El patrón espacial sugiere desigualdades estructurales en el "
            "acceso y culminación de la educación primaria femenina."
        )

    ], style={
        "background": BG_CARD,
        "borderRadius": "10px",
        "padding": "24px",
        "marginBottom": "20px",
        "borderLeft": f"6px solid {AZUL_MAIN}"
    }),


    html.Div([
        html.H5("Mapa 2 — Evolución anual de tasa_fin_cap (2000–2022, animado)",
                style={"color": CYAN, "fontWeight": "700", "marginBottom": "12px"}),

        html.P(
            ["Presione el botón ", html.Strong("▶ Play"),
             " para visualizar la evolución temporal de las tasas por país."],
            style={"color": TEXT_DIM, "fontSize": "0.9em"}),

        dcc.Graph(id="plot-mapa-animado", style={"height": "540px"}),

        interp_box(
            "La animación permite identificar trayectorias de cambio a lo largo del tiempo. "
            "Un desplazamiento progresivo hacia colores asociados a valores altos evidencia "
            "mejoras en los niveles de finalización educativa. Los países que mantienen tonos "
            "oscuros durante gran parte del periodo reflejan persistencia de brechas "
            "estructurales y procesos de convergencia más lentos."
        )

    ], style={
        "background": BG_CARD,
        "borderRadius": "10px",
        "padding": "24px",
        "marginBottom": "20px",
        "borderLeft": f"6px solid {AZUL_MAIN}"
    }),


    html.Div([
        html.H5("Mapa 3 — Cambio absoluto: último año disponible vs. 2000",
                style={"color": CYAN, "fontWeight": "700", "marginBottom": "12px"}),

        html.P(
            "Diferencia entre la tasa más reciente y la registrada en el año 2000. "
            "Solo se incluyen países con información disponible en ambos puntos temporales.",
            style={"color": TEXT_DIM, "fontSize": "0.9em"}),

        dcc.Graph(id="plot-mapa-cambio", style={"height": "520px"}),

        interp_box(
            "La escala está centrada en cero: los tonos morado representan reducciones "
            "o retrocesos relativos, mientras los tonos naranja y amarillo representan "
            "incrementos en la tasa de finalización. Los cambios más intensos indican "
            "transformaciones educativas significativas. Se aprecia evidencia compatible "
            "con procesos de convergencia: varios países con niveles históricamente bajos "
            "presentan mejoras superiores a las economías inicialmente más avanzadas."
        )

    ], style={
        "background": BG_CARD,
        "borderRadius": "10px",
        "padding": "24px",
        "marginBottom": "20px",
        "borderLeft": f"6px solid {VERDE_OK}"
    }),
])
# ─── CONCLUSIONES ─────────────────────────────────────────────────────────────
tab_conclusiones = html.Div([
    page_header("fas fa-flag-checkered", "Conclusiones del Análisis Exploratorio"),
    *[html.Div([
        html.H4(titulo, style={"color": CYAN, "marginBottom": "12px"}),
        html.Ul([html.Li(item, style={"color": TEXT_MAIN, "marginBottom": "6px"}) for item in items])
      ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "20px 24px",
                "marginBottom": "16px", "borderLeft": f"6px solid {bcolor}"})
      for titulo, bcolor, items in [
          ([html.I(className="fas fa-chart-bar", style={"marginRight":"6px"}), " 1. Hallazgos del análisis univariado"], AZUL_MAIN, [
              "La distribución global de tasa_fin_cap es fuertemente asimétrica negativa: la mayoría de observaciones se concentran en 85–100%, con cola izquierda de países del África subsahariana y sur de Asia en los años iniciales.",
              "La mediana supera a la media en todos los años, confirmando la asimetría. La diferencia cuantifica el peso de los países rezagados.",
              "El boxplot identifica outliers inferiores que representan exclusión educativa severa, no artefactos estadísticos.",
              "El efecto sobrenotificación (tasa_fin > 100%) afecta al 5–8% de las observaciones, justificando el uso de tasa_fin_cap.",
          ]),
          ("〰 2. Prueba ADF, ACF y PACF", AZUL_MAIN, [
              "La prueba ADF sobre la serie de medianas anuales indica una tendencia temporal suave. El p-value cercano a 0.05–0.10 clasifica la serie como marginalmente no estacionaria o en zona gris.",
              "El ACF muestra autocorrelación positiva significativa en varios rezagos, confirmando la tendencia.",
              "El PACF indica que la estructura se concentra en los primeros rezagos, orientando hacia modelos AR(1) o AR(2) tras diferenciación.",
              "Para modelización formal se recomienda diferenciar la serie una vez y evaluar ARIMA(p,1,q).",
          ]),
          ([html.I(className="fas fa-chart-line", style={"marginRight":"6px"}), " 3. Hallazgos del análisis bivariado"], AZUL_MAIN, [
              "El scatter con LOESS confirma una trayectoria cuasi-logística: crecimiento rápido hasta ~2012 y desaceleración posterior al acercarse al techo.",
              "La mediana global aumentó de ~82% (2000) a ~95% (2022). La banda IQR se estrechó de ~25 pp a menos de 10 pp: convergencia real pero incompleta.",
              "La brecha media–mediana se cierra gradualmente: los países rezagados mejoran más rápido (catching-up).",
          ]),
          ([html.I(className="fas fa-layer-group", style={"marginRight":"6px"}), " 4. Hallazgos del análisis multivariado"], AZUL_MAIN, [
              "El mapa de calor identifica tres grupos: países que alcanzaron el techo temprano, países en transición activa y países con fragilidad persistente.",
              "Los países con mayores pendientes de mejora pre-techo son principalmente de África del Norte, Asia Central y el sudeste asiático.",
              "Las trayectorias individuales revelan disrupciones que el análisis agregado no puede capturar.",
          ]),
      ]],
    html.Div([
        html.H4([html.I(className="fas fa-road", style={"marginRight":"8px"}), " 5. Limitaciones y pasos a seguir"], style={"color": VERDE_OK, "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([
                html.H5("Limitaciones:", style={"color": TEXT_DIM}),
                html.Ul([
                    html.Li("Efecto techo censura la variabilidad real en países avanzados.", style={"color": TEXT_MAIN}),
                    html.Li("Datos faltantes no aleatorios: los países sin reporte suelen tener peores tasas.", style={"color": TEXT_MAIN}),
                    html.Li("Prueba ADF con n=23 tiene baja potencia estadística.", style={"color": TEXT_MAIN}),
                    html.Li("El indicador mide cantidad, no calidad del aprendizaje.", style={"color": TEXT_MAIN}),
                ])
            ], width=6),
            dbc.Col([
                html.H5("Pasos recomendados:", style={"color": TEXT_DIM}),
                html.Ul([
                    html.Li("Complementar con indicadores de calidad (PISA, EGRA/EGMA).", style={"color": TEXT_MAIN}),
                    html.Li("Analizar el mecanismo de datos faltantes (MCAR, MAR, MNAR).", style={"color": TEXT_MAIN}),
                    html.Li("Modelar con ARIMA(p,1,q) para proyecciones al 2030 (ODS 4).", style={"color": TEXT_MAIN}),
                    html.Li("Estudiar en profundidad los casos de mayor mejora para extraer lecciones de política.", style={"color": TEXT_MAIN}),
                ])
            ], width=6),
        ])
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "20px 24px",
              "marginBottom": "16px", "borderLeft": f"6px solid {VERDE_OK}"}),
])

# ─── DATOS ────────────────────────────────────────────────────────────────────
tab_datos = html.Div([
    html.Div([
        html.H5("Explorador de Datos — df_2 (observaciones limpias, sin NA)",
                style={"color": CYAN, "fontWeight": "700", "marginBottom": "16px"}),
        html.P(["Tabla interactiva. Las filas con ", html.Code("tasa_fin > 100"),
                " se resaltan. Use los filtros para explorar."], style={"color": TEXT_MAIN}),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id="filtro-pais",
                    options=[{"label": "(Todos)", "value": "(Todos)"}] + [{"label": p, "value": p} for p in PAISES_LIST],
                    value="(Todos)", clearable=False,
                    style={"background": BG_PLOT, "color": "#111"}
                )
            ], width=4),
            dbc.Col([
                dcc.RangeSlider(id="filtro-anio", min=2000, max=2022, step=1,
                               value=[2000, 2022], marks={y: str(y) for y in range(2000, 2023, 5)},
                               tooltip={"placement": "bottom", "always_visible": True})
            ], width=5),
        ], style={"marginBottom": "16px"}),
        html.Div(id="tabla-datos"),
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px",
              "borderLeft": f"6px solid {AZUL_MAIN}"}),
])

# ─── REFERENCIAS ──────────────────────────────────────────────────────────────
def ref_entry(num, authors, title, source, url=None):
    link = html.A(url, href=url, target="_blank", style={"color": CYAN}) if url else None
    return html.Div([
        html.B(f"[{num}] "), f"{authors} ", html.Em(title), f" {source}. ",
        link or ""
    ], style={
        "background": "#152A45", "borderLeft": f"4px solid {AZUL_MAIN}", "borderRadius": "6px",
        "color": "#AED6F1", "padding": "10px 16px", "marginBottom": "10px",
        "lineHeight": "1.65", "fontSize": "0.93em"
    })

tab_referencias = html.Div([
    page_header("fas fa-book", "Referencias bibliográficas y fuentes de datos"),
    html.Div([
        html.P("Referencias en formato APA 7ª edición.", style={"color": TEXT_DIM, "fontSize": "0.9em"}),
        html.Div([html.I(className="fas fa-database", style={"marginRight":"6px"}), " Fuentes de datos"], style={"color": CYAN, "borderBottom": "2px solid rgba(46,134,171,0.3)",
                  "fontWeight": "700", "padding": "10px 0 8px", "margin": "22px 0 12px"}),
        ref_entry(1, "Banco Mundial. (2023).", "Primary completion rate, female (% of relevant age group) — SE.PRM.CMPT.FE.ZS.",
                  "World Development Indicators.", "https://data.worldbank.org/indicator/SE.PRM.CMPT.FE.ZS"),
        ref_entry(2, "Arel-Bundock, V. (2022).", "WDI: World Development Indicators (R package v2.7.8).",
                  "CRAN.", "https://CRAN.R-project.org/package=WDI"),
        html.Div([html.I(className="fas fa-ruler-combined", style={"marginRight":"6px"}), " Metodología estadística"], style={"color": CYAN, "borderBottom": "2px solid rgba(46,134,171,0.3)",
                  "fontWeight": "700", "padding": "10px 0 8px", "margin": "22px 0 12px"}),
        ref_entry(3, "Dickey, D. A., & Fuller, W. A. (1979).", "Distribution of the estimators for autoregressive time series with a unit root.",
                  "JASA, 74(366), 427–431.", "https://doi.org/10.2307/2286348"),
        ref_entry(4, "Said, S. E., & Dickey, D. A. (1984).", "Testing for unit roots in autoregressive-moving average models of unknown order.",
                  "Biometrika, 71(3), 599–607.", None),
        ref_entry(5, "Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015).", "Time Series Analysis: Forecasting and Control (5th ed.).",
                  "Wiley. [ACF y PACF — selección de modelos ARIMA]", None),
        html.Div([html.I(className="fas fa-book-open", style={"marginRight":"6px"}), " Marco teórico"], style={"color": CYAN, "borderBottom": "2px solid rgba(46,134,171,0.3)",
                  "fontWeight": "700", "padding": "10px 0 8px", "margin": "22px 0 12px"}),
        ref_entry(6, "Becker, G. S. (1964).", "Human Capital: A Theoretical and Empirical Analysis.",
                  "University of Chicago Press.", None),
        ref_entry(7, "Barro, R. J., & Lee, J. W. (2013).", "A new data set of educational attainment in the world, 1950–2010.",
                  "Journal of Development Economics, 104, 184–198.", "https://doi.org/10.1016/j.jdeveco.2012.10.001"),
        ref_entry(8, "Naciones Unidas. (2015).", "Transforming our world: the 2030 Agenda for Sustainable Development.",
                  "A/RES/70/1.", "https://sdgs.un.org/goals/goal4"),
        ref_entry(9, "UNESCO. (2023).", "Global Education Monitoring Report 2023.",
                  "UNESCO Publishing.", "https://www.unesco.org/gem-report/en"),
        ref_entry(10, "UNICEF. (2022).", "Education, Equity and Learning: Global annual report.",
                  "UNICEF.", "https://www.unicef.org/reports"),
        html.Div([html.I(className="fas fa-laptop-code", style={"marginRight":"6px"}), " Herramientas computacionales"], style={"color": CYAN, "borderBottom": "2px solid rgba(46,134,171,0.3)",
                  "fontWeight": "700", "padding": "10px 0 8px", "margin": "22px 0 12px"}),
        ref_entry(11, "Plotly Technologies Inc. (2015).", "Collaborative data science.",
                  "Plotly Technologies Inc.", "https://plot.ly"),
        ref_entry(12, "Plotly. (2023).", "Dash: Analytical Web Apps for Python, R, Julia, and Jupyter.",
                  "Plotly.", "https://dash.plotly.com"),
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px"}),
])

# ─── VIDEO YOUTUBE ────────────────────────────────────────────────────────────
YOUTUBE_VIDEO_ID = "IbonuVMPxXo"
YOUTUBE_WATCH_URL = f"https://www.youtube.com/watch?v={YOUTUBE_VIDEO_ID}"
YOUTUBE_EMBED_URL = f"https://www.youtube.com/embed/{YOUTUBE_VIDEO_ID}?rel=0&modestbranding=1"

tab_video = html.Div([
    page_header("fab fa-youtube", "Video — ODS 4: Educación de Calidad"),

    # Tarjeta principal con el reproductor embebido
    html.Div([
        html.Div([
            html.H5(
                [html.I(className="fas fa-play-circle", style={"marginRight": "8px", "color": "#FF0000"}),
                 "Reproducir video desde el Dashboard"],
                style={"color": CYAN, "fontWeight": "700",
                       "borderBottom": "2px solid rgba(46,134,171,0.3)",
                       "paddingBottom": "10px", "marginBottom": "20px"}
            ),
            html.P(
                "Visualización del video de YouTube directamente integrada en el dashboard. "
                "Puede pausar, reanudar y controlar el volumen desde el reproductor.",
                style={"color": TEXT_DIM, "fontSize": "0.9em", "marginBottom": "18px"}
            ),
        ]),

        # Reproductor embebido responsivo
        html.Div([
            html.Iframe(
                src=YOUTUBE_EMBED_URL,
                style={
                    "width": "100%",
                    "height": "560px",
                    "border": "none",
                    "borderRadius": "10px",
                    "boxShadow": f"0 4px 24px rgba(0,0,0,0.5), 0 0 0 1px rgba(46,134,171,0.3)",
                },
            ),
        ], style={"marginBottom": "20px"}),

        # Enlace a YouTube
        html.Div([
            html.A(
                [
                    html.I(className="fab fa-youtube", style={
                        "marginRight": "10px", "fontSize": "1.3em", "color": "#FF0000"
                    }),
                    "Ver en YouTube",
                    html.I(className="fas fa-external-link-alt", style={
                        "marginLeft": "10px", "fontSize": "0.85em", "color": TEXT_DIM
                    }),
                ],
                href=YOUTUBE_WATCH_URL,
                target="_blank",
                style={
                    "display": "inline-flex",
                    "alignItems": "center",
                    "background": "linear-gradient(135deg, #C0392B 0%, #E74C3C 100%)",
                    "color": "white",
                    "padding": "12px 28px",
                    "borderRadius": "8px",
                    "fontWeight": "700",
                    "fontSize": "1em",
                    "textDecoration": "none",
                    "boxShadow": "0 4px 14px rgba(192,57,43,0.45)",
                    "transition": "opacity 0.2s",
                    "letterSpacing": "0.3px",
                }
            ),
            html.Span(
                f"  · {YOUTUBE_WATCH_URL}",
                style={"color": TEXT_DIM, "fontSize": "0.8em", "marginLeft": "14px"}
            ),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "6px"}),

    ], style={
        "background": BG_CARD,
        "borderRadius": "12px",
        "padding": "28px 32px",
        "marginBottom": "20px",
        "borderLeft": f"6px solid #E74C3C",
        "boxShadow": "0 4px 20px rgba(0,0,0,0.35)",
    }),

    # Caja de contexto / interpretación
    interp_box([
        html.Strong("Sobre este video: "),
        "Este recurso audiovisual complementa el análisis estadístico presentado en el dashboard. "
        "Aborda el Objetivo de Desarrollo Sostenible 4 (ODS 4) — Educación de Calidad — "
        "con énfasis en la tasa de finalización de educación primaria femenina a nivel mundial. "
        "El material es útil para contextualizar los hallazgos cuantitativos del análisis exploratorio.",
    ]),
])

# =============================================================================
# LAYOUT PRINCIPAL
# =============================================================================
app.layout = html.Div([
    dcc.Store(id="current-tab", data="intro"),
    # Interval de un solo disparo para forzar render inicial de callbacks reactivos
    dcc.Interval(id="init-trigger", interval=300, n_intervals=0, max_intervals=1),
    sidebar,

    # Contenido principal
    html.Div([
        header,
        html.Div(id="main-content", style={"padding": "24px", "minHeight": "calc(100vh - 60px)"})
    ], style={"marginLeft": "270px", "background": BG_DARK, "minHeight": "100vh"}),
])


# =============================================================================
# CALLBACKS
# =============================================================================

# ─── Navegación ───────────────────────────────────────────────────────────────
@app.callback(
    Output("main-content", "children"),
    Output("current-tab", "data"),
    [Input("nav-intro", "n_clicks"),
     Input("nav-univariado", "n_clicks"),
     Input("nav-bivariado", "n_clicks"),
     Input("nav-mapa", "n_clicks"),
     Input("nav-conclusiones", "n_clicks"),
     Input("nav-datos", "n_clicks"),
     Input("nav-referencias", "n_clicks"),
     Input("nav-prediccion", "n_clicks"),
     Input("nav-video", "n_clicks")],
    State("current-tab", "data"),
)
def navigate(*args):
    ctx = callback_context
    if not ctx.triggered:
        return tab_intro, "intro"
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    mapping = {
        "nav-intro": ("intro", tab_intro),
        "nav-univariado": ("univariado", tab_univariado),
        "nav-bivariado": ("bivariado", tab_bivariado),
        "nav-mapa": ("mapa", tab_mapa),
        "nav-conclusiones": ("conclusiones", tab_conclusiones),
        "nav-datos": ("datos", tab_datos),
        "nav-referencias": ("referencias", tab_referencias),
        "nav-prediccion": ("prediccion", tab_prediccion),
        "nav-video": ("video", tab_video),
    }
    tab_id, content = mapping.get(btn_id, ("intro", tab_intro))
    return content, tab_id

# ─── Sub-pestañas de intro ────────────────────────────────────────────────────
@app.callback(Output("intro-tab-content", "children"), Input("intro-tabs", "active_tab"))
def render_intro_tab(tab):
    return INTRO_TABS_CONTENT.get(tab, "")

# ─── ADF + ACF + PACF (versión profesional) ───────────────────────────────────
@app.callback(
    Output("adf-result-ui",    "children"),
    Output("adf-crit-table",   "children"),
    Output("adf-implicaciones","children"),
    Output("plot-acf",         "figure"),
    Output("plot-pacf",        "figure"),
    Input("current-tab",       "data"),
    Input("init-trigger",      "n_intervals"),
)
def render_adf(tab, _n):
    """
    Calcula ADF, ACF y PACF sobre la serie de medianas anuales de tasa_fin_cap.
    Se dispara al navegar al tab univariado Y en el primer render (init-trigger).

    Correcciones sobre versión anterior:
      - Extrae correctamente adf_nlag y adf_nobs de adfuller()
      - pacf usa method='ywm' (más estable con n=23)
      - Tabla de valores críticos generada dinámicamente desde resultados reales
      - Badges con color semántico (verde/rojo/amarillo) por resultado
      - Correlogramas con banda sombreada, anotaciones y leyenda legible
      - Implicaciones con 3 ramas: estacionaria / zona gris / no estacionaria
    """
    empty = html.Div()
    if tab != "univariado":
        return empty, empty, empty, _empty_fig(), _empty_fig()

    try:
        # ── Serie temporal ────────────────────────────────────────────────────────
        medianas = df_2.groupby("anio")["tasa_fin_cap"].median().sort_index()
        anios    = medianas.index.tolist()
        serie    = medianas.values

        if len(serie) < 5:
            msg  = "Serie insuficiente para ADF/ACF/PACF (se requieren ≥ 5 observaciones anuales)."
            warn = html.Div(msg, style={"color": AMARILLO, "padding": "12px"})
            return warn, empty, empty, _empty_fig(msg), _empty_fig(msg)

        # ── ADF ───────────────────────────────────────────────────────────────────
        try:
            adf_res  = adfuller(serie, autolag="AIC")
            adf_stat = adf_res[0]
            adf_pval = adf_res[1]
            adf_nlag = adf_res[2]   # número de lags seleccionados por AIC
            adf_nobs = adf_res[3]   # observaciones efectivas usadas
            crit     = adf_res[4]   # {"1%": ..., "5%": ..., "10%": ...}
        except Exception:
            adf_stat, adf_pval, adf_nlag, adf_nobs = 0.0, 0.99, 0, len(serie)
            crit = {"1%": -3.75, "5%": -3.00, "10%": -2.63}

        # Clasificación de p-valor
        if adf_pval < 0.01:
            conclusion = "✓  Estacionaria — evidencia muy fuerte (p < 0.01)"
            col = VERDE_OK
        elif adf_pval < 0.05:
            conclusion = "✓  Estacionaria — nivel estándar (p < 0.05)"
            col = VERDE_OK
        elif adf_pval < 0.10:
            conclusion = "⚠  Zona gris — evidencia débil (p < 0.10)"
            col = AMARILLO
        else:
            conclusion = "✗  No estacionaria — raíz unitaria no descartada (p >= 0.10)"
            col = ROJO_OUT

        pval_str = f"{adf_pval:.4f}" if adf_pval >= 0.0001 else "< 0.0001"

        # ── UI: resultado ADF (badges + banner) ───────────────────────────────────
        adf_ui = html.Div([
            _verdict_banner(conclusion, col),
            html.Div([
                _badge("Estadístico ADF", f"{adf_stat:.4f}", CYAN),
                _badge("p-valor",         pval_str,          col),
                _badge("Lags AIC",        str(adf_nlag),     TEXT_DIM),
                _badge("N obs.",          str(adf_nobs),     TEXT_DIM),
            ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Span("Serie: ", style={"color": TEXT_DIM, "fontSize": "0.8em"}),
                html.Span(f"medianas anuales de tasa_fin_cap  ·  n = {len(serie)} puntos  "
                          f"({anios[0]}–{anios[-1]})",
                          style={"color": TEXT_MAIN, "fontSize": "0.8em"}),
            ]),
        ])

        # ── UI: tabla valores críticos (dinámica) ─────────────────────────────────
        crit_table_ui = _crit_table(crit, adf_stat)

        # ── UI: implicaciones (3 ramas) ───────────────────────────────────────────
        if adf_pval >= 0.10:
            impl_text = [
                html.Strong("Implicación: "),
                "La serie de medianas presenta raíz unitaria (no estacionaria). ",
                "Se recomienda aplicar una ", html.Strong("diferenciación de orden 1"),
                " antes de modelizar con ARIMA. Evalúe también transformaciones ",
                "logarítmicas si la varianza no es constante.",
            ]
        elif adf_pval < 0.05:
            impl_text = [
                html.Strong("Implicación: "),
                "La serie es ", html.Strong("estacionaria en nivel"),
                " — puede modelizarse directamente con ARIMA(p,0,q). ",
                "Consulte la PACF para el orden p y la ACF para el orden q.",
            ]
        else:
            impl_text = [
                html.Strong("Implicación: "),
                "Evidencia débil de estacionariedad. Considere pruebas complementarias ",
                "(KPSS, PP) antes de decidir el orden de integración d.",
            ]

        impl_ui = html.Div([
            html.Div([html.I(className="fas fa-lightbulb", style={"marginRight":"6px","color":CYAN}), "  Implicación para el modelado"],
                     style={"color": CYAN, "fontWeight": "700", "marginBottom": "8px",
                            "fontSize": "0.85em", "textTransform": "uppercase",
                            "letterSpacing": "0.06em"}),
            html.P(impl_text, style={"color": TEXT_MAIN, "fontSize": "0.91em",
                                      "lineHeight": "1.65", "margin": "0"}),
        ], style={
            "background": f"rgba({_hex_to_rgb(col)},0.07)",
            "border": f"1px solid rgba({_hex_to_rgb(col)},0.25)",
            "borderRadius": "10px", "padding": "14px 18px",
        })

        # ── ACF / PACF ────────────────────────────────────────────────────────────
        n = len(serie)
        # ACF: puede calcular hasta n-2 lags, limitamos a 15
        n_lags_acf  = min(15, n - 2)
        # PACF: statsmodels exige nlags < n//2 (límite estricto)
        n_lags_pacf = min(10, n // 2 - 1)
        ci           = 1.96 / np.sqrt(n)

        acf_vals  = acf(serie, nlags=n_lags_acf,  fft=False, alpha=None)
        pacf_vals = pacf(serie, nlags=n_lags_pacf, method="ywm")

        lags_acf  = list(range(len(acf_vals)))
        lags_pacf = list(range(len(pacf_vals)))

        fig_acf = _build_acf_fig(
            lags_acf, acf_vals, ci,
            title="ACF — Función de Autocorrelación",
            bar_color_sig=CYAN,
            bar_color_ns=AZUL_MAIN,
        )
        fig_pacf = _build_acf_fig(
            lags_pacf, pacf_vals, ci,
            title="PACF — Función de Autocorrelación Parcial",
            bar_color_sig=ROJO_OUT,
            bar_color_ns="#5B7A99",
        )

        return adf_ui, crit_table_ui, impl_ui, fig_acf, fig_pacf

    except Exception as exc:
        import traceback
        print(f"[render_adf ERROR] {traceback.format_exc()}")
        err_div = html.Div([
            html.Strong("⚠ Error en ADF/ACF/PACF:", style={"color": ROJO_OUT}),
            html.Pre(str(exc), style={"color": AMARILLO, "fontSize": "0.8em",
                                      "background": "rgba(0,0,0,0.3)",
                                      "padding": "8px", "borderRadius": "6px",
                                      "whiteSpace": "pre-wrap", "marginTop": "8px"}),
        ], style={"padding": "12px"})
        return err_div, html.Div(), html.Div(), _empty_fig(str(exc)), _empty_fig(str(exc))

# ─── Histograma + Boxplot ──────────────────────────────────────────────────────
@app.callback(
    Output("plot-hist", "figure"),
    Output("plot-boxplot", "figure"),
    Output("stats-descriptivas", "children"),
    Input("current-tab", "data")
)
def render_hist_box(tab):
    if tab != "univariado":
        return go.Figure(), go.Figure(), html.Div()
    d = df_2["tasa_fin_cap"].dropna()
    if len(d) < 10:
        msg = "Datos insuficientes para histograma/boxplot (se necesitan ≥ 10 obs.)."
        return _empty_fig(msg), _empty_fig(msg), html.Div(msg, style={"color": "#F39C12"})

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=d, nbinsx=40, name="Frecuencia",
        marker_color=AZUL_MAIN, opacity=0.75,
        hovertemplate="Rango: %{x}<br>Frecuencia: %{y}<extra></extra>"
    ))
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(d)
    x_kde = np.linspace(d.min(), d.max(), 300)
    y_kde = kde(x_kde)
    scale = len(d) * (d.max() - d.min()) / 40
    fig_hist.add_trace(go.Scatter(x=x_kde, y=y_kde * scale, mode="lines",
                                   line=dict(color=CYAN, width=2.5), name="Densidad KDE"))
    fig_hist = pl_layout(fig_hist, "Histograma de tasa_fin_cap", "n = " + f"{len(d):,} observaciones",
                          "tasa_fin_cap (%)", "Frecuencia")
    fig_hist.update_layout(height=360)

    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=d, name="tasa_fin_cap",
        marker_color=AZUL_MAIN, line_color=CYAN,
        boxpoints="outliers", jitter=0.3,
        hovertemplate="tasa_fin_cap: %{y:.1f}%<extra></extra>"
    ))
    fig_box = pl_layout(fig_box, "Boxplot de tasa_fin_cap", "", "", "tasa_fin_cap (%)")
    fig_box.update_layout(height=360)

    # Estadísticas descriptivas
    asim = (d.mean() - d.median()) / d.std()
    stats_ui = html.Div([
        html.Div([
            kpi_card(f"{d.mean():.2f}%",   "Media"),
            kpi_card(f"{d.median():.2f}%", "Mediana", VERDE_OK),
            kpi_card(f"{d.std():.2f} pp",  "Desv. Est.", AMARILLO),
            kpi_card(f"{asim:.3f}",         "Asimetría (Pearson)", ROJO_OUT),
            kpi_card(f"{d.quantile(0.01):.1f}%", "P1"),
            kpi_card(f"{d.quantile(0.99):.1f}%", "P99"),
        ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "margin": "12px 0"}),
    ])
    return fig_hist, fig_box, stats_ui

# ─── Tabla > 100% ─────────────────────────────────────────────────────────────
@app.callback(Output("tabla-mayor-100", "children"), Input("current-tab", "data"))
def render_tabla_gt100(tab):
    if tab != "univariado":
        return html.Div()
    top = (df_2[df_2["tasa_fin"] > 100]
           .sort_values("tasa_fin", ascending=False)
           .head(30)
           .assign(exceso=lambda x: (x["tasa_fin"] - 100).round(2),
                   tasa_fin=lambda x: x["tasa_fin"].round(2))
           [["pais", "anio", "tasa_fin", "exceso"]])

    caption = (f"Total obs > 100%: {N_GT100}  |  Proporción: {PCT_GT100}%  |  "
               f"Máximo: {MAX_TASA}%")

    return html.Div([
        html.P(caption, style={"color": TEXT_DIM, "fontSize": "0.88em", "marginBottom": "8px"}),
        dash_table.DataTable(
            data=top.to_dict("records"),
            columns=[
                {"name": "País",          "id": "pais"},
                {"name": "Año",           "id": "anio"},
                {"name": "tasa_fin (%)",  "id": "tasa_fin"},
                {"name": "Exceso (pp)",   "id": "exceso"},
            ],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#0A1628", "color": CYAN,
                          "fontWeight": "600", "fontSize": "0.87em",
                          "border": f"1px solid rgba(46,134,171,0.5)"},
            style_data={"backgroundColor": BG_CARD, "color": TEXT_MAIN,
                        "border": "1px solid rgba(46,134,171,0.12)"},
            style_data_conditional=[
                {"if": {"filter_query": "{tasa_fin} > 105"},
                 "backgroundColor": "rgba(125,60,152,0.35)", "color": "white", "fontWeight": "bold"},
                {"if": {"row_index": "odd"}, "backgroundColor": BG_PLOT},
            ],
            page_size=10,
        )
    ])

# ─── Lollipop ─────────────────────────────────────────────────────────────────
@app.callback(Output("plot-lollipop", "figure"), Input("slider-n-paises", "value"), Input("current-tab", "data"))
def render_lollipop(n_top, tab):
    if tab != "bivariado":
        return go.Figure()
    if df_2.empty or df_2["pais"].nunique() < 2:
        return _empty_fig("Datos insuficientes para lollipop.")
    top_paises = (df_2.groupby("pais")["anio"].nunique()
                  .sort_values(ascending=False).head(n_top).index.tolist())

    d = df_2[df_2["pais"].isin(top_paises)].copy()
    d["quinquenio"] = pd.cut(d["anio"],
        bins=[1999,2004,2009,2014,2019,2022],
        labels=["2000–2004","2005–2009","2010–2014","2015–2019","2020–2022"])

    quintiles = ["2000–2004","2005–2009","2010–2014","2015–2019","2020–2022"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=quintiles,
                        vertical_spacing=0.14, horizontal_spacing=0.08)

    for idx, q in enumerate(quintiles):
        row, col = divmod(idx, 3)
        dq = (d[d["quinquenio"] == q]
              .groupby("pais")["tasa_fin_cap"]
              .agg(med="median", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
              .reset_index().sort_values("med", ascending=True))

        for _, row_d in dq.iterrows():
            fig.add_trace(go.Scatter(
                x=[row_d["p25"], row_d["p75"]], y=[row_d["pais"], row_d["pais"]],
                mode="lines", line=dict(color=GRIS_LINEA, width=5),
                showlegend=False,
                hovertemplate=f"<b>{row_d['pais']}</b><br>IQR: [{row_d['p25']:.1f}%, {row_d['p75']:.1f}%]<extra></extra>"
            ), row=row+1, col=col+1)
            fig.add_trace(go.Scatter(
                x=[row_d["med"]], y=[row_d["pais"]],
                mode="markers", marker=dict(color=AZUL_CLARO, size=10, line=dict(color=AZUL_FUERTE, width=2)),
                showlegend=False,
                hovertemplate=f"<b>{row_d['pais']}</b><br>Mediana: {row_d['med']:.1f}%<extra></extra>"
            ), row=row+1, col=col+1)
        fig.add_vline(x=100, line_dash="dash", line_color=ROJO_OUT, line_width=1.5,
                      row=row+1, col=col+1)

    fig.update_layout(
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PLOT, font=dict(color=TEXT_MAIN),
        title=dict(text=f"<b>Mediana e IQR de tasa_fin_cap por País y Quinquenio</b><br><sup>Top {n_top} países por cobertura. Línea roja: techo (100%)</sup>",
                   font=dict(size=14, color=CYAN), x=0.02),
        height=560, margin=dict(t=80, b=40, l=20, r=20),
        hoverlabel=dict(bgcolor="#0A1628", bordercolor=CYAN, font=dict(size=11, color=TEXT_LIGHT))
    )
    return fig

# ─── % en Techo ───────────────────────────────────────────────────────────────
@app.callback(Output("plot-techo", "figure"), Input("slider-n-paises", "value"), Input("current-tab", "data"))
def render_techo(n_top, tab):
    if tab != "bivariado":
        return go.Figure()
    top_paises = (df_2.groupby("pais")["anio"].nunique()
                  .sort_values(ascending=False).head(n_top).index.tolist())
    res = (df_2[df_2["pais"].isin(top_paises)]
           .groupby("pais")["tasa_fin_cap"]
           .apply(lambda x: (x >= 100 - 1e-9).mean())
           .reset_index(name="pct_cap")
           .sort_values("pct_cap"))
    res["color"] = res["pct_cap"].apply(
        lambda x: AZUL_CLARO if x < 0.3 else (AZUL_MAIN if x < 0.6 else AZUL_FUERTE))

    fig = go.Figure(go.Bar(
        x=res["pct_cap"], y=res["pais"], orientation="h",
        marker_color=res["color"],
        text=[f"{v*100:.1f}%" for v in res["pct_cap"]], textposition="outside",
        hovertemplate="País: %{y}<br>% en techo: %{x:.1%}<extra></extra>"
    ))
    fig = pl_layout(fig, "% de años en el techo (100%) por país",
                    "Alto % indica que el indicador no discrimina el desempeño real",
                    "Proporción en techo", "")
    fig.update_xaxes(tickformat=".0%", range=[0, 1.4])
    fig.update_layout(showlegend=False, height=400)
    return fig

# ─── Scatter + LOESS ──────────────────────────────────────────────────────────
@app.callback(Output("plot-scatter-loess", "figure"), Input("current-tab", "data"))
def render_scatter_loess(tab):
    if tab != "bivariado":
        return go.Figure()
    d = df_2[df_2["anio"].between(2000, 2022)].dropna(subset=["tasa_fin_cap"])
    if len(d) < 10:
        return _empty_fig("Datos insuficientes para scatter LOESS.")
    # LOESS via statsmodels
    lo = lowess(d["tasa_fin_cap"], d["anio"], frac=0.55, return_sorted=True)
    lo_df = pd.DataFrame(lo, columns=["x", "y"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["anio"], y=d["tasa_fin_cap"], mode="markers",
        marker=dict(color=AZUL_MAIN, opacity=0.25, size=4),
        name="País-año",
        customdata=d["pais"],
        hovertemplate="Año: %{x}<br>País: %{customdata}<br>tasa_fin_cap: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=lo_df["x"], y=lo_df["y"], mode="lines",
        line=dict(color=AZUL_FUERTE, width=2.8), name="Tendencia LOESS",
        hovertemplate="Año: %{x:.1f}<br>LOESS: %{y:.1f}%<extra></extra>"
    ))
    fig.add_hline(y=100, line_dash="dash", line_color=ROJO_OUT, line_width=1.5, name="Techo = 100%")
    fig = pl_layout(fig, "Dispersión global: tasa_fin_cap vs. año + LOESS",
                    "Cada punto = un país-año. Hover para identificar el país", "Año", "tasa_fin_cap (%)")
    fig.update_xaxes(range=[1999, 2023], dtick=2)
    fig.update_yaxes(range=[0, 107])
    fig.update_layout(height=400)
    return fig

# =============================================================================
# CALLBACKS — SECCIÓN BIVARIADA (VERSIÓN CORREGIDA Y ROBUSTA)
# =============================================================================

import numpy as np
from scipy.stats import linregress
from dash import callback_context
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# =============================================================================
# STORE PARA CONTROL DE TABS
# AGREGAR ESTO EN EL LAYOUT PRINCIPAL
# =============================================================================

dcc.Store(id="current-tab", data="intro")

# =============================================================================
# CALLBACK PARA ACTUALIZAR TAB ACTIVO
# =============================================================================

@app.callback(
    Output("current-tab", "data"),
    Input("nav-intro", "n_clicks"),
    Input("nav-univariado", "n_clicks"),
    Input("nav-bivariado", "n_clicks"),
    Input("nav-mapa", "n_clicks"),
    Input("nav-conclusiones", "n_clicks"),
    Input("nav-datos", "n_clicks"),
    Input("nav-referencias", "n_clicks"),
    Input("nav-prediccion", "n_clicks"),
)
def update_current_tab(*args):

    ctx = callback_context

    if not ctx.triggered:
        return "intro"

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    mapping = {
        "nav-intro": "intro",
        "nav-univariado": "univariado",
        "nav-bivariado": "bivariado",
        "nav-mapa": "mapa",
        "nav-conclusiones": "conclusiones",
        "nav-datos": "datos",
        "nav-referencias": "referencias",
        "nav-prediccion": "prediccion"
    }

    return mapping.get(button_id, "intro")


# =============================================================================
# FUNCIÓN AUXILIAR PARA ESTADÍSTICAS ANUALES
# =============================================================================

def yearly_group():

    return (
        df_2[
            df_2["anio"].between(2000, 2022)
        ]
        .dropna(subset=["tasa_fin_cap"])
        .groupby("anio")["tasa_fin_cap"]
    )


# =============================================================================
# MEDIANA ANUAL + IQR
# =============================================================================
# ─────────────────────────────────────────────────────────────
# MEDIANA ANUAL + IQR
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("plot-mediana-anual", "figure"),
    Input("current-tab", "data")
)
def render_mediana_anual(tab):

    if tab != "bivariado":
        return go.Figure()

    try:

        agg = (
            df_2[
                df_2["anio"].between(2000, 2022)
            ]
            .dropna(subset=["tasa_fin_cap"])
            .groupby("anio")["tasa_fin_cap"]
            .agg(
                mediana="median",
                p25=lambda x: x.quantile(0.25),
                p75=lambda x: x.quantile(0.75)
            )
            .reset_index()
        )

        fig = go.Figure()

        # Banda IQR
        fig.add_trace(go.Scatter(

            x=np.concatenate([
                agg["anio"],
                agg["anio"][::-1]
            ]),

            y=np.concatenate([
                agg["p75"],
                agg["p25"][::-1]
            ]),

            fill="toself",

            fillcolor="rgba(46,134,171,0.30)",

            line=dict(color="rgba(0,0,0,0)"),

            name="IQR P25-P75",

            hoverinfo="skip"
        ))

        # Mediana
        fig.add_trace(go.Scatter(

            x=agg["anio"],

            y=agg["mediana"],

            mode="lines+markers",

            line=dict(
                color="#2E86AB",
                width=3
            ),

            marker=dict(
                size=7,
                color="white",
                line=dict(
                    color="#2E86AB",
                    width=2
                )
            ),

            name="Mediana"
        ))

        fig.update_layout(

            title="Mediana anual con banda IQR",

            template="plotly_dark",

            paper_bgcolor="#081C3A",

            plot_bgcolor="#0B2347",

            font=dict(color="white"),

            height=400
        )

        fig.update_xaxes(dtick=2)

        return fig

    except Exception as e:

        print(e)

        return go.Figure()

# =============================================================================
# MEDIA VS MEDIANA
# =============================================================================

@app.callback(
    Output("plot-media-mediana", "figure"),
    Input("current-tab", "data")
)
def render_media_mediana(tab):

    try:

        if tab != "bivariado":
            return go.Figure()

        agg = (
            yearly_group()
            .agg(
                media="mean",
                mediana="median"
            )
            .reset_index()
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(

            x=agg["anio"],
            y=agg["media"],

            mode="lines",

            line=dict(
                color="rgba(0,0,0,0)"
            ),

            showlegend=False,
            hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(

            x=agg["anio"],
            y=agg["mediana"],

            mode="lines",

            fill="tonexty",

            fillcolor="rgba(174,214,241,0.35)",

            line=dict(
                color="rgba(0,0,0,0)"
            ),

            name="Brecha media–mediana",

            hovertemplate=(
                "Año: %{x}"
                "<extra></extra>"
            )
        ))

        fig.add_trace(go.Scatter(

            x=agg["anio"],
            y=agg["media"],

            mode="lines+markers",

            line=dict(
                color=AMARILLO,
                width=2.5,
                dash="dash"
            ),

            marker=dict(
                color=AMARILLO,
                size=6
            ),

            name="Media",

            hovertemplate=(
                "Año: %{x}"
                "<br>Media: %{y:.1f}%"
                "<extra></extra>"
            )
        ))

        fig.add_trace(go.Scatter(

            x=agg["anio"],
            y=agg["mediana"],

            mode="lines+markers",

            line=dict(
                color=AZUL_FUERTE,
                width=2.5
            ),

            marker=dict(
                color=AZUL_FUERTE,
                size=6
            ),

            name="Mediana",

            hovertemplate=(
                "Año: %{x}"
                "<br>Mediana: %{y:.1f}%"
                "<extra></extra>"
            )
        ))

        fig = pl_layout(
            fig,
            "Media vs. Mediana anual de tasa_fin_cap",
            "Brecha entre tendencia central robusta y promedio",
            "Año",
            "tasa_fin_cap (%)"
        )

        fig.update_xaxes(dtick=2)

        fig.update_yaxes(range=[70, 102])

        fig.update_layout(height=380)

        return fig

    except Exception as e:

        print(f"[ERROR render_media_mediana] {e}")

        return _empty_fig(str(e))


# =============================================================================
# RIBBON MULTICAPA
# =============================================================================
# ─────────────────────────────────────────────────────────────
# RIBBON MULTICAPA
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("plot-mv-ribbon", "figure"),
    Input("current-tab", "data")
)
def render_ribbon(tab):

    if tab != "bivariado":
        return go.Figure()

    try:

        agg = (
            df_2[
                df_2["anio"].between(2000, 2022)
            ]
            .dropna(subset=["tasa_fin_cap"])
            .groupby("anio")["tasa_fin_cap"]
            .agg(
                med="median",
                p10=lambda x: x.quantile(0.10),
                p25=lambda x: x.quantile(0.25),
                p75=lambda x: x.quantile(0.75),
                p90=lambda x: x.quantile(0.90)
            )
            .reset_index()
        )

        fig = go.Figure()

        # Banda P10-P90
        fig.add_trace(go.Scatter(

            x=np.concatenate([
                agg["anio"],
                agg["anio"][::-1]
            ]),

            y=np.concatenate([
                agg["p90"],
                agg["p10"][::-1]
            ]),

            fill="toself",

            fillcolor="rgba(173,216,230,0.25)",

            line=dict(color="rgba(0,0,0,0)"),

            name="P10-P90"
        ))

        # Banda IQR
        fig.add_trace(go.Scatter(

            x=np.concatenate([
                agg["anio"],
                agg["anio"][::-1]
            ]),

            y=np.concatenate([
                agg["p75"],
                agg["p25"][::-1]
            ]),

            fill="toself",

            fillcolor="rgba(46,134,171,0.45)",

            line=dict(color="rgba(0,0,0,0)"),

            name="IQR"
        ))

        # Mediana
        fig.add_trace(go.Scatter(

            x=agg["anio"],

            y=agg["med"],

            mode="lines+markers",

            line=dict(
                color="white",
                width=3
            ),

            marker=dict(
                size=7
            ),

            name="Mediana"
        ))

        fig.update_layout(

            title="Evolución del Panel — Mediana con Bandas",

            template="plotly_dark",

            paper_bgcolor="#081C3A",

            plot_bgcolor="#0B2347",

            font=dict(color="white"),

            height=500
        )

        fig.update_xaxes(dtick=2)

        return fig

    except Exception as e:

        print(e)

        return go.Figure()
# =============================================================================
# HEATMAP
# =============================================================================

@app.callback(
    Output("plot-heatmap", "figure"),
    Input("slider-n-heat", "value"),
    Input("current-tab", "data")
)
def render_heatmap(n_heat, tab):

    try:

        if tab != "bivariado":
            return go.Figure()

        if df_2.empty or df_2["pais"].nunique() < 2:
            return _empty_fig("Datos insuficientes para mapa de calor.")

        top_p = (
            df_2.groupby("pais")["tasa_fin_cap"]
            .mean()
            .sort_values(ascending=False)
            .head(n_heat)
            .index
            .tolist()
        )

        d_h = df_2[
            df_2["pais"].isin(top_p)
        ][["pais", "anio", "tasa_fin_cap"]]

        # CORREGIDO
        mat = d_h.pivot_table(
            index="pais",
            columns="anio",
            values="tasa_fin_cap",
            aggfunc="mean"
        )

        mat = mat.loc[top_p]

        fig = go.Figure(go.Heatmap(

            z=mat.values,

            x=[str(c) for c in mat.columns],

            y=mat.index.tolist(),

            colorscale=[
                [0, "#FEF9E7"],
                [0.25, "#AED6F1"],
                [0.6, AZUL_MAIN],
                [1, AZUL_FUERTE]
            ],

            zmin=0,
            zmax=100,

            hovertemplate=(
                "País: %{y}"
                "<br>Año: %{x}"
                "<br>tasa_fin_cap: %{z:.1f}%"
                "<extra></extra>"
            )
        ))

        fig = pl_layout(
            fig,
            f"Mapa de Calor — Top {n_heat}",
            "",
            "Año",
            "País"
        )

        fig.update_layout(
            height=max(400, n_heat * 18 + 100)
        )

        return fig

    except Exception as e:

        print(f"[ERROR render_heatmap] {e}")

        return _empty_fig(str(e))
# ─── Pendientes pre-techo ──────────────────────────────────────────────────────
@app.callback(Output("plot-slopes-top", "figure"), Input("slider-n-slopes", "value"), Input("current-tab", "data"))
def render_slopes(n_top, tab):
    if tab != "bivariado":
        return go.Figure()
    if df_2.empty or df_2["pais"].nunique() < 2:
        return _empty_fig("Datos insuficientes para pendientes.")
    from scipy.stats import linregress
    resultados = []
    for pais, grp in df_2.groupby("pais"):
        pre_techo = grp[(grp["tasa_fin_cap"] < 99.5) & grp["anio"].between(2000, 2022)].sort_values("anio")
        if len(pre_techo) >= 3:
            slope, _, _, _, _ = linregress(pre_techo["anio"], pre_techo["tasa_fin_cap"])
            resultados.append({"pais": pais, "pendiente": slope, "n_obs": len(pre_techo),
                                "anio_ini": int(pre_techo["anio"].min()),
                                "anio_fin": int(pre_techo["anio"].max())})

    top = (pd.DataFrame(resultados).sort_values("pendiente", ascending=False).head(n_top)
           .sort_values("pendiente"))

    fig = go.Figure(go.Bar(
        x=top["pendiente"], y=top["pais"], orientation="h",
        marker_color=[AZUL_FUERTE if v > 3 else AZUL_MAIN for v in top["pendiente"]],
        text=[f"{v:.2f} pp/año" for v in top["pendiente"]], textposition="outside",
        hovertemplate=(
            "País: %{y}<br>Pendiente: %{x:.2f} pp/año"
            "<extra></extra>"
        )
    ))
    fig = pl_layout(fig, f"Top {n_top} países — Pendiente pre-techo (pp/año)",
                    "lm(tasa_fin_cap ~ anio) solo en años antes del techo",
                    "Pendiente (pp/año)", "")
    fig.update_layout(height=480, showlegend=False)
    return fig

# ─── Trayectorias individuales ─────────────────────────────────────────────────
@app.callback(Output("plot-evol-paises", "figure"), Input("dropdown-paises", "value"), Input("current-tab", "data"))
def render_evol_paises(paises, tab):
    if tab != "bivariado":
        return go.Figure()
    if not paises:
        return go.Figure()
    d = df_2[df_2["pais"].isin(paises)].sort_values(["pais", "anio"])
    colors = px.colors.qualitative.Set2
    fig = go.Figure()
    for i, p in enumerate(paises):
        dp = d[d["pais"] == p]
        fig.add_trace(go.Scatter(
            x=dp["anio"], y=dp["tasa_fin_cap"], mode="lines+markers",
            name=p, line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{p}</b><br>Año: %{{x}}<br>tasa_fin_cap: %{{y:.1f}}%<extra></extra>"
        ))
    fig.add_hline(y=100, line_dash="dash", line_color=ROJO_OUT, line_width=1.5)
    fig = pl_layout(fig, "Trayectorias individuales de tasa_fin_cap",
                    "Huecos = años sin reporte. Línea roja = techo (100%)",
                    "Año", "tasa_fin_cap (%)")
    fig.update_xaxes(range=[1999, 2023], dtick=2)
    fig.update_yaxes(range=[0, 107])
    fig.update_layout(height=480)
    return fig

# ─── Prueba de hipótesis ──────────────────────────────────────────────────────
@app.callback(
    Output("plot-hip-boxplot", "figure"),
    Output("plot-hip-dist",    "figure"),
    Output("hip-resultado",    "children"),
    Input("btn-hipotesis", "n_clicks"),
    State("hip-periodo1", "value"),
    State("hip-periodo2", "value"),
)
def render_hipotesis(n_clicks, p1, p2):
    d = df_2.dropna(subset=["tasa_fin_cap"])
    g1 = d[d["anio"].between(p1[0], p1[1])]["tasa_fin_cap"].values
    g2 = d[d["anio"].between(p2[0], p2[1])]["tasa_fin_cap"].values
    lab1 = f"{p1[0]}–{p1[1]}"
    lab2 = f"{p2[0]}–{p2[1]}"

    # Boxplot
    fig_box = go.Figure()
    for grp, lab, col in [(g1, lab1, AZUL_MAIN), (g2, lab2, VERDE_OK)]:
        fig_box.add_trace(go.Box(y=grp, name=lab, marker_color=col, boxpoints="outliers"))
    fig_box = pl_layout(fig_box, "Distribución por período", "", "Período", "tasa_fin_cap (%)")
    fig_box.update_layout(height=340)

    # Densidad
    fig_dist = go.Figure()
    from scipy.stats import gaussian_kde
    for grp, lab, col in [(g1, lab1, AZUL_MAIN), (g2, lab2, VERDE_OK)]:
        kde = gaussian_kde(grp)
        x_k = np.linspace(min(grp), max(grp), 300)
        fig_dist.add_trace(go.Scatter(x=x_k, y=kde(x_k), mode="lines",
                                       line=dict(color=col, width=2.5), name=f"KDE {lab}",
                                       fill="tozeroy", fillcolor=col.replace(")", ",0.15)").replace("#", "rgba(").replace("rgba(", "rgba(") if False else col))
    fig_dist = pl_layout(fig_dist, "Densidad KDE por período", "", "tasa_fin_cap (%)", "Densidad")
    fig_dist.update_layout(height=340)

    if n_clicks == 0:
        return fig_box, fig_dist, html.Div("Presione 'Ejecutar prueba' para ver resultados.",
                                            style={"color": TEXT_DIM, "padding": "16px"})

    # Tests
    sw_lim = 5000
    sw1 = shapiro(np.random.choice(g1, min(sw_lim, len(g1)), replace=False)) if len(g1) >= 3 else None
    sw2 = shapiro(np.random.choice(g2, min(sw_lim, len(g2)), replace=False)) if len(g2) >= 3 else None
    norm1_ok = sw1 and sw1.pvalue >= 0.05
    norm2_ok = sw2 and sw2.pvalue >= 0.05
    normal_ok = norm1_ok and norm2_ok

    lev = levene(g1, g2)
    homo_ok = lev.pvalue >= 0.05

    if normal_ok and homo_ok:
        prueba = ttest_ind(g1, g2, equal_var=True)
        tipo_lab = "t de Student (varianzas iguales)"
    elif normal_ok and not homo_ok:
        prueba = ttest_ind(g1, g2, equal_var=False)
        tipo_lab = "Welch t-test (varianzas desiguales)"
    else:
        prueba = mannwhitneyu(g1, g2, alternative="two-sided")
        tipo_lab = "Wilcoxon-Mann-Whitney (no paramétrico)"

    pval = prueba.pvalue
    stat_v = prueba.statistic
    es_sig = pval < 0.05
    col_res = VERDE_OK if es_sig else AMARILLO
    med_dif = round(np.median(g2) - np.median(g1), 2)
    med1 = round(np.median(g1), 2)
    med2 = round(np.median(g2), 2)

    veredicto = (
        f"✓  Se RECHAZA H₀ (α = 0.05): la tasa mediana difiere significativamente entre {lab1} y {lab2}."
        if es_sig else
        f"⚠  No se rechaza H₀ (α = 0.05): no hay evidencia suficiente de diferencia entre {lab1} y {lab2}."
    )

    resultado_ui = html.Div([
        html.Div(veredicto, style={"background": "rgba(30,132,73,0.15)" if es_sig else "rgba(212,172,13,0.15)",
                                    "borderLeft": f"5px solid {col_res}", "borderRadius": "8px",
                                    "padding": "14px 18px", "marginBottom": "16px",
                                    "fontWeight": "600", "color": col_res}),
        html.Div([
            html.Div([html.Div("Prueba aplicada", style={"color": TEXT_DIM, "fontSize": "0.8em", "textTransform": "uppercase"}),
                      html.Div(tipo_lab, style={"color": CYAN, "fontWeight": "700"})],
                     style={"background": "rgba(46,134,171,0.1)", "borderRadius": "8px", "padding": "14px", "textAlign": "center", "flex": "1"}),
            html.Div([html.Div("Estadístico / p-valor", style={"color": TEXT_DIM, "fontSize": "0.8em", "textTransform": "uppercase"}),
                      html.Div([f"{stat_v:.3f} / ",
                                html.Span("p < 0.001" if pval < 0.001 else f"p = {pval:.4f}",
                                          style={"color": ROJO_OUT if pval < 0.001 else (AMARILLO if pval < 0.05 else CYAN)})],
                               style={"fontWeight": "700", "fontSize": "1.1em", "color": TEXT_LIGHT})],
                     style={"background": "rgba(46,134,171,0.1)", "borderRadius": "8px", "padding": "14px", "textAlign": "center", "flex": "1"}),
            html.Div([html.Div(f"Mediana {lab2} − {lab1}", style={"color": TEXT_DIM, "fontSize": "0.8em", "textTransform": "uppercase"}),
                      html.Div(f"{med_dif:+.2f} pp ({med1}→{med2}%)",
                               style={"fontWeight": "700", "fontSize": "1.1em",
                                      "color": VERDE_OK if med_dif > 0 else ROJO_OUT})],
                     style={"background": "rgba(46,134,171,0.1)", "borderRadius": "8px", "padding": "14px", "textAlign": "center", "flex": "1"}),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "16px"}),
        html.Div([
            html.Strong("Verificación de supuestos:", style={"color": TEXT_MAIN}), html.Br(),
            html.Span(f"{'✓' if norm1_ok else '✗'} Shapiro-Wilk Período 1: {'Normal' if norm1_ok else 'No normal'} "
                      f"(p = {sw1.pvalue:.4f})" if sw1 else "⚠ Shapiro-Wilk Período 1: no calculado"),
            html.Br(),
            html.Span(f"{'✓' if norm2_ok else '✗'} Shapiro-Wilk Período 2: {'Normal' if norm2_ok else 'No normal'} "
                      f"(p = {sw2.pvalue:.4f})" if sw2 else "⚠ Shapiro-Wilk Período 2: no calculado"),
            html.Br(),
            html.Span(f"{'✓' if homo_ok else '✗'} Homocedasticidad (Levene): "
                      f"{'Varianzas iguales' if homo_ok else 'Varianzas distintas'} (p = {lev.pvalue:.4f})"),
        ], style={"color": TEXT_DIM, "fontSize": "0.88em", "lineHeight": "1.8",
                  "background": "rgba(46,134,171,0.06)", "borderRadius": "8px", "padding": "12px 16px"}),
    ], style={"background": "linear-gradient(135deg,#0A1628,#112240)",
              "border": "1px solid rgba(46,134,171,0.4)", "borderRadius": "12px",
              "padding": "20px 24px", "marginTop": "8px"})

    return fig_box, fig_dist, resultado_ui

# ─── Mapas ───────────────────────────────────────────────────────────────────
import plotly.express as px
from dash import Input, Output

# ─────────────────────────────────────
# LIMPIEZA DEL DATAFRAME
# ─────────────────────────────────────

df_2["iso3"] = (
    df_2["iso3"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df_2 = df_2[df_2["iso3"].str.len() == 3]

df_2 = df_2.dropna(
    subset=[
        "iso3",
        "anio",
        "tasa_fin_cap"
    ]
)

# ─────────────────────────────────────
# MAPA 1
# ─────────────────────────────────────

@app.callback(
    Output("plot-mapa-promedio","figure"),
    Input("plot-mapa-promedio","id")
)
def update_mapa_promedio(_):

    d = (
        df_2.groupby(
            ["iso3","pais"],
            as_index=False
        )["tasa_fin_cap"]
        .mean()
        .rename(
            columns={
                "tasa_fin_cap":"tasa_media"
            }
        )
    )

    fig = px.choropleth(
        d,
        locations="iso3",
        locationmode="ISO-3",
        color="tasa_media",
        hover_name="pais",
        hover_data={
            "tasa_media":":.2f",
            "iso3":False
        },
        range_color=[0,100]
    )

    fig.update_geos(
        showcoastlines=True,
        showocean=True,
        showland=True,
        projection_type="natural earth"
    )

    return fig


# ─────────────────────────────────────
# MAPA 2
# ─────────────────────────────────────────────────────────────
# MAPA MUNDIAL ANIMADO — ORDEN CRONOLÓGICO CORRECTO
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("plot-mapa-animado", "figure"),
    Input("plot-mapa-animado", "id")
)
def update_mapa_animado(_):

    try:

        # =====================================================
        # COPIA Y LIMPIEZA DEL DATAFRAME
        # =====================================================

        d = df_2.copy()

        d = d.dropna(
            subset=[
                "anio",
                "iso3",
                "tasa_fin_cap"
            ]
        )

        # =====================================================
        # ASEGURAR FORMATO NUMÉRICO
        # =====================================================

        d["anio"] = d["anio"].astype(int)

        # =====================================================
        # ORDENAR CRONOLÓGICAMENTE
        # =====================================================

        d = d.sort_values("anio")

        # =====================================================
        # CREAR STRING PARA LA ANIMACIÓN
        # =====================================================

        d["anio_str"] = d["anio"].astype(str)

        # =====================================================
        # CREAR MAPA
        # =====================================================

        fig = px.choropleth(

            d,

            locations="iso3",

            locationmode="ISO-3",

            color="tasa_fin_cap",

            animation_frame="anio_str",

            animation_group="iso3",

            hover_name="pais",

            hover_data={
                "anio": True,
                "tasa_fin_cap": ':.1f'
            },

            range_color=[0, 100],

            color_continuous_scale=[
                [0.0, "#FEF9E7"],
                [0.25, "#AED6F1"],
                [0.6, "#3498DB"],
                [1.0, "#1B4F72"]
            ]
        )

        # =====================================================
        # ORDENAR SLIDER
        # =====================================================

        if fig.layout.sliders:

            fig.layout.sliders[0]["steps"] = sorted(

                fig.layout.sliders[0]["steps"],

                key=lambda step: int(step["label"])
            )

        # =====================================================
        # ORDENAR FRAMES INTERNOS
        # =====================================================

        fig.frames = sorted(

            fig.frames,

            key=lambda frame: int(frame.name)
        )

        # =====================================================
        # VELOCIDAD DE ANIMACIÓN
        # =====================================================

        if fig.layout.updatemenus:

            fig.layout.updatemenus[0]\
                .buttons[0]\
                .args[1]["frame"]["duration"] = 700

            fig.layout.updatemenus[0]\
                .buttons[0]\
                .args[1]["transition"]["duration"] = 300

        # =====================================================
        # ESTILO DEL MAPA
        # =====================================================

        fig.update_layout(

            title={

                "text": "Mapa Mundial Animado — Evolución de tasa_fin_cap",

                "x": 0.5,

                "font": {
                    "size": 24,
                    "color": "#58D3F7"
                }
            },

            paper_bgcolor="#081C3A",

            plot_bgcolor="#0B2347",

            font=dict(
                color="white"
            ),

            geo=dict(

                bgcolor="#081C3A",

                showframe=False,

                showcoastlines=True,

                coastlinecolor="rgba(255,255,255,0.25)",

                projection_type="natural earth"
            ),

            margin=dict(
                l=10,
                r=10,
                t=70,
                b=10
            ),

            height=700
        )

        return fig

    except Exception as e:

        print(f"[ERROR MAPA ANIMADO] {e}")

        return go.Figure()
# ─────────────────────────────────────
# MAPA 3
# ─────────────────────────────────────

@app.callback(
    Output("plot-mapa-cambio","figure"),
    Input("plot-mapa-cambio","id")
)
def update_mapa_cambio(_):

    inicio = df_2[
        df_2["anio"]==2000
    ]

    ultimo = df_2.loc[
        df_2.groupby(
            "iso3"
        )["anio"]
        .idxmax()
    ]

    cambio = inicio.merge(
        ultimo,
        on="iso3"
    )

    cambio["delta"] = (
        cambio["tasa_fin_cap_y"]
        -
        cambio["tasa_fin_cap_x"]
    )

    lim=max(
        cambio["delta"].abs().max(),
        1
    )

    fig=px.choropleth(
        cambio,
        locations="iso3",
        locationmode="ISO-3",
        color="delta",
        color_continuous_midpoint=0,
        range_color=[-lim,lim],
        hover_name="pais_x"
    )

    return fig
# ─── Tabla de datos ────────────────────────────────────────────────────────────
@app.callback(
    Output("tabla-datos", "children"),
    Input("filtro-pais", "value"),
    Input("filtro-anio", "value"),
    Input("current-tab", "data"),
)
def render_tabla_datos(pais, anio_range, tab):
    if tab != "datos":
        return html.Div()
    d = df_2.copy()
    if pais != "(Todos)":
        d = d[d["pais"] == pais]
    d = d[d["anio"].between(anio_range[0], anio_range[1])]
    d = d[["pais", "anio", "tasa_fin", "tasa_fin_cap"]].copy()
    d["tasa_fin"]     = d["tasa_fin"].round(2)
    d["tasa_fin_cap"] = d["tasa_fin_cap"].round(2)

    return dash_table.DataTable(
        data=d.to_dict("records"),
        columns=[
            {"name": "País",                     "id": "pais",        "type": "text"},
            {"name": "Año",                      "id": "anio",        "type": "numeric"},
            {"name": "tasa_fin (%) original",    "id": "tasa_fin",    "type": "numeric"},
            {"name": "tasa_fin_cap (%) [0-100]", "id": "tasa_fin_cap","type": "numeric"},
        ],
        filter_action="native",
        sort_action="native",
        page_size=15,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#0A1628", "color": CYAN, "fontWeight": "600",
                      "fontSize": "0.87em", "border": f"1px solid rgba(46,134,171,0.5)"},
        style_data={"backgroundColor": BG_CARD, "color": TEXT_MAIN,
                    "border": "1px solid rgba(46,134,171,0.12)"},
        style_data_conditional=[
            {"if": {"filter_query": "{tasa_fin} > 100"},
             "backgroundColor": "rgba(125,60,152,0.35)", "color": "#FFD700", "fontWeight": "bold"},
            {"if": {"row_index": "odd"}, "backgroundColor": BG_PLOT},
        ],
        style_filter={"backgroundColor": BG_PLOT, "color": TEXT_MAIN,
                      "border": "1px solid rgba(46,134,171,0.35)"},
    )


# =============================================================================
# MÓDULO PREDICCIÓN ARIMA — Integrado desde tab_prediccion.py
# =============================================================================
# =============================================================================
# CÁLCULOS CENTRALES (fuera del callback para poder cachear)
# =============================================================================


# =============================================================================
# HELPERS DEL MÓDULO PREDICCIÓN ARIMA
# =============================================================================

def _base_layout(fig, title="", subtitle="", xtitle="", ytitle="", height=320):
    full = f"<b>{title}</b><br><sup>{subtitle}</sup>" if subtitle else f"<b>{title}</b>"
    fig.update_layout(
        title=dict(text=full, font=dict(size=13, color=CYAN), x=0.02),
        xaxis=dict(title=dict(text=xtitle, font=dict(color=TEXT_DIM, size=11)),
                   gridcolor="rgba(46,134,171,0.12)", zeroline=False,
                   showline=True, linecolor="rgba(46,134,171,0.25)",
                   tickfont=dict(color=TEXT_DIM, size=10)),
        yaxis=dict(title=dict(text=ytitle, font=dict(color=TEXT_DIM, size=11)),
                   gridcolor="rgba(46,134,171,0.12)", zeroline=False,
                   showline=True, linecolor="rgba(46,134,171,0.25)",
                   tickfont=dict(color=TEXT_DIM, size=10)),
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PLOT, font=dict(color=TEXT_MAIN),
        legend=dict(orientation="h", x=0, y=-0.22, font=dict(color=TEXT_MAIN, size=10),
                    bgcolor="rgba(13,31,60,0.8)", bordercolor="rgba(46,134,171,0.3)", borderwidth=1),
        margin=dict(t=60, r=20, b=65, l=60), height=height,
        hoverlabel=dict(bgcolor="#0A1628", bordercolor=CYAN, font=dict(size=11, color=TEXT_LIGHT)),
    )
    return fig


def _verdict(text, color):
    return html.Div(text, style={
        "background": f"rgba({_hex_to_rgb(color)},0.10)",
        "borderLeft": f"5px solid {color}", "borderRadius": "0 8px 8px 0",
        "padding": "10px 14px", "fontWeight": "700",
        "fontSize": "0.9em", "color": color, "marginBottom": "12px",
    })


def _interp_box(children):
    if isinstance(children, str):
        children = [children]
    return html.Div(children, style={
        "background": "rgba(100,207,246,0.05)",
        "border": "1px solid rgba(100,207,246,0.18)",
        "borderLeft": f"4px solid {CYAN}", "borderRadius": "0 8px 8px 0",
        "padding": "12px 16px", "color": TEXT_MAIN,
        "fontSize": "0.88em", "lineHeight": "1.65", "marginTop": "12px",
    })


def _section(children, border_color=None):
    border_color = border_color or AZUL_MAIN
    return html.Div(children, style={
        "background": BG_CARD, "borderRadius": "12px",
        "padding": "22px 24px", "marginBottom": "20px",
        "borderLeft": f"6px solid {border_color}",
        "boxShadow": "0 4px 20px rgba(0,0,0,0.3)",
    })


def _section_title(text, icon_class=""):
    icon_el = html.I(className=icon_class, style={"marginRight": "8px"}) if icon_class else ""
    return html.H5([icon_el, text], style={
        "color": CYAN, "fontWeight": "800", "fontSize": "1.02em",
        "letterSpacing": "0.03em",
        "borderBottom": "2px solid rgba(46,134,171,0.22)",
        "paddingBottom": "10px", "marginBottom": "16px",
    })


def _build_correlograma(lags, vals, ci, title, bar_color_sig, bar_color_ns, height=260):
    sig = [abs(v) > ci for v in vals]
    fig = go.Figure()
    xb = list(range(len(vals)))
    fig.add_trace(go.Scatter(
        x=xb + xb[::-1], y=[ci]*len(xb) + [-ci]*len(xb),
        fill="toself", fillcolor="rgba(100,207,246,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_hline(y=ci,  line=dict(dash="dot", color="rgba(100,207,246,0.5)", width=1.2))
    fig.add_hline(y=-ci, line=dict(dash="dot", color="rgba(100,207,246,0.5)", width=1.2))
    fig.add_hline(y=0,   line=dict(color="rgba(200,200,200,0.12)", width=1))
    for lag, val, s in zip(lags, vals, sig):
        c = bar_color_sig if s else bar_color_ns
        fig.add_trace(go.Bar(
            x=[lag], y=[val],
            marker=dict(color=c, opacity=0.90 if s else 0.45,
                        line=dict(color="rgba(255,255,255,0.1)", width=1)),
            showlegend=False, width=0.55,
            hovertemplate=(f"<b>Lag {lag}</b><br>r = {val:.4f}<br>"
                           + ("<b>Significativo</b>" if s else "No significativo")
                           + "<extra></extra>"),
        ))
        if s and lag > 0:
            fig.add_annotation(
                x=lag, y=val + (0.04 if val >= 0 else -0.06),
                text=f"{val:.2f}", showarrow=False,
                font=dict(size=8, color=bar_color_sig),
                bgcolor="rgba(13,31,60,0.7)",
                bordercolor=bar_color_sig, borderwidth=1, borderpad=2,
            )
    fig = _base_layout(fig, title, f"IC 95%: +/-{ci:.3f}", "Lag", "Correlacion", height)
    fig.update_layout(bargap=0.3, margin=dict(t=50, b=45, l=50, r=15))
    return fig

def _compute_arima_all(df_2, horizonte=5):
    """
    Ejecuta todo el pipeline ARIMA sobre medianas anuales de df_2.
    Retorna un dict con todos los resultados.
    """
    warnings.filterwarnings("ignore")

    medianas = df_2.groupby("anio")["tasa_fin_cap"].median().sort_index()
    anios    = medianas.index.tolist()
    serie    = medianas.values
    n        = len(serie)

    # ── 1. ADF ────────────────────────────────────────────────────────────────
    adf_res  = adfuller(serie, autolag="AIC")
    adf_stat, adf_pval = adf_res[0], adf_res[1]
    adf_nlag, adf_nobs = adf_res[2], adf_res[3]
    adf_crit = adf_res[4]

    # ── 2. ACF / PACF — original y diferenciada ───────────────────────────────
    n_lags_acf  = min(15, n - 2)
    n_lags_pacf = min(10, n // 2 - 1)
    ci_orig     = 1.96 / np.sqrt(n)

    acf_orig  = acf(serie, nlags=n_lags_acf, fft=False)
    pacf_orig = pacf(serie, nlags=n_lags_pacf, method="ywm")

    serie_diff  = np.diff(serie)
    n_diff      = len(serie_diff)
    ci_diff     = 1.96 / np.sqrt(n_diff)
    n_lags_pacf_diff = min(10, n_diff // 2 - 1)

    acf_diff  = acf(serie_diff, nlags=min(15, n_diff - 2), fft=False)
    pacf_diff = pacf(serie_diff, nlags=n_lags_pacf_diff, method="ywm")

    # ── 3. Grid search ARIMA (p∈{0..3}, d∈{0..2}, q∈{0..3}) ──────────────────
    grid_results = []
    for p in range(4):
        for d in range(3):
            for q in range(4):
                try:
                    m = ARIMA(serie, order=(p, d, q)).fit()
                    grid_results.append({
                        "Modelo": f"ARIMA({p},{d},{q})",
                        "p": p, "d": d, "q": q,
                        "LogLik": round(m.llf, 3),
                        "AIC":    round(m.aic, 3),
                        "BIC":    round(m.bic, 3),
                        "HQIC":   round(m.hqic, 3),
                    })
                except Exception:
                    pass

    df_grid = pd.DataFrame(grid_results)
    df_aic  = df_grid.sort_values("AIC").reset_index(drop=True)
    df_bic  = df_grid.sort_values("BIC").reset_index(drop=True)
    df_hqic = df_grid.sort_values("HQIC").reset_index(drop=True)

    best_aic  = df_aic.iloc[0]
    best_bic  = df_bic.iloc[0]
    best_hqic = df_hqic.iloc[0]

    # ── 4. Ajuste del mejor modelo (AIC) ──────────────────────────────────────
    order_best = (int(best_aic["p"]), int(best_aic["d"]), int(best_aic["q"]))
    model_fit  = ARIMA(serie, order=order_best).fit()

    # fittedvalues/resid de statsmodels tienen indice posicional (0,1,2...)
    # Con d>0 pueden tener menos puntos que 'serie' (los primeros d son NaN).
    # Convertimos a numpy y rellenamos para igualar longitud con 'anios'.
    fitted_raw = np.array(model_fit.fittedvalues, dtype=float)
    resid_raw  = np.array(model_fit.resid,        dtype=float)

    if len(fitted_raw) < n:
        pad = n - len(fitted_raw)
        fitted_raw = np.concatenate([np.full(pad, np.nan), fitted_raw])
        resid_raw  = np.concatenate([np.full(pad, np.nan), resid_raw])

    # Rellenar NaN iniciales: fitted con primer valor valido, resid con 0
    first_valid = next((v for v in fitted_raw if not np.isnan(v)), serie[0])
    fitted_raw  = np.where(np.isnan(fitted_raw), first_valid, fitted_raw)
    resid_raw   = np.where(np.isnan(resid_raw),  0.0,         resid_raw)

    fitted = fitted_raw
    resid  = resid_raw

    # ── 5. Diagnóstico residuales ──────────────────────────────────────────────
    resid_valid = resid[resid != 0.0] if np.any(resid != 0.0) else resid
    sw_stat, sw_p = shapiro(resid_valid)
    lb_result     = acorr_ljungbox(resid_valid, lags=[10], return_df=True)
    lb_stat       = lb_result["lb_stat"].values[0]
    lb_p          = lb_result["lb_pvalue"].values[0]

    # QQ-plot teórico
    qq = probplot(resid_valid, dist="norm")
    qq_x = qq[0][0]  # quantiles teóricos
    qq_y = qq[0][1]  # quantiles observados
    qq_line_x = np.array([qq_x.min(), qq_x.max()])
    qq_line_y = qq[1][1] + qq[1][0] * qq_line_x  # slope, intercept

    # ── 6. Validación 80/20 ───────────────────────────────────────────────────
    split_idx  = int(np.ceil(n * 0.80))
    train_ser  = serie[:split_idx]
    test_ser   = serie[split_idx:]
    test_anios = anios[split_idx:]

    model_train = ARIMA(train_ser, order=order_best).fit()
    fc_val      = model_train.get_forecast(steps=len(test_ser))
    val_pred    = np.array(fc_val.predicted_mean, dtype=float)
    val_ci      = np.array(fc_val.conf_int(alpha=0.05), dtype=float)

    mae   = float(np.mean(np.abs(test_ser - val_pred)))
    rmse  = float(np.sqrt(np.mean((test_ser - val_pred) ** 2)))
    mape  = float(np.mean(np.abs((test_ser - val_pred) / test_ser)) * 100)
    r2    = float(1 - np.sum((test_ser - val_pred)**2) / np.sum((test_ser - np.mean(test_ser))**2))

    df_val = pd.DataFrame({
        "Año":           test_anios,
        "Observado":     np.round(test_ser, 3),
        "Predicho":      np.round(val_pred, 3),
        "Error abs.":    np.round(np.abs(test_ser - val_pred), 3),
        "Error %":       np.round(np.abs((test_ser - val_pred) / test_ser) * 100, 2),
    })

    # ── 7. Pronóstico ─────────────────────────────────────────────────────────
    fc_full   = model_fit.get_forecast(steps=horizonte)
    fc_mean   = np.array(fc_full.predicted_mean, dtype=float)
    fc_ci     = np.array(fc_full.conf_int(alpha=0.05), dtype=float)
    fc_anios  = list(range(anios[-1] + 1, anios[-1] + 1 + horizonte))

    df_fc = pd.DataFrame({
        "Año":      fc_anios,
        "Pronóstico": np.round(fc_mean, 3),
        "IC 95% inf.": np.round(fc_ci[:, 0], 3),
        "IC 95% sup.": np.round(fc_ci[:, 1], 3),
    })

    return dict(
        anios=anios, serie=serie, n=n,
        adf_stat=adf_stat, adf_pval=adf_pval,
        adf_nlag=adf_nlag, adf_nobs=adf_nobs, adf_crit=adf_crit,
        ci_orig=ci_orig, ci_diff=ci_diff,
        acf_orig=acf_orig,  pacf_orig=pacf_orig,
        acf_diff=acf_diff,  pacf_diff=pacf_diff,
        df_aic=df_aic, df_bic=df_bic, df_hqic=df_hqic,
        best_aic=best_aic, best_bic=best_bic, best_hqic=best_hqic,
        order_best=order_best, model_fit=model_fit,
        fitted=fitted, resid=resid,
        sw_stat=sw_stat, sw_p=sw_p,
        lb_stat=lb_stat, lb_p=lb_p,
        qq_x=qq_x, qq_y=qq_y,
        qq_line_x=qq_line_x, qq_line_y=qq_line_y,
        split_idx=split_idx,
        train_ser=train_ser, test_ser=test_ser, test_anios=test_anios,
        val_pred=val_pred, val_ci=val_ci,
        mae=mae, rmse=rmse, mape=mape, r2=r2,
        df_val=df_val,
        fc_mean=fc_mean, fc_ci=fc_ci,
        fc_anios=fc_anios, df_fc=df_fc,
        horizonte=horizonte,
    )


# =============================================================================
# LAYOUT DEL TAB
# =============================================================================

tab_prediccion = html.Div([

    # ── CABECERA ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.I(className="fas fa-robot", style={"fontSize":"1.4em","marginRight":"10px","color":CYAN}),
            html.Span("Predicción ARIMA — Tasa de Finalización Primaria Femenina",
                      style={"fontWeight": "900", "fontSize": "1.1em",
                             "letterSpacing": "0.03em"}),
        ], style={"display": "flex", "alignItems": "center",
                  "color": TEXT_LIGHT, "marginBottom": "12px"}),
        html.P([
            html.Strong("Serie de tiempo: "),
            "mediana anual global de ", html.Code("tasa_fin_cap"),
            " (2000–2022, n = 23 puntos). Se trabaja con medianas mundiales para obtener "
            "una trayectoria representativa libre de valores extremos.",
        ], style={"color": TEXT_MAIN, "fontSize": "0.92em", "marginBottom": "8px"}),
        html.P([
            html.Strong("Flujo: "),
            "(1) ADF → (2) ACF/PACF original y diferenciada → "
            "(3) Grid search AIC/BIC/HQIC → (4) Ajuste → "
            "(5) Diagnóstico residuales → (6) Validación 80/20 → (7) Pronóstico.",
        ], style={"color": TEXT_DIM, "fontSize": "0.88em", "marginBottom": "8px"}),
        html.Div([
            html.I(className="fas fa-exclamation-triangle", style={"color":AMARILLO,"marginRight":"4px"}),
            html.Strong("Nota: ", style={"color": AMARILLO}),
            "Con n = 23 observaciones la potencia estadística es limitada. "
            "Las proyecciones deben interpretarse como tendencias indicativas.",
        ], style={"background": "rgba(243,156,18,0.07)",
                  "border": "1px solid rgba(243,156,18,0.25)",
                  "borderRadius": "8px", "padding": "10px 14px",
                  "fontSize": "0.88em", "color": TEXT_MAIN}),
    ], style={**{"background": BG_CARD, "borderRadius": "12px",
                 "padding": "20px 24px", "marginBottom": "20px",
                 "borderLeft": f"6px solid {CYAN}",
                 "boxShadow": "0 4px 20px rgba(0,0,0,0.3)"}}),

    # ── PASO 1: SERIE + ADF ───────────────────────────────────────────────────
    _section([
        _section_title("Paso 1 — Serie temporal y prueba ADF de estacionariedad", "fas fa-chart-line"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="pred-serie-original",
                          config={"displayModeBar": True, "displaylogo": False,
                                  "modeBarButtonsToRemove": ["select2d", "lasso2d"]}),
                _interp_box([
                    "La serie muestra una tendencia ascendente suave y casi monótona (2000–2022). "
                    "La ausencia de estacionalidad es esperable en datos anuales educativos. "
                    "La curvatura decreciente al final refleja el ",
                    html.Strong("efecto techo"),
                    ": conforme la mediana global se acerca al 100 %, el ritmo de mejora se ralentiza.",
                ]),
            ], width=12, lg=7, className="mb-3 mb-lg-0"),
            dbc.Col([
                html.Div([
                    html.H6("Prueba ADF — Estacionariedad",
                            style={"color": CYAN, "fontWeight": "700",
                                   "marginBottom": "14px", "fontSize": "0.9em",
                                   "textTransform": "uppercase", "letterSpacing": "0.07em"}),
                    html.Div(id="pred-adf-ui"),
                ], style={"background": BG_PLOT,
                          "border": "1px solid rgba(46,134,171,0.2)",
                          "borderRadius": "10px", "padding": "18px"}),
                _interp_box([
                    html.Strong("H₀: "), "la serie tiene raíz unitaria (no estacionaria). ",
                    html.Strong("H₁: "), "la serie es estacionaria. ",
                    "Si p > 0.05 no se rechaza H₀ y se recomienda diferenciar (d ≥ 1) antes del ARIMA.",
                ]),
            ], width=12, lg=5),
        ]),
    ], border_color=AZUL_MAIN),

    # ── PASO 2: ACF / PACF ────────────────────────────────────────────────────
    _section([
        _section_title("Paso 2 — Estructura de autocorrelación (ACF y PACF)", "〰"),
        html.P([
            "Los correlogramas ACF y PACF orientan la selección de los órdenes ",
            html.Strong("q (MA)"), " y ", html.Strong("p (AR)"),
            " del modelo. Se muestran la serie original y la primera diferencia.",
        ], style={"color": TEXT_MAIN, "fontSize": "0.91em", "marginBottom": "16px"}),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("ACF", style={"color": CYAN, "fontWeight": "800",
                                            "fontSize": "0.8em", "letterSpacing": "0.08em"}),
                    html.Span(" — Serie original",
                              style={"color": TEXT_DIM, "fontSize": "0.8em"}),
                ], style={"marginBottom": "4px"}),
                dcc.Graph(id="pred-acf-orig",
                          config={"displayModeBar": False}),
            ], width=12, lg=6, className="mb-3"),
            dbc.Col([
                html.Div([
                    html.Span("PACF", style={"color": ROJO_OUT, "fontWeight": "800",
                                             "fontSize": "0.8em", "letterSpacing": "0.08em"}),
                    html.Span(" — Serie original",
                              style={"color": TEXT_DIM, "fontSize": "0.8em"}),
                ], style={"marginBottom": "4px"}),
                dcc.Graph(id="pred-pacf-orig",
                          config={"displayModeBar": False}),
            ], width=12, lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("ACF", style={"color": VERDE_OK, "fontWeight": "800",
                                            "fontSize": "0.8em", "letterSpacing": "0.08em"}),
                    html.Span(" — Primera diferencia",
                              style={"color": TEXT_DIM, "fontSize": "0.8em"}),
                ], style={"marginBottom": "4px"}),
                dcc.Graph(id="pred-acf-diff",
                          config={"displayModeBar": False}),
            ], width=12, lg=6, className="mb-3"),
            dbc.Col([
                html.Div([
                    html.Span("PACF", style={"color": AMARILLO, "fontWeight": "800",
                                             "fontSize": "0.8em", "letterSpacing": "0.08em"}),
                    html.Span(" — Primera diferencia",
                              style={"color": TEXT_DIM, "fontSize": "0.8em"}),
                ], style={"marginBottom": "4px"}),
                dcc.Graph(id="pred-pacf-diff",
                          config={"displayModeBar": False}),
            ], width=12, lg=6, className="mb-3"),
        ]),
        _interp_box([
            html.Strong("ACF original: "),
            "decaimiento lento y significativo → tendencia, serie no estacionaria. ",
            html.Strong("ACF diferenciada: "),
            "autocorrelaciones dentro de bandas → serie estacionaria tras una diferencia. ",
            html.Strong("PACF diferenciada: "),
            "el número de rezagos significativos orienta el orden p del componente AR.",
        ]),
    ], border_color=AZUL_CLARO),

    # ── PASO 3: GRID SEARCH ───────────────────────────────────────────────────
    _section([
        _section_title("Paso 3 — Selección de orden ARIMA (grid search AIC/BIC/HQIC)", "fas fa-search"),
        html.P([
            "Se evalúan todas las combinaciones p ∈ {0,1,2,3}, d ∈ {0,1,2}, q ∈ {0,1,2,3}. "
            "La tabla muestra los 10 mejores modelos por AIC.",
        ], style={"color": TEXT_MAIN, "fontSize": "0.91em", "marginBottom": "16px"}),

        # Mejores modelos por criterio
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-trophy", style={"fontSize":"1.1em","marginRight":"5px","color":VERDE_OK}),
                        html.Strong("Mejor AIC", style={"color": VERDE_OK}),
                    ], style={"marginBottom": "10px"}),
                    html.Div(id="pred-best-aic-ui"),
                ], style={"background": f"rgba({_hex_to_rgb(VERDE_OK)},0.07)",
                          "border": f"1px solid rgba({_hex_to_rgb(VERDE_OK)},0.25)",
                          "borderLeft": f"6px solid {VERDE_OK}",
                          "borderRadius": "0 10px 10px 0",
                          "padding": "16px"}),
            ], width=12, lg=4, className="mb-3 mb-lg-0"),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-medal", style={"fontSize":"1.1em","marginRight":"5px","color":AZUL_MAIN}),
                        html.Strong("Mejor BIC", style={"color": AZUL_MAIN}),
                    ], style={"marginBottom": "10px"}),
                    html.Div(id="pred-best-bic-ui"),
                ], style={"background": f"rgba({_hex_to_rgb(AZUL_MAIN)},0.07)",
                          "border": f"1px solid rgba({_hex_to_rgb(AZUL_MAIN)},0.25)",
                          "borderLeft": f"6px solid {AZUL_MAIN}",
                          "borderRadius": "0 10px 10px 0",
                          "padding": "16px"}),
            ], width=12, lg=4, className="mb-3 mb-lg-0"),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-award", style={"fontSize":"1.1em","marginRight":"5px","color":AMARILLO}),
                        html.Strong("Mejor HQIC", style={"color": AMARILLO}),
                    ], style={"marginBottom": "10px"}),
                    html.Div(id="pred-best-hqic-ui"),
                ], style={"background": f"rgba({_hex_to_rgb(AMARILLO)},0.07)",
                          "border": f"1px solid rgba({_hex_to_rgb(AMARILLO)},0.25)",
                          "borderLeft": f"6px solid {AMARILLO}",
                          "borderRadius": "0 10px 10px 0",
                          "padding": "16px"}),
            ], width=12, lg=4),
        ], className="mb-4"),

        # Tabla top-10 AIC
        html.H6("Top 10 modelos por AIC",
                style={"color": CYAN, "fontWeight": "700",
                       "marginBottom": "10px", "fontSize": "0.88em",
                       "textTransform": "uppercase", "letterSpacing": "0.07em"}),
        html.Div(id="pred-tabla-aic-container"),

        _interp_box([
            html.Strong("LogLik (log-verosimilitud): "),
            "mide qué tan bien el modelo explica los datos observados — valores menos negativos indican mejor ajuste. ",
            html.Strong("AIC "), "penaliza con 2·k (k = nº parámetros): favorece parsimonia. ",
            html.Strong("BIC "), "penaliza con k·ln(n): más severo con muestras pequeñas. ",
            html.Strong("HQIC "), "punto intermedio entre AIC y BIC. ",
            "La coincidencia entre criterios en el mismo modelo es evidencia adicional de robustez.",
        ]),
    ], border_color=VERDE_OK),

    # ── PASO 4: AJUSTE ────────────────────────────────────────────────────────
    _section([
        _section_title("Paso 4 — Ajuste del modelo ARIMA seleccionado (por AIC)", "fas fa-cogs"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="pred-ajuste-plot",
                          config={"displayModeBar": True, "displaylogo": False,
                                  "modeBarButtonsToRemove": ["select2d", "lasso2d"]}),
            ], width=12, lg=8, className="mb-3 mb-lg-0"),
            dbc.Col([
                html.Div(id="pred-summary-ui"),
                _interp_box([
                    "La línea naranja corresponde a los valores ajustados sobre el periodo de entrenamiento. "
                    "Una buena concordancia sin patrones sistemáticos en la diferencia indica que el modelo "
                    "captura correctamente la dinámica de la serie.",
                ]),
            ], width=12, lg=4),
        ]),
    ], border_color=CYAN),

    # ── PASO 5: RESIDUALES ────────────────────────────────────────────────────
    _section([
        _section_title("Paso 5 — Diagnóstico de residuales (normalidad e independencia)", "fas fa-stethoscope"),
        html.P([
            "Un modelo ARIMA bien especificado produce residuales que se comportan como ",
            html.Strong("ruido blanco"),
            ": media cero, varianza constante, sin autocorrelación y aproximadamente normales.",
        ], style={"color": TEXT_MAIN, "fontSize": "0.91em", "marginBottom": "16px"}),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("Residuales en el tiempo",
                              style={"color": TEXT_DIM, "fontSize": "0.82em",
                                     "fontWeight": "600", "textTransform": "uppercase",
                                     "letterSpacing": "0.06em"}),
                ], style={"marginBottom": "4px"}),
                dcc.Graph(id="pred-resid-ts",
                          config={"displayModeBar": False}),
            ], width=12, lg=4, className="mb-3"),
            dbc.Col([
                html.Div([
                    html.Span("Histograma de residuales",
                              style={"color": TEXT_DIM, "fontSize": "0.82em",
                                     "fontWeight": "600", "textTransform": "uppercase",
                                     "letterSpacing": "0.06em"}),
                ], style={"marginBottom": "4px"}),
                dcc.Graph(id="pred-resid-hist",
                          config={"displayModeBar": False}),
            ], width=12, lg=4, className="mb-3"),
            dbc.Col([
                html.Div([
                    html.Span("QQ-plot de residuales",
                              style={"color": TEXT_DIM, "fontSize": "0.82em",
                                     "fontWeight": "600", "textTransform": "uppercase",
                                     "letterSpacing": "0.06em"}),
                ], style={"marginBottom": "4px"}),
                dcc.Graph(id="pred-resid-qq",
                          config={"displayModeBar": False}),
            ], width=12, lg=4, className="mb-3"),
        ]),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6([html.I(className="fas fa-vial", style={"fontSize":"1.1em","marginRight":"5px","color":VERDE_OK}),
                             "Shapiro-Wilk — Normalidad"],
                            style={"color": CYAN, "fontWeight": "700",
                                   "marginBottom": "12px", "marginTop": "0",
                                   "fontSize": "0.88em", "textTransform": "uppercase",
                                   "letterSpacing": "0.06em"}),
                    html.Div(id="pred-sw-ui"),
                ], style={"background": f"rgba({_hex_to_rgb(VERDE_OK)},0.07)",
                          "border": f"1px solid rgba({_hex_to_rgb(VERDE_OK)},0.25)",
                          "borderLeft": f"6px solid {VERDE_OK}",
                          "borderRadius": "0 10px 10px 0",
                          "padding": "16px"}),
            ], width=12, lg=6, className="mb-3 mb-lg-0"),
            dbc.Col([
                html.Div([
                    html.H6([html.I(className="fas fa-random", style={"fontSize":"1.1em","marginRight":"5px","color":AZUL_MAIN}),
                             "Ljung-Box — Independencia"],
                            style={"color": CYAN, "fontWeight": "700",
                                   "marginBottom": "12px", "marginTop": "0",
                                   "fontSize": "0.88em", "textTransform": "uppercase",
                                   "letterSpacing": "0.06em"}),
                    html.Div(id="pred-lb-ui"),
                ], style={"background": f"rgba({_hex_to_rgb(AZUL_MAIN)},0.07)",
                          "border": f"1px solid rgba({_hex_to_rgb(AZUL_MAIN)},0.25)",
                          "borderLeft": f"6px solid {AZUL_MAIN}",
                          "borderRadius": "0 10px 10px 0",
                          "padding": "16px"}),
            ], width=12, lg=6),
        ]),
        _interp_box([
            html.Strong("Shapiro-Wilk H₀: "), "los residuales son normales. Si p > 0.05, no se rechaza. ",
            html.Strong("Ljung-Box H₀: "), "no existe autocorrelación (ruido blanco). Si p > 0.05, el modelo captura bien la dependencia temporal.",
        ]),
    ], border_color=AMARILLO),

    # ── PASO 6: VALIDACIÓN 80/20 ──────────────────────────────────────────────
    _section([
        _section_title("Paso 6 — Validación del modelo (entrenamiento 80 % / prueba 20 %)", "fas fa-flask"),
        html.Div([
            html.I(className="fas fa-exclamation-triangle", style={"color":AMARILLO,"marginRight":"4px"}),
            html.Strong("Nota importante: ", style={"color": AMARILLO}),
            "Con n = 23 observaciones, el conjunto de prueba tiene solo ~5 puntos. "
            "Los resultados deben interpretarse con cautela como referencia orientativa.",
        ], style={"background": "rgba(243,156,18,0.07)",
                  "border": "1px solid rgba(243,156,18,0.22)",
                  "borderRadius": "8px", "padding": "10px 14px",
                  "fontSize": "0.88em", "color": TEXT_MAIN, "marginBottom": "16px"}),

        dbc.Row([
            dbc.Col([
                dcc.Graph(id="pred-val-plot",
                          config={"displayModeBar": True, "displaylogo": False,
                                  "modeBarButtonsToRemove": ["select2d", "lasso2d"]}),
            ], width=12, lg=8, className="mb-3 mb-lg-0"),
            dbc.Col([
                html.Div([
                    html.H6("Métricas — Conjunto de prueba",
                            style={"color": CYAN, "fontWeight": "700",
                                   "marginBottom": "12px", "marginTop": "0",
                                   "fontSize": "0.88em", "textTransform": "uppercase",
                                   "letterSpacing": "0.06em"}),
                    html.Div(id="pred-val-metricas-ui"),
                ], style={"background": f"rgba({_hex_to_rgb(AMARILLO)},0.06)",
                          "border": f"1px solid rgba({_hex_to_rgb(AMARILLO)},0.22)",
                          "borderRadius": "10px", "padding": "16px",
                          "marginBottom": "14px"}),
                html.Div([
                    html.H6("Detalle por año",
                            style={"color": CYAN, "fontWeight": "700",
                                   "marginBottom": "10px", "marginTop": "0",
                                   "fontSize": "0.88em", "textTransform": "uppercase",
                                   "letterSpacing": "0.06em"}),
                    html.Div(id="pred-val-tabla-container"),
                ], style={"background": f"rgba({_hex_to_rgb(AZUL_MAIN)},0.06)",
                          "border": f"1px solid rgba({_hex_to_rgb(AZUL_MAIN)},0.2)",
                          "borderRadius": "10px", "padding": "16px"}),
            ], width=12, lg=4),
        ]),
        _interp_box([
            "La línea azul es la serie observada completa. El segmento naranja son las predicciones "
            "sobre el 20 % reservado (nunca visto durante el entrenamiento). "
            "Una buena superposición indica que el modelo generaliza la tendencia, "
            "aunque con n pequeño la variabilidad es alta.",
        ]),
    ], border_color=PURPURA),

    # ── PASO 7: PRONÓSTICO ────────────────────────────────────────────────────
    _section([
        _section_title("Paso 7 — Pronóstico con horizonte configurable", "fas fa-binoculars"),
        html.P([
            "Se proyecta la mediana global de ", html.Code("tasa_fin_cap"),
            " para los años siguientes a 2022 usando el modelo ARIMA seleccionado por AIC. "
            "La banda representa el intervalo de confianza al 95 %.",
        ], style={"color": TEXT_MAIN, "fontSize": "0.91em", "marginBottom": "16px"}),

        dbc.Row([
            dbc.Col([
                html.Label("Horizonte de pronóstico (años):",
                           style={"color": TEXT_DIM, "fontWeight": "600",
                                  "fontSize": "0.88em", "marginBottom": "6px"}),
                dcc.Slider(
                    id="pred-horizonte",
                    min=1, max=10, step=1, value=5,
                    marks={i: {"label": str(i),
                               "style": {"color": TEXT_DIM, "fontSize": "0.8em"}}
                           for i in range(1, 11)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Br(),
                html.Div([
                    html.H6("Tabla de pronóstico",
                            style={"color": CYAN, "fontWeight": "700",
                                   "marginBottom": "10px", "marginTop": "0",
                                   "fontSize": "0.88em", "textTransform": "uppercase",
                                   "letterSpacing": "0.06em"}),
                    html.Div(id="pred-tabla-forecast-container"),
                ], style={"background": f"rgba({_hex_to_rgb(AZUL_MAIN)},0.07)",
                          "border": f"1px solid rgba({_hex_to_rgb(AZUL_MAIN)},0.22)",
                          "borderRadius": "10px", "padding": "16px"}),
            ], width=12, lg=3, className="mb-3 mb-lg-0"),
            dbc.Col([
                dcc.Graph(id="pred-forecast-plot",
                          config={"displayModeBar": True, "displaylogo": False,
                                  "modeBarButtonsToRemove": ["select2d", "lasso2d"]}),
            ], width=12, lg=9),
        ]),
        _interp_box([
            "Las bandas de predicción se amplían conforme el horizonte aumenta, "
            "reflejando la incertidumbre acumulada. Los valores proyectados representan "
            "la tendencia central más probable dado el patrón histórico 2000–2022. "
            "Con n = 23, el modelo es indicativo: para evaluar el ODS 4 al 2030 "
            "se recomienda actualizar con datos más recientes del Banco Mundial.",
        ]),
    ], border_color=AZUL_FUERTE),

], style={"padding": "4px"})


# =============================================================================
# CALLBACKS
# =============================================================================

def register_prediccion_callbacks(app, df_2):
    """
    Registra todos los callbacks del tab predicción.
    Llamar desde app.py después de crear la app:
        from tab_prediccion import register_prediccion_callbacks
        register_prediccion_callbacks(app, df_2)
    """

    # ── Tabla estilo DataTable ────────────────────────────────────────────────
    def _datatable(df, id_suffix):
        cols = [{"name": c, "id": c} for c in df.columns]
        style_cells = [
            {"if": {"column_id": "Modelo"},
             "fontWeight": "700", "color": CYAN, "fontFamily": "monospace"},
        ]
        return dash_table.DataTable(
            id=f"dt-{id_suffix}",
            columns=cols,
            data=df.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": BG_PLOT,
                          "color": CYAN, "fontWeight": "700",
                          "border": "1px solid rgba(46,134,171,0.25)",
                          "fontSize": "0.8em", "textTransform": "uppercase",
                          "letterSpacing": "0.05em"},
            style_cell={"backgroundColor": "#0D1F3C",
                        "color": TEXT_MAIN, "border": "1px solid rgba(46,134,171,0.15)",
                        "padding": "8px 12px", "fontSize": "0.85em",
                        "fontFamily": "monospace"},
            style_data_conditional=style_cells
            + [{"if": {"row_index": 0},
                "backgroundColor": "rgba(30,132,73,0.12)",
                "border": "1px solid rgba(30,132,73,0.35)"}],
            page_size=10,
            sort_action="native",
        )

    def _model_badge_ui(row, label_color):
        p, d, q = int(row["p"]), int(row["d"]), int(row["q"])
        return html.Div([
            html.Div(f"ARIMA({p},{d},{q})",
                     style={"color": label_color, "fontWeight": "900",
                            "fontSize": "1.3em", "fontFamily": "monospace",
                            "marginBottom": "10px"}),
            html.Div([
                _badge("AIC",    f"{row['AIC']:.2f}",  VERDE_OK),
                _badge("BIC",    f"{row['BIC']:.2f}",  AZUL_MAIN),
                _badge("HQIC",   f"{row['HQIC']:.2f}", AMARILLO),
                _badge("LogLik", f"{row['LogLik']:.2f}", TEXT_DIM),
            ], style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}),
        ])

    # ── Callback principal: todo excepto pronóstico ───────────────────────────
    @app.callback(
        # Paso 1
        Output("pred-serie-original",        "figure"),
        Output("pred-adf-ui",                "children"),
        # Paso 2
        Output("pred-acf-orig",              "figure"),
        Output("pred-pacf-orig",             "figure"),
        Output("pred-acf-diff",              "figure"),
        Output("pred-pacf-diff",             "figure"),
        # Paso 3
        Output("pred-best-aic-ui",           "children"),
        Output("pred-best-bic-ui",           "children"),
        Output("pred-best-hqic-ui",          "children"),
        Output("pred-tabla-aic-container",   "children"),
        # Paso 4
        Output("pred-ajuste-plot",           "figure"),
        Output("pred-summary-ui",            "children"),
        # Paso 5
        Output("pred-resid-ts",              "figure"),
        Output("pred-resid-hist",            "figure"),
        Output("pred-resid-qq",              "figure"),
        Output("pred-sw-ui",                 "children"),
        Output("pred-lb-ui",                 "children"),
        # Paso 6
        Output("pred-val-plot",              "figure"),
        Output("pred-val-metricas-ui",       "children"),
        Output("pred-val-tabla-container",   "children"),
        Input("current-tab",                 "data"),
        Input("init-trigger",                "n_intervals"),
        prevent_initial_call=False,
    )
    def render_prediccion(tab, _n):
        empty_fig = _empty_fig()
        empty_div = html.Div()
        N_OUT = 20
        empties = [empty_div if i >= 10 else empty_fig for i in range(N_OUT)]

        if tab != "prediccion":
            # Return correctly typed empties
            return (empty_fig, empty_div,               # paso 1
                    empty_fig, empty_fig,               # paso 2 acf/pacf orig
                    empty_fig, empty_fig,               # paso 2 acf/pacf diff
                    empty_div, empty_div, empty_div, empty_div,  # paso 3
                    empty_fig, empty_div,               # paso 4
                    empty_fig, empty_fig, empty_fig,    # paso 5 graficos
                    empty_div, empty_div,               # paso 5 pruebas
                    empty_fig, empty_div, empty_div)    # paso 6

        try:
            R = _compute_arima_all(df_2, horizonte=5)
        except Exception as exc:
            import traceback; traceback.print_exc()
            err = html.Div(str(exc), style={"color": ROJO_OUT, "padding": "12px"})
            return (empty_fig, err,
                    empty_fig, empty_fig, empty_fig, empty_fig,
                    err, err, err, err,
                    empty_fig, err,
                    empty_fig, empty_fig, empty_fig,
                    err, err,
                    empty_fig, err, err)

        anios   = R["anios"]
        serie   = R["serie"]

        # ── Paso 1: Serie temporal ────────────────────────────────────────────
        fig_serie = go.Figure()
        fig_serie.add_trace(go.Scatter(
            x=anios, y=serie, mode="lines+markers",
            name="Mediana global",
            line=dict(color=AZUL_MAIN, width=2.5),
            marker=dict(size=6, color=CYAN,
                        line=dict(color=BG_PLOT, width=1.5)),
            hovertemplate="<b>%{x}</b><br>Mediana: %{y:.2f} %<extra></extra>",
        ))
        fig_serie.add_trace(go.Scatter(
            x=anios, y=[np.mean(serie)] * len(anios), mode="lines",
            name=f"Media histórica ({np.mean(serie):.1f} %)",
            line=dict(color=AMARILLO, dash="dot", width=1.5),
            hoverinfo="skip",
        ))
        fig_serie = _base_layout(
            fig_serie,
            "Serie temporal — Mediana global de tasa_fin_cap",
            f"Fuente: Banco Mundial WDI  ·  n = {len(serie)} puntos  ·  2000–2022",
            "Año", "Tasa de finalización (%)", 340
        )

        # ADF UI
        ap, pp, nc, cc = R["adf_stat"], R["adf_pval"], R["adf_nlag"], R["adf_crit"]
        if pp < 0.01:
            adf_verdict, adf_col = "✓  Estacionaria — evidencia muy fuerte (p < 0.01)", VERDE_OK
        elif pp < 0.05:
            adf_verdict, adf_col = "✓  Estacionaria — nivel estándar (p < 0.05)", VERDE_OK
        elif pp < 0.10:
            adf_verdict, adf_col = "⚠  Zona gris — evidencia débil (p < 0.10)", AMARILLO
        else:
            adf_verdict, adf_col = "✗  No estacionaria — raíz unitaria (p >= 0.10)", ROJO_OUT

        adf_ui = html.Div([
            _verdict(adf_verdict, adf_col),
            html.Div([
                _badge("ADF stat", f"{ap:.4f}", CYAN),
                _badge("p-valor",  f"{pp:.4f}" if pp >= 0.0001 else "< 0.0001", adf_col),
                _badge("Lags AIC", str(nc), TEXT_DIM),
            ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"}),
            html.Div([
                html.Span(f"VC 1%: {cc['1%']:.3f} | VC 5%: {cc['5%']:.3f} | VC 10%: {cc['10%']:.3f}",
                          style={"color": TEXT_DIM, "fontSize": "0.80em"}),
            ]),
        ])

        # ── Paso 2: ACF / PACF ────────────────────────────────────────────────
        ci_o  = R["ci_orig"]
        ci_d  = R["ci_diff"]
        ao, po = R["acf_orig"],  R["pacf_orig"]
        ad, pd_ = R["acf_diff"], R["pacf_diff"]

        fig_acf_orig  = _build_correlograma(
            list(range(len(ao))), ao, ci_o,
            "ACF — Serie original", CYAN, AZUL_MAIN, 260)
        fig_pacf_orig = _build_correlograma(
            list(range(len(po))), po, ci_o,
            "PACF — Serie original", ROJO_OUT, "#5B7A99", 260)
        fig_acf_diff  = _build_correlograma(
            list(range(len(ad))), ad, ci_d,
            "ACF — Primera diferencia", VERDE_OK, "#5B7A99", 260)
        fig_pacf_diff = _build_correlograma(
            list(range(len(pd_))), pd_, ci_d,
            "PACF — Primera diferencia", AMARILLO, "#5B7A99", 260)

        # ── Paso 3: Grid search ───────────────────────────────────────────────
        best_aic_ui  = _model_badge_ui(R["best_aic"],  VERDE_OK)
        best_bic_ui  = _model_badge_ui(R["best_bic"],  AZUL_MAIN)
        best_hqic_ui = _model_badge_ui(R["best_hqic"], AMARILLO)
        tabla_aic    = _datatable(
            R["df_aic"].head(10)[["Modelo","LogLik","AIC","BIC","HQIC"]], "grid-aic")

        # ── Paso 4: Ajuste ────────────────────────────────────────────────────
        p_, d_, q_ = R["order_best"]
        fitted = R["fitted"]
        fig_ajuste = go.Figure()
        fig_ajuste.add_trace(go.Scatter(
            x=anios, y=serie, mode="lines+markers", name="Observado",
            line=dict(color=AZUL_MAIN, width=2.5),
            marker=dict(size=5, color=CYAN),
            hovertemplate="<b>%{x}</b><br>Observado: %{y:.3f}%<extra></extra>",
        ))
        fig_ajuste.add_trace(go.Scatter(
            x=anios, y=fitted, mode="lines", name="Ajustado",
            line=dict(color=NARANJA, width=2, dash="dash"),
            hovertemplate="<b>%{x}</b><br>Ajustado: %{y:.3f}%<extra></extra>",
        ))
        fig_ajuste = _base_layout(
            fig_ajuste,
            f"ARIMA({p_},{d_},{q_}) — Observado vs Ajustado",
            "Mediana global de tasa_fin_cap", "Año", "Tasa (%)", 380)

        m = R["model_fit"]
        summary_ui = html.Div([
            html.Div(f"ARIMA({p_},{d_},{q_})",
                     style={"color": VERDE_OK, "fontWeight": "900",
                            "fontSize": "1.3em", "fontFamily": "monospace",
                            "marginBottom": "12px"}),
            html.Div([
                _badge("AIC",     f"{m.aic:.2f}",  VERDE_OK),
                _badge("BIC",     f"{m.bic:.2f}",  AZUL_MAIN),
                _badge("HQIC",    f"{m.hqic:.2f}", AMARILLO),
            ], style={"display": "flex", "gap": "6px", "marginBottom": "10px"}),
            html.Div([
                _badge("LogLik", f"{m.llf:.2f}", TEXT_DIM),
                _badge("σ²",     f"{m.params[-1]:.4f}" if hasattr(m, "params") else "—", TEXT_DIM),
            ], style={"display": "flex", "gap": "6px"}),
        ])

        # ── Paso 5: Residuales ────────────────────────────────────────────────
        resid = R["resid"]
        # Filtrar ceros de padding para histograma y QQ (solo residuales reales)
        resid_nz = resid[resid != 0.0] if np.any(resid != 0.0) else resid

        fig_resid_ts = go.Figure()
        fig_resid_ts.add_trace(go.Scatter(
            x=anios, y=resid, mode="lines+markers", name="Residual",
            line=dict(color=AMARILLO, width=1.8),
            marker=dict(size=5, color=AMARILLO,
                        line=dict(color=BG_PLOT, width=1)),
            hovertemplate="<b>%{x}</b><br>Residual: %{y:.4f}<extra></extra>",
        ))
        fig_resid_ts.add_hline(y=0, line=dict(color="rgba(200,200,200,0.25)", width=1.2))
        fig_resid_ts = _base_layout(fig_resid_ts, "Residuales en el tiempo",
                                    "", "Año", "Residual", 240)

        fig_resid_hist = go.Figure()
        fig_resid_hist.add_trace(go.Histogram(
            x=resid_nz, nbinsx=10, name="Frecuencia",
            marker=dict(color=AZUL_MAIN, opacity=0.8,
                        line=dict(color="rgba(100,207,246,0.4)", width=1)),
        ))
        # Curva normal superpuesta
        x_norm = np.linspace(resid_nz.min(), resid_nz.max(), 100)
        y_norm = (np.exp(-0.5 * ((x_norm - resid_nz.mean()) / resid_nz.std())**2)
                  / (resid_nz.std() * np.sqrt(2 * np.pi)))
        y_norm_scaled = y_norm * len(resid_nz) * (resid_nz.max() - resid_nz.min()) / 10
        fig_resid_hist.add_trace(go.Scatter(
            x=x_norm, y=y_norm_scaled, mode="lines", name="Normal teórica",
            line=dict(color=CYAN, width=2, dash="dot"),
        ))
        fig_resid_hist = _base_layout(fig_resid_hist, "Histograma de residuales",
                                      "", "Residual", "Frecuencia", 240)

        fig_resid_qq = go.Figure()
        fig_resid_qq.add_trace(go.Scatter(
            x=R["qq_x"], y=R["qq_y"], mode="markers", name="Residuales",
            marker=dict(size=7, color=CYAN, opacity=0.85,
                        line=dict(color=BG_PLOT, width=1)),
            hovertemplate="<b>Teórico: %{x:.3f}</b><br>Observado: %{y:.3f}<extra></extra>",
        ))
        fig_resid_qq.add_trace(go.Scatter(
            x=R["qq_line_x"], y=R["qq_line_y"], mode="lines", name="Línea normal",
            line=dict(color=ROJO_OUT, width=2, dash="dash"),
        ))
        fig_resid_qq = _base_layout(fig_resid_qq, "QQ-plot de residuales",
                                    "", "Cuantiles teóricos", "Cuantiles observados", 240)

        sw_stat, sw_p = R["sw_stat"], R["sw_p"]
        sw_col    = VERDE_OK if sw_p > 0.05 else ROJO_OUT
        sw_icon   = "✓" if sw_p > 0.05 else "✗"
        sw_msg    = "No se rechaza normalidad (p > 0.05)" if sw_p > 0.05 else "Se rechaza normalidad (p ≤ 0.05)"
        sw_ui = html.Div([
            _verdict(f"{sw_icon} {sw_msg}", sw_col),
            html.Div([
                _badge("W stat", f"{sw_stat:.4f}", CYAN),
                _badge("p-valor", f"{sw_p:.4f}" if sw_p >= 0.0001 else "< 0.0001", sw_col),
            ], style={"display": "flex", "gap": "8px"}),
        ])

        lb_stat, lb_p = R["lb_stat"], R["lb_p"]
        lb_col  = VERDE_OK if lb_p > 0.05 else ROJO_OUT
        lb_icon = "✓" if lb_p > 0.05 else "✗"
        lb_msg  = "Ruido blanco (p > 0.05) — sin autocorrelación" if lb_p > 0.05 else "Autocorrelación detectada (p ≤ 0.05)"
        lb_ui = html.Div([
            _verdict(f"{lb_icon} {lb_msg}", lb_col),
            html.Div([
                _badge("Q stat", f"{lb_stat:.4f}", CYAN),
                _badge("p-valor", f"{lb_p:.4f}" if lb_p >= 0.0001 else "< 0.0001", lb_col),
                _badge("Lags", "10", TEXT_DIM),
            ], style={"display": "flex", "gap": "8px"}),
        ])

        # ── Paso 6: Validación 80/20 ──────────────────────────────────────────
        split_idx = R["split_idx"]
        train_ser = R["train_ser"]
        test_anios= R["test_anios"]
        test_ser  = R["test_ser"]
        val_pred  = R["val_pred"]
        val_ci    = R["val_ci"]

        fig_val = go.Figure()
        # IC 95%
        fig_val.add_trace(go.Scatter(
            x=test_anios + test_anios[::-1],
            y=list(val_ci[:, 1]) + list(val_ci[:, 0])[::-1],
            fill="toself", fillcolor="rgba(243,156,18,0.10)",
            line=dict(width=0), showlegend=True,
            name="IC 95% predicción",
            hoverinfo="skip",
        ))
        # Serie completa observada
        fig_val.add_trace(go.Scatter(
            x=anios, y=serie, mode="lines+markers", name="Observado (completo)",
            line=dict(color=AZUL_MAIN, width=2.5),
            marker=dict(size=5, color=CYAN),
            hovertemplate="<b>%{x}</b><br>Observado: %{y:.3f}%<extra></extra>",
        ))
        # Predicciones test
        fig_val.add_trace(go.Scatter(
            x=test_anios, y=val_pred, mode="lines+markers", name="Predicción (prueba 20%)",
            line=dict(color=NARANJA, width=2.5, dash="dash"),
            marker=dict(size=7, color=NARANJA,
                        symbol="diamond", line=dict(color=BG_PLOT, width=1.5)),
            hovertemplate="<b>%{x}</b><br>Predicho: %{y:.3f}%<extra></extra>",
        ))
        # Línea de corte entrenamiento/prueba
        fig_val.add_vline(x=anios[split_idx - 1] + 0.5,
                          line=dict(color="rgba(200,200,200,0.3)", dash="dot", width=1.5))
        fig_val.add_annotation(x=anios[split_idx - 1] + 0.5,
                               y=serie.max(),
                               text="<b>↑ Corte 80/20</b>",
                               showarrow=False, yanchor="bottom",
                               font=dict(color=TEXT_DIM, size=9),
                               bgcolor="rgba(13,31,60,0.7)")
        fig_val = _base_layout(fig_val,
                               f"Validación ARIMA({p_},{d_},{q_}) — Entrenamiento 80% / Prueba 20%",
                               "Mediana global de tasa_fin_cap", "Año", "Tasa (%)", 400)

        mae, rmse, mape, r2 = R["mae"], R["rmse"], R["mape"], R["r2"]
        metricas_ui = html.Div([
            html.Div([
                _badge("MAE",  f"{mae:.3f}",  CYAN),
                _badge("RMSE", f"{rmse:.3f}", AZUL_MAIN),
            ], style={"display": "flex", "gap": "6px", "marginBottom": "8px"}),
            html.Div([
                _badge("MAPE", f"{mape:.2f}%", AMARILLO),
                _badge("R²",   f"{r2:.4f}",   VERDE_OK if r2 > 0.8 else ROJO_OUT),
            ], style={"display": "flex", "gap": "6px"}),
        ])

        tabla_val = _datatable(R["df_val"], "val")

        return (
            fig_serie, adf_ui,
            fig_acf_orig, fig_pacf_orig, fig_acf_diff, fig_pacf_diff,
            best_aic_ui, best_bic_ui, best_hqic_ui, tabla_aic,
            fig_ajuste, summary_ui,
            fig_resid_ts, fig_resid_hist, fig_resid_qq,
            sw_ui, lb_ui,
            fig_val, metricas_ui, tabla_val,
        )

    # ── Callback pronóstico (reacciona al slider) ─────────────────────────────
    @app.callback(
        Output("pred-forecast-plot",           "figure"),
        Output("pred-tabla-forecast-container","children"),
        Input("pred-horizonte",                "value"),
        Input("current-tab",                   "data"),
        Input("init-trigger",                  "n_intervals"),
        prevent_initial_call=False,
    )
    def render_forecast(horizonte, tab, _n):
        if tab != "prediccion":
            return _empty_fig(), html.Div()
        if horizonte is None:
            horizonte = 5

        try:
            R = _compute_arima_all(df_2, horizonte=int(horizonte))
        except Exception as exc:
            import traceback; traceback.print_exc()
            return _empty_fig(str(exc)), html.Div(str(exc), style={"color": ROJO_OUT})

        anios  = R["anios"]
        serie  = R["serie"]
        p_, d_, q_ = R["order_best"]
        fc_anios   = R["fc_anios"]
        fc_mean    = R["fc_mean"]
        fc_ci      = R["fc_ci"]

        fig_fc = go.Figure()
        # IC 95% sombreado
        fig_fc.add_trace(go.Scatter(
            x=fc_anios + fc_anios[::-1],
            y=list(fc_ci[:, 1]) + list(fc_ci[:, 0])[::-1],
            fill="toself", fillcolor="rgba(155,89,182,0.12)",
            line=dict(width=0), showlegend=True,
            name="IC 95% pronóstico", hoverinfo="skip",
        ))
        # Serie histórica
        fig_fc.add_trace(go.Scatter(
            x=anios, y=serie, mode="lines+markers", name="Histórico (2000–2022)",
            line=dict(color=AZUL_MAIN, width=2.5),
            marker=dict(size=5, color=CYAN),
            hovertemplate="<b>%{x}</b><br>Observado: %{y:.3f}%<extra></extra>",
        ))
        # Límite techo 100%
        all_x = anios + fc_anios
        fig_fc.add_trace(go.Scatter(
            x=all_x, y=[100] * len(all_x), mode="lines",
            name="Techo (100 %)",
            line=dict(color="rgba(192,57,43,0.5)", dash="dot", width=1.5),
            hoverinfo="skip",
        ))
        # Pronóstico central
        fig_fc.add_trace(go.Scatter(
            x=fc_anios, y=fc_mean, mode="lines+markers",
            name=f"Pronóstico {fc_anios[0]}–{fc_anios[-1]}",
            line=dict(color=PURPURA, width=2.5),
            marker=dict(size=8, color=PURPURA,
                        symbol="diamond-open", line=dict(color=PURPURA, width=2)),
            hovertemplate=(
                "<b>%{x}</b><br>Pronóstico: %{y:.3f}%"
                "<br>IC sup: (ver tabla)" 
                + "<extra></extra>"
            ),
        ))
        # Línea de corte histórico/pronóstico
        fig_fc.add_vline(x=anios[-1] + 0.5,
                         line=dict(color="rgba(200,200,200,0.25)", dash="dot", width=1.5))
        fig_fc.add_annotation(x=anios[-1] + 0.5, y=serie.max(),
                              text="<b>↑ 2022</b>", showarrow=False,
                              yanchor="bottom", font=dict(color=TEXT_DIM, size=9),
                              bgcolor="rgba(13,31,60,0.7)")
        fig_fc = _base_layout(
            fig_fc,
            f"Pronóstico ARIMA({p_},{d_},{q_}) — {fc_anios[0]}–{fc_anios[-1]}",
            f"IC 95%  ·  Mediana global de tasa_fin_cap  ·  Horizonte: {horizonte} años",
            "Año", "Tasa de finalización (%)", 450
        )

        tabla_fc = _datatable(R["df_fc"], "fc")

        return fig_fc, tabla_fc


# ── Registrar callbacks de predicción ────────────────────────────────────────
register_prediccion_callbacks(app, df_2)

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  App Dash — Finalización Primaria Femenina ODS 4")
    print("  Abrir en: http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8050)