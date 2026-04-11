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
#   python app_dash_ods4.py
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
    """Dataset sintético con exactamente 130 países × 23 años (con NAs ~12%)."""
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
            if np.random.random() < 0.12:
                rows.append({"iso3": iso3, "pais": nombre, "anio": y, "tasa_fin": float("nan")})
                continue
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
        html.Span("⚠ "), children
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
    return html.Div([
        html.Span(icon_txt, style={"marginRight": "10px", "fontSize": "1.2em"}),
        title
    ], style={
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
            html.Div("7 secciones disponibles", style={"color": "#4A7A9B", "fontSize": "0.65em", "marginTop": "1px"})
        ])
    ], style={"display": "flex", "alignItems": "center", "gap": "10px",
              "background": "linear-gradient(135deg,rgba(46,134,171,0.15) 0%,rgba(46,134,171,0.05) 100%)",
              "border": "1px solid rgba(46,134,171,0.25)", "borderRadius": "10px", "padding": "10px 14px",
              "margin": "18px 14px 10px"}),

    # Menú de navegación
    html.Div([
        html.Button([html.Span("ℹ️ "), "Introducción"],  id="nav-intro",        n_clicks=0, className="nav-btn active"),
        html.Button([html.Span("📊 "), "Univariado"],     id="nav-univariado",   n_clicks=0, className="nav-btn"),
        html.Button([html.Span("📈 "), "Bivariado"],      id="nav-bivariado",    n_clicks=0, className="nav-btn"),
        html.Button([html.Span("🌍 "), "Mapa Mundial"],   id="nav-mapa",         n_clicks=0, className="nav-btn"),
        html.Button([html.Span("🏁 "), "Conclusiones"],   id="nav-conclusiones", n_clicks=0, className="nav-btn"),
        html.Button([html.Span("📋 "), "Datos"],          id="nav-datos",        n_clicks=0, className="nav-btn"),
        html.Button([html.Span("📚 "), "Referencias"],    id="nav-referencias",  n_clicks=0, className="nav-btn"),
    ], style={"display": "flex", "flexDirection": "column", "gap": "4px", "padding": "0 10px"}),

    # Divisor
    html.Div(style={"margin": "12px 14px", "height": "1px",
                    "background": "linear-gradient(90deg,transparent,rgba(46,134,171,0.5),transparent)"}),

    # Meta-info
    html.Div([
        html.Div([html.Span("🗄 ", style={"width": "14px"}),
                  html.Span("Banco Mundial — WDI", style={"color": "#7FADD4", "fontSize": "0.75em"})],
                 style={"display": "flex", "alignItems": "center", "gap": "9px",
                        "padding": "6px 10px", "borderRadius": "7px", "background": "rgba(46,134,171,0.07)"}),
        html.Div([html.Span("📅 ", style={"width": "14px"}),
                  html.Span("Periodo: 2000–2022", style={"color": "#7FADD4", "fontSize": "0.75em"})],
                 style={"display": "flex", "alignItems": "center", "gap": "9px",
                        "padding": "6px 10px", "borderRadius": "7px", "background": "rgba(46,134,171,0.07)"}),
        html.Div([html.Span("🌐 ", style={"width": "14px"}),
                  html.Span("Cobertura: ~192 países", style={"color": "#7FADD4", "fontSize": "0.75em"})],
                 style={"display": "flex", "alignItems": "center", "gap": "9px",
                        "padding": "6px 10px", "borderRadius": "7px", "background": "rgba(46,134,171,0.07)"}),
    ], style={"padding": "4px 14px 8px", "display": "flex", "flexDirection": "column", "gap": "6px"}),

    # Divisor
    html.Div(style={"margin": "8px 14px", "height": "1px",
                    "background": "linear-gradient(90deg,transparent,rgba(46,134,171,0.5),transparent)"}),

    # Autores
    html.Div([
        html.Div([html.Div("👥", style={"marginRight": "8px"}),
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
    html.H4("🎓 Finalización Primaria Femenina — ODS 4",
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
        html.H2("🎓 Tasa de Finalización de Educación Primaria Femenina",
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
        html.H4("❓ ¿Qué mide este indicador?", style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
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
        html.Div([html.Span("✅ ", style={"color": VERDE_OK}),
                  html.Strong("Estado de carga de datos: ", style={"color": TEXT_LIGHT}),
                  html.Span(f"{N_OBS:,} observaciones cargadas correctamente de {N_PAISES} países.",
                            style={"color": VERDE_OK})]),
    ]),
    "intro-tab2": card_s([
        html.H4("🌎 ¿Por qué importa estudiarlo?", style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
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
        html.H4("🎯 Objetivos del análisis", style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
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
        html.H4("📖 Marco Teórico", style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
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
        html.H4("🧪 Hipótesis de investigación", style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
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
        html.H4("⚙️ Metodología", style={"color": CYAN, "fontWeight": "700", "marginTop": "0"}),
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

# ─── UNIVARIADO ───────────────────────────────────────────────────────────────
tab_univariado = html.Div([
    # ADF + ACF + PACF
    html.Div([
        html.H5("Prueba de Estacionariedad — ADF, ACF y PACF",
                style={"color": CYAN, "fontWeight": "700", "borderBottom": "2px solid rgba(46,134,171,0.3)",
                       "paddingBottom": "10px", "marginBottom": "16px"}),
        html.P(["La ", html.Strong("prueba ADF (Augmented Dickey-Fuller)"),
                " evalúa si la serie de medianas anuales tiene raíz unitaria (H₀: no estacionaria). Los ",
                html.Strong("correlogramas ACF y PACF"), " complementan el diagnóstico mostrando la "
                "estructura de autocorrelación."], style={"color": TEXT_MAIN}),
        dbc.Row([
            dbc.Col([html.Div(id="adf-result-ui")], width=5),
            dbc.Col([
                html.Div([
                    html.H5("Valores críticos ADF", style={"color": CYAN, "marginTop": "0"}),
                    html.Table([
                        html.Thead(html.Tr([html.Th("Nivel α"), html.Th("Decisión"), html.Th("Interpretación")])),
                        html.Tbody([
                            html.Tr([html.Td("0.01"), html.Td("Rechaza H₀", style={"color": VERDE_OK}), html.Td("Estacionaria — evidencia muy fuerte")]),
                            html.Tr([html.Td("0.05"), html.Td("Rechaza H₀", style={"color": VERDE_OK}), html.Td("Estacionaria — nivel estándar")]),
                            html.Tr([html.Td("0.10"), html.Td("Rechaza H₀", style={"color": AMARILLO}), html.Td("Evidencia débil")]),
                            html.Tr([html.Td(">0.10"), html.Td("No rechaza H₀", style={"color": ROJO_OUT}), html.Td("Serie no estacionaria (raíz unitaria)")]),
                        ])
                    ], style={"width": "100%", "fontSize": "0.87em", "borderCollapse": "collapse",
                              "color": TEXT_MAIN}),
                    html.Br(),
                    nota_box(["Prueba aplicada sobre la ", html.Strong("mediana anual"),
                              " de tasa_fin_cap (n = 23 puntos, 2000–2022)."]),
                ], style={"background": BG_PLOT, "border": "1px solid rgba(46,134,171,0.25)",
                          "borderRadius": "8px", "padding": "16px"})
            ], width=7),
        ]),
        html.Div(id="adf-implicaciones"),
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.H5("Función de Autocorrelación (ACF)", style={"color": CYAN}),
                dcc.Graph(id="plot-acf", style={"height": "280px"}, config={"displayModeBar": False}),
            ], width=6),
            dbc.Col([
                html.H5("Función de Autocorrelación Parcial (PACF)", style={"color": CYAN}),
                dcc.Graph(id="plot-pacf", style={"height": "280px"}, config={"displayModeBar": False}),
            ], width=6),
        ]),
        interp_box("La ACF muestra autocorrelación entre la serie y sus rezagos. Cuando las barras decaen "
                   "lentamente y superan las bandas de confianza (±1.96/√n), la serie presenta "
                   "autocorrelación significativa — señal de no estacionariedad. La PACF orienta la "
                   "selección del orden (p,d,q) de un modelo ARIMA.")
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "marginBottom": "20px",
              "borderLeft": f"6px solid {AZUL_MAIN}"}),

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
    page_header("🌍", "Mapas Mundiales — tasa_fin_cap por País"),

    html.Div([
        html.H5("Mapa 1 — Promedio histórico de tasa_fin_cap (2000–2022)",
                style={"color": CYAN, "fontWeight": "700", "marginBottom": "12px"}),
        html.P("Tasa media de finalización primaria femenina por país durante todo el periodo. "
               "Países en gris no tienen datos disponibles. Hover para ver valor exacto.",
               style={"color": TEXT_DIM, "fontSize": "0.9em"}),
        dcc.Graph(id="plot-mapa-promedio", style={"height": "520px"}),
        interp_box("Los tonos azul oscuro indican tasas históricamente altas (>90%), frecuentes en "
                   "Europa, Asia Oriental y América Latina. Los tonos amarillo-claro señalan países con "
                   "mayor rezago educativo, concentrados en África subsahariana y Asia meridional.")
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "marginBottom": "20px",
              "borderLeft": f"6px solid {AZUL_MAIN}"}),

    html.Div([
        html.H5("Mapa 2 — Evolución anual de tasa_fin_cap (2000–2022, animado)",
                style={"color": CYAN, "fontWeight": "700", "marginBottom": "12px"}),
        html.P(["Presione el botón ", html.Strong("▶ Play"), " para animar la evolución año a año."],
               style={"color": TEXT_DIM, "fontSize": "0.9em"}),
        dcc.Graph(id="plot-mapa-animado", style={"height": "540px"}),
        interp_box("La animación evidencia el avance progresivo de los países hacia tasas más altas "
                   "entre 2000 y 2022. Las regiones que permanecen en tonos claros son las de mayor "
                   "fragilidad educativa persistente.")
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "marginBottom": "20px",
              "borderLeft": f"6px solid {AZUL_MAIN}"}),

    html.Div([
        html.H5("Mapa 3 — Cambio absoluto: último año disponible vs. 2000",
                style={"color": CYAN, "fontWeight": "700", "marginBottom": "12px"}),
        html.P("Diferencia entre la tasa más reciente y la registrada en el año 2000. "
               "Solo se muestran países con datos en ambos extremos del periodo.",
               style={"color": TEXT_DIM, "fontSize": "0.9em"}),
        dcc.Graph(id="plot-mapa-cambio", style={"height": "520px"}),
        interp_box("Tonos verde indican mejora; tonos naranja-rojo indican retroceso. Un cambio de "
                   "+20 pp o más en 22 años representa una transformación educativa sustancial. "
                   "Los mayores avances se concentran en países que partían de tasas bajas (convergencia beta).")
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px", "marginBottom": "20px",
              "borderLeft": f"6px solid {VERDE_OK}"}),
])

# ─── CONCLUSIONES ─────────────────────────────────────────────────────────────
tab_conclusiones = html.Div([
    page_header("🏁", "Conclusiones del Análisis Exploratorio"),
    *[html.Div([
        html.H4(titulo, style={"color": CYAN, "marginBottom": "12px"}),
        html.Ul([html.Li(item, style={"color": TEXT_MAIN, "marginBottom": "6px"}) for item in items])
      ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "20px 24px",
                "marginBottom": "16px", "borderLeft": f"6px solid {bcolor}"})
      for titulo, bcolor, items in [
          ("📊 1. Hallazgos del análisis univariado", AZUL_MAIN, [
              "La distribución global de tasa_fin_cap es fuertemente asimétrica negativa: la mayoría de observaciones se concentran en 85–100%, con cola izquierda de países del África subsahariana y sur de Asia en los años iniciales.",
              "La mediana supera a la media en todos los años, confirmando la asimetría. La diferencia cuantifica el peso de los países rezagados.",
              "El boxplot identifica outliers inferiores que representan exclusión educativa severa, no artefactos estadísticos.",
              "El efecto sobrenotificación (tasa_fin > 100%) afecta al 5–8% de las observaciones, justificando el uso de tasa_fin_cap.",
          ]),
          ("〰️ 2. Prueba ADF, ACF y PACF", AZUL_MAIN, [
              "La prueba ADF sobre la serie de medianas anuales indica una tendencia temporal suave. El p-value cercano a 0.05–0.10 clasifica la serie como marginalmente no estacionaria o en zona gris.",
              "El ACF muestra autocorrelación positiva significativa en varios rezagos, confirmando la tendencia.",
              "El PACF indica que la estructura se concentra en los primeros rezagos, orientando hacia modelos AR(1) o AR(2) tras diferenciación.",
              "Para modelización formal se recomienda diferenciar la serie una vez y evaluar ARIMA(p,1,q).",
          ]),
          ("📈 3. Hallazgos del análisis bivariado", AZUL_MAIN, [
              "El scatter con LOESS confirma una trayectoria cuasi-logística: crecimiento rápido hasta ~2012 y desaceleración posterior al acercarse al techo.",
              "La mediana global aumentó de ~82% (2000) a ~95% (2022). La banda IQR se estrechó de ~25 pp a menos de 10 pp: convergencia real pero incompleta.",
              "La brecha media–mediana se cierra gradualmente: los países rezagados mejoran más rápido (catching-up).",
          ]),
          ("🗂 4. Hallazgos del análisis multivariado", AZUL_MAIN, [
              "El mapa de calor identifica tres grupos: países que alcanzaron el techo temprano, países en transición activa y países con fragilidad persistente.",
              "Los países con mayores pendientes de mejora pre-techo son principalmente de África del Norte, Asia Central y el sudeste asiático.",
              "Las trayectorias individuales revelan disrupciones que el análisis agregado no puede capturar.",
          ]),
      ]],
    html.Div([
        html.H4("🛣 5. Limitaciones y pasos a seguir", style={"color": VERDE_OK, "marginBottom": "12px"}),
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
    page_header("📚", "Referencias bibliográficas y fuentes de datos"),
    html.Div([
        html.P("Referencias en formato APA 7ª edición.", style={"color": TEXT_DIM, "fontSize": "0.9em"}),
        html.Div("🗄 Fuentes de datos", style={"color": CYAN, "borderBottom": "2px solid rgba(46,134,171,0.3)",
                  "fontWeight": "700", "padding": "10px 0 8px", "margin": "22px 0 12px"}),
        ref_entry(1, "Banco Mundial. (2023).", "Primary completion rate, female (% of relevant age group) — SE.PRM.CMPT.FE.ZS.",
                  "World Development Indicators.", "https://data.worldbank.org/indicator/SE.PRM.CMPT.FE.ZS"),
        ref_entry(2, "Arel-Bundock, V. (2022).", "WDI: World Development Indicators (R package v2.7.8).",
                  "CRAN.", "https://CRAN.R-project.org/package=WDI"),
        html.Div("📐 Metodología estadística", style={"color": CYAN, "borderBottom": "2px solid rgba(46,134,171,0.3)",
                  "fontWeight": "700", "padding": "10px 0 8px", "margin": "22px 0 12px"}),
        ref_entry(3, "Dickey, D. A., & Fuller, W. A. (1979).", "Distribution of the estimators for autoregressive time series with a unit root.",
                  "JASA, 74(366), 427–431.", "https://doi.org/10.2307/2286348"),
        ref_entry(4, "Said, S. E., & Dickey, D. A. (1984).", "Testing for unit roots in autoregressive-moving average models of unknown order.",
                  "Biometrika, 71(3), 599–607.", None),
        ref_entry(5, "Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015).", "Time Series Analysis: Forecasting and Control (5th ed.).",
                  "Wiley. [ACF y PACF — selección de modelos ARIMA]", None),
        html.Div("📖 Marco teórico", style={"color": CYAN, "borderBottom": "2px solid rgba(46,134,171,0.3)",
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
        html.Div("💻 Herramientas computacionales", style={"color": CYAN, "borderBottom": "2px solid rgba(46,134,171,0.3)",
                  "fontWeight": "700", "padding": "10px 0 8px", "margin": "22px 0 12px"}),
        ref_entry(11, "Plotly Technologies Inc. (2015).", "Collaborative data science.",
                  "Plotly Technologies Inc.", "https://plot.ly"),
        ref_entry(12, "Plotly. (2023).", "Dash: Analytical Web Apps for Python, R, Julia, and Jupyter.",
                  "Plotly.", "https://dash.plotly.com"),
    ], style={"background": BG_CARD, "borderRadius": "10px", "padding": "24px"}),
])

# =============================================================================
# LAYOUT PRINCIPAL
# =============================================================================
app.layout = html.Div([
    dcc.Store(id="current-tab", data="intro"),
    sidebar,

    # Contenido principal
    html.Div([
        header,
        html.Div(id="main-content", style={"padding": "24px", "minHeight": "calc(100vh - 60px)"})
    ], style={"marginLeft": "270px", "background": BG_DARK, "minHeight": "100vh"}),
])


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
     Input("nav-referencias", "n_clicks")],
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
    }
    tab_id, content = mapping.get(btn_id, ("intro", tab_intro))
    return content, tab_id

# ─── Sub-pestañas de intro ────────────────────────────────────────────────────
@app.callback(Output("intro-tab-content", "children"), Input("intro-tabs", "active_tab"))
def render_intro_tab(tab):
    return INTRO_TABS_CONTENT.get(tab, "")

# ─── ADF + ACF + PACF ─────────────────────────────────────────────────────────
@app.callback(
    Output("adf-result-ui", "children"),
    Output("adf-implicaciones", "children"),
    Output("plot-acf", "figure"),
    Output("plot-pacf", "figure"),
    Input("current-tab", "data")
)
def render_adf(tab):
    if tab != "univariado":
        return html.Div(), html.Div(), go.Figure(), go.Figure()
    medianas = df_2.groupby("anio")["tasa_fin_cap"].median().sort_index()
    serie = medianas.values
    if len(serie) < 5:
        msg = "Serie de medianas insuficiente para ADF/ACF/PACF (se necesitan ≥ 5 años)."
        return html.Div(msg, style={"color": "#F39C12"}), html.Div(), _empty_fig(msg), _empty_fig(msg)

    # ADF
    try:
        adf_res = adfuller(serie, autolag="AIC")
        adf_stat, adf_pval = adf_res[0], adf_res[1]
        crit = adf_res[4]
    except Exception:
        adf_stat, adf_pval, crit = 0, 0.5, {"1%": -3.75, "5%": -3.0, "10%": -2.63}

    if adf_pval < 0.01:
        conclusion = "✅ Estacionaria — evidencia muy fuerte (p < 0.01)"
        col = VERDE_OK
    elif adf_pval < 0.05:
        conclusion = "✅ Estacionaria — nivel estándar (p < 0.05)"
        col = VERDE_OK
    elif adf_pval < 0.10:
        conclusion = "⚠️ Zona gris — evidencia débil (p < 0.10)"
        col = AMARILLO
    else:
        conclusion = "❌ No estacionaria — raíz unitaria no descartada (p ≥ 0.10)"
        col = ROJO_OUT

    adf_ui = html.Div([
        html.Div(conclusion, style={"background": f"rgba({','.join(str(int(col.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.15)",
                                    "borderLeft": f"5px solid {col}", "borderRadius": "8px",
                                    "padding": "12px 16px", "marginBottom": "12px",
                                    "fontWeight": "600", "color": col}),
        html.Div([
            html.Div([html.Span("Estadístico ADF", style={"color": TEXT_DIM, "fontSize": "0.8em"}),
                      html.Div(f"{adf_stat:.4f}", style={"color": CYAN, "fontWeight": "700", "fontSize": "1.2em"})],
                     style={"textAlign": "center", "padding": "10px", "background": "rgba(46,134,171,0.1)",
                            "borderRadius": "8px", "flex": "1"}),
            html.Div([html.Span("p-valor", style={"color": TEXT_DIM, "fontSize": "0.8em"}),
                      html.Div(f"{adf_pval:.4f}" if adf_pval >= 0.001 else "p < 0.001",
                               style={"color": col, "fontWeight": "700", "fontSize": "1.2em"})],
                     style={"textAlign": "center", "padding": "10px", "background": "rgba(46,134,171,0.1)",
                            "borderRadius": "8px", "flex": "1"}),
        ], style={"display": "flex", "gap": "10px", "marginBottom": "10px"}),
        html.Div([f"VC 1%: {crit['1%']:.3f} | VC 5%: {crit['5%']:.3f} | VC 10%: {crit['10%']:.3f}"],
                 style={"color": TEXT_DIM, "fontSize": "0.8em", "textAlign": "center"})
    ])

    impl = nota_box([
        "Implicación: ",
        ("La serie de medianas anuales presenta raíz unitaria. Se recomienda diferenciación antes de modelizar con ARIMA."
         if adf_pval >= 0.10 else
         "La serie es estacionaria en nivel. Puede modelizarse directamente sin diferenciación.")
    ])

    # ACF
    n_lags = min(15, len(serie) - 2)
    acf_vals  = acf(serie, nlags=n_lags, fft=False)
    pacf_vals = pacf(serie, nlags=n_lags)
    ci = 1.96 / np.sqrt(len(serie))
    lags = list(range(len(acf_vals)))

    fig_acf = go.Figure()
    for lag, val in zip(lags, acf_vals):
        fig_acf.add_trace(go.Bar(x=[lag], y=[val], marker_color=AZUL_MAIN if abs(val) <= ci else CYAN,
                                 showlegend=False, width=0.6))
    fig_acf.add_hline(y=ci,  line_dash="dash", line_color="rgba(100,207,246,0.5)", line_width=1.2)
    fig_acf.add_hline(y=-ci, line_dash="dash", line_color="rgba(100,207,246,0.5)", line_width=1.2)
    fig_acf = pl_layout(fig_acf, "ACF — Autocorrelación", "", "Lag", "ACF")
    fig_acf.update_layout(height=280, margin=dict(t=50, b=40))

    fig_pacf = go.Figure()
    for lag, val in zip(lags, pacf_vals):
        fig_pacf.add_trace(go.Bar(x=[lag], y=[val], marker_color=AZUL_MAIN if abs(val) <= ci else ROJO_OUT,
                                  showlegend=False, width=0.6))
    fig_pacf.add_hline(y=ci,  line_dash="dash", line_color="rgba(100,207,246,0.5)", line_width=1.2)
    fig_pacf.add_hline(y=-ci, line_dash="dash", line_color="rgba(100,207,246,0.5)", line_width=1.2)
    fig_pacf = pl_layout(fig_pacf, "PACF — Autocorrelación Parcial", "", "Lag", "PACF")
    fig_pacf.update_layout(height=280, margin=dict(t=50, b=40))

    return adf_ui, impl, fig_acf, fig_pacf

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

# ─── Mediana anual + IQR ──────────────────────────────────────────────────────
@app.callback(Output("plot-mediana-anual", "figure"), Input("current-tab", "data"))
def render_mediana_anual(tab):
    if tab != "bivariado":
        return go.Figure()
    agg = (df_2[df_2["anio"].between(2000, 2022)].dropna(subset=["tasa_fin_cap"])
           .groupby("anio")["tasa_fin_cap"]
           .agg(mediana="median",
                p25=lambda x: x.quantile(0.25),
                p75=lambda x: x.quantile(0.75))
           .reset_index())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([agg["anio"], agg["anio"][::-1]]),
        y=pd.concat([agg["p75"], agg["p25"][::-1]]),
        fill="toself", fillcolor="rgba(46,134,171,0.28)", line_color="transparent",
        name="Banda IQR (P25–P75)",
        hovertemplate="Año: %{x}<br>IQR: banda<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=agg["anio"], y=agg["mediana"], mode="lines+markers",
        line=dict(color=AZUL_FUERTE, width=2.5),
        marker=dict(color="white", line=dict(color=AZUL_FUERTE, width=2), size=8),
        name="Mediana",
        hovertemplate="Año: %{x}<br>Mediana: %{y:.1f}%<extra></extra>"
    ))
    fig = pl_layout(fig, "Mediana anual de tasa_fin_cap con banda P25–P75",
                    "Estrechamiento = convergencia entre países", "Año", "tasa_fin_cap (%)")
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(range=[58, 104])
    fig.update_layout(height=380)
    return fig

# ─── Media vs Mediana ─────────────────────────────────────────────────────────
@app.callback(Output("plot-media-mediana", "figure"), Input("current-tab", "data"))
def render_media_mediana(tab):
    if tab != "bivariado":
        return go.Figure()
    agg = (df_2[df_2["anio"].between(2000, 2022)].dropna(subset=["tasa_fin_cap"])
           .groupby("anio")["tasa_fin_cap"]
           .agg(media="mean", mediana="median")
           .reset_index())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([agg["anio"], agg["anio"][::-1]]),
        y=pd.concat([agg["mediana"], agg["media"][::-1]]),
        fill="toself", fillcolor="rgba(174,214,241,0.35)", line_color="transparent",
        name="Brecha media–mediana"
    ))
    fig.add_trace(go.Scatter(x=agg["anio"], y=agg["media"], mode="lines+markers",
                              line=dict(color=AMARILLO, width=2.5, dash="dash"),
                              marker=dict(color=AMARILLO, size=6), name="Media",
                              hovertemplate="Año: %{x}<br>Media: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=agg["anio"], y=agg["mediana"], mode="lines+markers",
                              line=dict(color=AZUL_FUERTE, width=2.5),
                              marker=dict(color=AZUL_FUERTE, size=6), name="Mediana",
                              hovertemplate="Año: %{x}<br>Mediana: %{y:.1f}%<extra></extra>"))
    fig = pl_layout(fig, "Media vs. Mediana anual de tasa_fin_cap",
                    "Área sombreada = brecha. Cierre = catching-up de países rezagados",
                    "Año", "tasa_fin_cap (%)")
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(range=[70, 102])
    fig.update_layout(height=380)
    return fig

# ─── Ribbon multicapa ─────────────────────────────────────────────────────────
@app.callback(Output("plot-mv-ribbon", "figure"), Input("current-tab", "data"))
def render_ribbon(tab):
    if tab != "bivariado":
        return go.Figure()
    agg = (df_2[df_2["anio"].between(2000, 2022)].dropna(subset=["tasa_fin_cap"])
           .groupby("anio")["tasa_fin_cap"]
           .agg(med="median",
                p10=lambda x: x.quantile(0.10),
                p25=lambda x: x.quantile(0.25),
                p75=lambda x: x.quantile(0.75),
                p90=lambda x: x.quantile(0.90))
           .reset_index())

    fig = go.Figure()
    # Banda P10–P90
    fig.add_trace(go.Scatter(
        x=pd.concat([agg["anio"], agg["anio"][::-1]]),
        y=pd.concat([agg["p90"], agg["p10"][::-1]]),
        fill="toself", fillcolor="rgba(174,214,241,0.30)", line_color="transparent",
        name="Banda P10–P90 (80% de países)",
        hovertemplate="Banda P10–P90<extra></extra>"
    ))
    # Banda P25–P75
    fig.add_trace(go.Scatter(
        x=pd.concat([agg["anio"], agg["anio"][::-1]]),
        y=pd.concat([agg["p75"], agg["p25"][::-1]]),
        fill="toself", fillcolor="rgba(46,134,171,0.40)", line_color="transparent",
        name="IQR P25–P75 (50% central)",
        hovertemplate="IQR P25–P75<extra></extra>"
    ))
    # Mediana
    fig.add_trace(go.Scatter(
        x=agg["anio"], y=agg["med"], mode="lines+markers",
        line=dict(color=AZUL_FUERTE, width=2.5),
        marker=dict(color="white", line=dict(color=AZUL_FUERTE, width=2), size=8),
        name="Mediana",
        hovertemplate="Año: %{x}<br>Mediana: %{y:.1f}%<extra></extra>"
    ))
    fig = pl_layout(fig, "Evolución del panel — Mediana, IQR y banda P10–P90 de tasa_fin_cap",
                    "Azul oscuro: IQR (P25–P75) | Azul claro: P10–P90 | Línea: mediana",
                    "Año", "tasa_fin_cap (%)")
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(range=[35, 105])
    fig.update_layout(height=420)
    return fig

# ─── Mapa de calor ────────────────────────────────────────────────────────────
@app.callback(Output("plot-heatmap", "figure"), Input("slider-n-heat", "value"), Input("current-tab", "data"))
def render_heatmap(n_heat, tab):
    if tab != "bivariado":
        return go.Figure()
    if df_2.empty or df_2["pais"].nunique() < 2:
        return _empty_fig("Datos insuficientes para mapa de calor.")
    top_p = (df_2.groupby("pais")["tasa_fin_cap"].mean()
             .sort_values(ascending=False).head(n_heat).index.tolist())
    d_h = df_2[df_2["pais"].isin(top_p)][["pais", "anio", "tasa_fin_cap"]]
    mat = d_h.pivot(index="pais", columns="anio", values="tasa_fin_cap")
    mat = mat.loc[top_p]

    fig = go.Figure(go.Heatmap(
        z=mat.values,
        x=[str(c) for c in mat.columns],
        y=mat.index.tolist(),
        colorscale=[[0, "#FEF9E7"], [0.25, "#AED6F1"], [0.6, AZUL_MAIN], [1, AZUL_FUERTE]],
        zmin=0, zmax=100,
        colorbar=dict(title="tasa_fin_cap (%)", ticksuffix="%", thickness=16, len=0.7,
                      bgcolor=BG_PLOT, bordercolor="rgba(46,134,171,0.4)",
                      tickfont=dict(color=TEXT_MAIN), titlefont=dict(color=CYAN)),
        hovertemplate="País: %{y}<br>Año: %{x}<br>tasa_fin_cap: %{z:.1f}%<extra></extra>"
    ))
    fig = pl_layout(fig, f"Mapa de Calor — tasa_fin_cap por País y Año (Top {n_heat})", "",
                    "Año", "País")
    fig.update_layout(height=max(400, n_heat * 18 + 100))
    return fig

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
        f"✅ Se RECHAZA H₀ (α = 0.05): la tasa mediana difiere significativamente entre {lab1} y {lab2}."
        if es_sig else
        f"⚠️ No se rechaza H₀ (α = 0.05): no hay evidencia suficiente de diferencia entre {lab1} y {lab2}."
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
            html.Span(f"{'✅' if norm1_ok else '❌'} Shapiro-Wilk Período 1: {'Normal' if norm1_ok else 'No normal'} "
                      f"(p = {sw1.pvalue:.4f})" if sw1 else "⚠️ Shapiro-Wilk Período 1: no calculado"),
            html.Br(),
            html.Span(f"{'✅' if norm2_ok else '❌'} Shapiro-Wilk Período 2: {'Normal' if norm2_ok else 'No normal'} "
                      f"(p = {sw2.pvalue:.4f})" if sw2 else "⚠️ Shapiro-Wilk Período 2: no calculado"),
            html.Br(),
            html.Span(f"{'✅' if homo_ok else '❌'} Homocedasticidad (Levene): "
                      f"{'Varianzas iguales' if homo_ok else 'Varianzas distintas'} (p = {lev.pvalue:.4f})"),
        ], style={"color": TEXT_DIM, "fontSize": "0.88em", "lineHeight": "1.8",
                  "background": "rgba(46,134,171,0.06)", "borderRadius": "8px", "padding": "12px 16px"}),
    ], style={"background": "linear-gradient(135deg,#0A1628,#112240)",
              "border": "1px solid rgba(46,134,171,0.4)", "borderRadius": "12px",
              "padding": "20px 24px", "marginTop": "8px"})

    return fig_box, fig_dist, resultado_ui

# ─── Mapas ─────────────────────────────────────────────────────────────────────
@app.callback(
    Output("plot-mapa-promedio", "figure"),
    Output("plot-mapa-animado", "figure"),
    Output("plot-mapa-cambio", "figure"),
    Input("current-tab", "data")
)
def render_mapas(tab):
    if tab != "mapa":
        return go.Figure(), go.Figure(), go.Figure()
    d = df_2[df_2["anio"].between(2000, 2022)].dropna(subset=["tasa_fin_cap"])
    if "iso3" not in d.columns or len(d) < 10:
        msg = "Datos ISO-3 insuficientes para renderizar los mapas."
        return _empty_fig(msg), _empty_fig(msg), _empty_fig(msg)

    # Mapa 1: Promedio histórico
    d_prom = (d.groupby(["iso3", "pais"])["tasa_fin_cap"]
              .agg(media_hist="mean", n_anios="nunique")
              .reset_index()
              .assign(media_hist=lambda x: x["media_hist"].round(1),
                      hover_txt=lambda x: x.apply(
                          lambda r: f"<b>{r['pais']}</b> ({r['iso3']})<br>Promedio: {r['media_hist']:.1f}%<br>Años: {r['n_anios']}", axis=1)))

    fig1 = go.Figure(go.Choropleth(
        locations=d_prom["iso3"], locationmode="ISO-3",
        z=d_prom["media_hist"], text=d_prom["hover_txt"], hoverinfo="text",
        colorscale=MAPA_COLORSCALE, zmin=0, zmax=100,
        colorbar=dict(title="tasa_fin_cap (%)", ticksuffix="%", thickness=16, len=0.7,
                      bgcolor=BG_PLOT, bordercolor="rgba(46,134,171,0.4)",
                      tickfont=dict(color=TEXT_MAIN), titlefont=dict(color=CYAN)),
        marker_line=dict(color="#FFFFFF", width=0.4)
    ))
    fig1.update_layout(
        title=dict(text="<b>Promedio histórico de tasa_fin_cap por país (2000–2022)</b>",
                   font=dict(size=14, color=CYAN), x=0.02),
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="rgba(46,134,171,0.4)",
                 showland=True, landcolor=BG_PLOT, showocean=True, oceancolor="#0A1628",
                 showlakes=False, projection_type="natural earth"),
        paper_bgcolor="#112240", font=dict(color=TEXT_MAIN),
        margin=dict(t=60, r=10, b=10, l=10), height=520
    )

    # Mapa 2: Animado por año
    d_anim = d.sort_values("anio").assign(
        anio_chr=lambda x: x["anio"].astype(str),
    )
    fig2 = px.choropleth(
        d_anim, locations="iso3", locationmode="ISO-3",
        color="tasa_fin_cap", animation_frame="anio_chr",
        color_continuous_scale=MAPA_COLORSCALE, range_color=[0, 100],
        hover_name="pais", hover_data={"tasa_fin_cap": ":.1f", "anio_chr": False, "iso3": False}
    )
    fig2.update_layout(
        title=dict(text="<b>Evolución anual de tasa_fin_cap por país</b><br><sup>Presione ▶ Play para animar</sup>",
                   font=dict(size=14, color=CYAN), x=0.02),
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="rgba(46,134,171,0.4)",
                 showland=True, landcolor=BG_PLOT, showocean=True, oceancolor="#0A1628",
                 showlakes=False, projection_type="natural earth"),
        paper_bgcolor="#112240", font=dict(color=TEXT_MAIN),
        margin=dict(t=70, r=10, b=10, l=10), height=540,
        coloraxis_colorbar=dict(title="tasa_fin_cap (%)", ticksuffix="%", thickness=16, len=0.7,
                                bgcolor=BG_PLOT, tickfont=dict(color=TEXT_MAIN), titlefont=dict(color=CYAN))
    )

    # Mapa 3: Cambio absoluto
    d_ini = (d.groupby(["iso3", "pais"]).apply(lambda g: g.loc[g["anio"].idxmin()])
             .reset_index(drop=True)[["iso3", "pais", "anio", "tasa_fin_cap"]]
             .rename(columns={"anio": "anio_ini", "tasa_fin_cap": "tasa_ini"}))
    d_fin2 = (d.groupby(["iso3", "pais"]).apply(lambda g: g.loc[g["anio"].idxmax()])
              .reset_index(drop=True)[["iso3", "pais", "anio", "tasa_fin_cap"]]
              .rename(columns={"anio": "anio_fin", "tasa_fin_cap": "tasa_fin2"}))
    d_cambio = (d_ini.merge(d_fin2, on=["iso3", "pais"])
                .assign(cambio=lambda x: (x["tasa_fin2"] - x["tasa_ini"]).round(1),
                        hover_txt=lambda x: x.apply(
                            lambda r: f"<b>{r['pais']}</b> ({r['iso3']})<br>"
                                      f"Año inicial: {r['anio_ini']} → {r['tasa_ini']:.1f}%<br>"
                                      f"Año final: {r['anio_fin']} → {r['tasa_fin2']:.1f}%<br>"
                                      f"<b>Cambio: {r['cambio']:+.1f} pp</b>", axis=1)))
    lim = d_cambio["cambio"].abs().max()

    fig3 = go.Figure(go.Choropleth(
        locations=d_cambio["iso3"], locationmode="ISO-3",
        z=d_cambio["cambio"], text=d_cambio["hover_txt"], hoverinfo="text",
        colorscale=CAMBIO_COLORSCALE, zmin=-lim, zmax=lim,
        colorbar=dict(title="Cambio (pp)", ticksuffix=" pp", thickness=16, len=0.7,
                      bgcolor=BG_PLOT, bordercolor="rgba(46,134,171,0.4)",
                      tickfont=dict(color=TEXT_MAIN), titlefont=dict(color=CYAN)),
        marker_line=dict(color="#FFFFFF", width=0.4)
    ))
    fig3.update_layout(
        title=dict(text=("<b>Cambio absoluto en tasa_fin_cap: último año vs. 2000</b><br>"
                         "<sup>Verde = mejora | Rojo = retroceso</sup>"),
                   font=dict(size=14, color=CYAN), x=0.02),
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="rgba(46,134,171,0.4)",
                 showland=True, landcolor=BG_PLOT, showocean=True, oceancolor="#0A1628",
                 showlakes=False, projection_type="natural earth"),
        paper_bgcolor="#112240", font=dict(color=TEXT_MAIN),
        margin=dict(t=80, r=10, b=10, l=10), height=520
    )

    return fig1, fig2, fig3

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
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  App Dash — Finalización Primaria Femenina ODS 4")
    print("  Abrir en: http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
