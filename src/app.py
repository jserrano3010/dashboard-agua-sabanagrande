"""
Dashboard: ¿Cuánta agua va a tener la finca?  · Entregable 1 (un solo archivo)

Edwin Yunis y Jairo Serrano · Maestría en Ingeniería Industrial
Machine Learning · Dr. Lihki Rubio

Tres ventanas:
  1. Contexto: problema, cifras clave y base de datos.
  2. Exploración de datos: solo las variables del Excel (NASA POWER, 1981–2020).
  3. Modelos: regresión logística y KNN para pronosticar si la semana h será seca,
     sin fuga de información (predictoras solo del pasado, partición cronológica con
     purga, validación cruzada temporal con separación, preprocesamiento dentro del
     Pipeline) y con control del sobreajuste (C y k elegidos por validación).

Cómo ejecutarlo en Visual Studio Code:
    1. Instalar dependencias en la terminal de VS Code:
         pip install "dash==2.18.2" "plotly>=5.24,<6" pandas numpy scikit-learn scipy openpyxl
    2. Guardar este archivo en la carpeta del proyecto (la que tiene la carpeta data/).
    3. Ejecutarlo (botón ▷ "Run Python File" o:  python dashboard_agua_finca.py)
    4. Abrir http://127.0.0.1:8050 en el navegador.

Datos: usa data/processed/dataset_sabanagrande.csv.gz; si no existe, lee la hoja
"Datos NASA" de Entregable1_Sabanagrande.xlsx (en data/raw/ o junto a este archivo).
"""
from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, dcc, html

warnings.filterwarnings("ignore")

# ==========================================================================
# CONFIGURACIÓN (la misma de src/config.py del proyecto)
# ==========================================================================
SEED = 42
_AQUI = Path(__file__).resolve().parent
ARCHIVO_DATOS = "dataset_sabanagrande.csv.gz"
ARCHIVO_EXCEL = "Entregable1_Sabanagrande.xlsx"
PUNTO_OBJETIVO = "FINCA_SABANAGRANDE"
LAT_FINCA, LON_FINCA = 10.79, -74.76
INICIO_TEST = pd.Timestamp("2021-01-01")       # prueba cronológica: 2021-2025
HORIZONTES = [1, 2, 4, 8, 13, 26, 52]          # semanas
DIAS = 7
LATENCIA = 3        # días de retraso de publicación de NASA POWER
N_SPLITS = 5
ARMONICOS = 3
CONFIANZA = 0.95
OBJ = "PRECTOTCORR"

# Solo las variables del Excel (sin ET0 ni construcciones derivadas)
VARIABLES = {
    "PRECTOTCORR": ("Lluvia", "mm/día"),
    "T2M": ("Temperatura media", "°C"),
    "T2M_MAX": ("Temperatura máxima", "°C"),
    "T2M_MIN": ("Temperatura mínima", "°C"),
    "RH2M": ("Humedad relativa", "%"),
    "WS2M": ("Viento a 2 m", "m/s"),
    "ALLSKY_SFC_SW_DWN": ("Radiación solar", "MJ/m²/día"),
    "PS": ("Presión en superficie", "kPa"),
    "GWETROOT": ("Humedad del suelo (raíces)", "0-1"),
    "GWETTOP": ("Humedad del suelo (superficie)", "0-1"),
}

UMBRAL_DEFECTO = 10.0    # mm: semana seca si llueve menos


def es(x, d=0):
    """Número con formato colombiano: 1.234,5"""
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


# ==========================================================================
# DATOS
# ==========================================================================
def _buscar(*partes):
    for base in (_AQUI, _AQUI.parent, Path.cwd(), Path.cwd().parent):
        ruta = base.joinpath(*partes)
        if ruta.exists():
            return ruta
    return None


@lru_cache(maxsize=1)
def cargar_datos():
    ruta = _buscar("data", "processed", ARCHIVO_DATOS)
    if ruta is not None:
        df = pd.read_csv(ruta, parse_dates=["fecha"])
    else:
        ruta = _buscar("data", "raw", ARCHIVO_EXCEL) or _buscar(ARCHIVO_EXCEL)
        if ruta is None:
            raise SystemExit(
                f"No se encontró {ARCHIVO_DATOS} ni {ARCHIVO_EXCEL}.\n"
                "Guarde este archivo en la carpeta del proyecto (la que contiene data/) "
                "o ponga el Excel del entregable junto a este archivo.")
        df = pd.read_excel(ruta, sheet_name="Datos NASA")
        df["fecha"] = pd.to_datetime(df["fecha"])
    cols = [v for v in VARIABLES if v in df]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").replace(-999, np.nan)
    puntos = df["punto"].unique()
    df = df[df["punto"] == (PUNTO_OBJETIVO if PUNTO_OBJETIVO in puntos else puntos[0])].copy()
    df["punto"] = PUNTO_OBJETIVO
    for col, valor in (("lat", LAT_FINCA), ("lon", LON_FINCA), ("elevacion_m", np.nan)):
        if col not in df:
            df[col] = valor
    df = df[["fecha", "punto", "lat", "lon", "elevacion_m"] + cols]
    return df.sort_values("fecha").reset_index(drop=True), ruta.name


def semanal(d, variables=None):
    """Semanas lunes-domingo completas: suma para la lluvia, media para el resto."""
    variables = variables or [v for v in VARIABLES if v in d]
    reglas = {v: ("sum" if v == OBJ else "mean") for v in variables}
    g = d.set_index("fecha").sort_index()
    w = g[variables].resample("W").agg(reglas)
    return w[g[OBJ].resample("W").size() == DIAS]


def acf(x, rezagos):
    x = np.asarray(x, float)
    x = x - x.mean()
    den = x @ x
    return np.array([(x[:-k] @ x[k:]) / den for k in range(1, rezagos + 1)])


# ==========================================================================
# MODELO DE REGRESIÓN (mismas funciones que src/modelado.py)
# ==========================================================================
def objetivo_semana_h(P, h):
    """y_h(t): lluvia acumulada en (t + 7(h-1), t + 7h]."""
    futuro = P[::-1].rolling(DIAS, min_periods=DIAS).sum()[::-1]
    return futuro.shift(-(DIAS * (h - 1) + 1))


def ingenuo_estacional(P, h):
    """Misma ventana objetivo, un año antes."""
    return objetivo_semana_h(P, h).shift(365)


def construir_predictoras(d):
    X = pd.DataFrame(index=d.index)
    P = d[OBJ]
    for v in [7, 30, 90, 180, 365]:
        X[f"P_suma_{v}d"] = P.rolling(v, min_periods=int(0.8 * v)).sum()
    X["dias_secos_consecutivos"] = P.lt(1).groupby(P.ge(1).cumsum()).cumsum()
    for var in ["T2M", "T2M_MAX", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "GWETROOT", "GWETTOP"]:
        if var in d:
            X[f"{var}_media_7d"] = d[var].rolling(7, min_periods=5).mean()
            X[f"{var}_media_30d"] = d[var].rolling(30, min_periods=24).mean()
    return X.shift(LATENCIA)      # en t solo se conoce lo observado hasta t - latencia


def componentes_ciclicos(fechas, h):
    """Armónicos del día del año del centro de la semana objetivo (calendario)."""
    centro = pd.DatetimeIndex(fechas) + pd.to_timedelta(DIAS * h - DIAS / 2, unit="D")
    ang = 2 * np.pi * centro.dayofyear / 365.25
    out = {"sin_doy": np.sin(ang), "cos_doy": np.cos(ang)}
    for k in range(2, ARMONICOS + 1):
        out[f"sin{k}_doy"] = np.sin(k * ang)
        out[f"cos{k}_doy"] = np.cos(k * ang)
    return pd.DataFrame(out, index=pd.DatetimeIndex(fechas))


@lru_cache(maxsize=None)
def tabla_modelado(h):
    """Una fila por lunes de emisión con predictoras, objetivo y línea base ingenua."""
    g = DF.set_index("fecha").sort_index()
    X = construir_predictoras(g)
    X = X.join(componentes_ciclicos(X.index, h))
    X["lat"], X["lon"], X["elevacion_m"] = g["lat"], g["lon"], g["elevacion_m"]
    X["y"] = objetivo_semana_h(g[OBJ], h)
    X["base_ingenuo_estacional"] = ingenuo_estacional(g[OBJ], h)
    X["punto"] = PUNTO_OBJETIVO
    X = X[X.index.dayofweek == 0]
    return X.reset_index().dropna(subset=["y"]).reset_index(drop=True)


def particion_cronologica(tabla, h):
    """Train: la semana objetivo termina antes de 2021 (purga). Test: emitidas desde 2021."""
    fin_objetivo = tabla["fecha"] + pd.to_timedelta(DIAS * h, unit="D")
    return tabla[fin_objetivo < INICIO_TEST].copy(), tabla[tabla["fecha"] >= INICIO_TEST].copy()


def cv_temporal(fechas, n_splits=N_SPLITS, gap_dias=0):
    """Ventana creciente con gap_dias de separación entre entrenamiento y validación."""
    fechas = pd.to_datetime(pd.Series(fechas)).reset_index(drop=True)
    bloques = np.array_split(np.sort(fechas.unique()), n_splits + 1)
    for k in range(1, n_splits + 1):
        tr = np.where(fechas <= bloques[k - 1][-1] - np.timedelta64(gap_dias, "D"))[0]
        va = np.where(fechas.isin(bloques[k]))[0]
        yield tr, va


def columnas_predictoras(tabla):
    excluir = {"fecha", "punto", "y", "seca", "base_ingenuo_estacional"}   # nunca el objetivo ni su etiqueta
    return [c for c in tabla.columns if c not in excluir and tabla[c].nunique(dropna=True) > 1]


# ==========================================================================
# FIGURAS
# ==========================================================================
TINTA = "#17332E"
RIO = "#1F6F78"
LLUVIA = "#3A7CA5"
SOL = "#D69E2E"
TIERRA = "#A5673F"
SALVIA = "#7F9C80"
GRIS = "#8FA59A"
REJILLA = "#DCE5E0"
FONDO_GRAF = "#F6F9F7"

ESCALA_LLUVIA = [[0, "#F3E6CC"], [0.12, "#E3CE98"], [0.35, "#9CC2B4"], [0.7, "#3F8C95"], [1, "#16414A"]]
ESCALA_DIVERGENTE = [[0, TIERRA], [0.5, "#F6F4EE"], [1, RIO]]

pio.templates["finca"] = go.layout.Template(layout=dict(
    font=dict(family="Figtree, 'Segoe UI', system-ui, sans-serif", size=13, color=TINTA),
    title=dict(font=dict(family="'Bricolage Grotesque', Figtree, sans-serif", size=17, color=TINTA), x=0.01, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=FONDO_GRAF,
    colorway=[RIO, SOL, LLUVIA, TIERRA, SALVIA, "#5B4630"],
    xaxis=dict(gridcolor=REJILLA, zerolinecolor=REJILLA, linecolor="#B9C9C1", ticks="outside", tickcolor="#B9C9C1"),
    yaxis=dict(gridcolor=REJILLA, zerolinecolor=REJILLA, linecolor="#B9C9C1"),
    legend=dict(bgcolor="rgba(255,255,255,0.6)", bordercolor=REJILLA, borderwidth=1, font=dict(size=12)),
    hoverlabel=dict(bgcolor="white", bordercolor=RIO, font=dict(family="Figtree, sans-serif", color=TINTA)),
    margin=dict(l=56, r=24, t=56, b=48),
    colorscale=dict(sequential=ESCALA_LLUVIA),
    separators=",.",
))
pio.templates.default = "plotly_white+finca"

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
LEYENDA_ARRIBA = dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")


def _etiqueta(var):
    n, u = VARIABLES.get(var, (var, ""))
    return f"{n} ({u})"


def vacia(mensaje):
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font=dict(size=14, color=SALVIA), x=.5, y=.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=360)
    return fig


# --------------------------------------------------------------------------
# Contexto
# --------------------------------------------------------------------------
def fig_anio_tipico(d):
    w = semanal(d, [OBJ])[OBJ]
    woy = w.index.isocalendar().week.clip(upper=52).values
    q = w.groupby(woy).quantile([0.1, 0.5, 0.9]).unstack()
    media = w.groupby(woy).mean()
    x = pd.date_range("2001-01-01", periods=len(q), freq="7D")
    fig = go.Figure()
    for x0, x1 in (("2001-01-01", "2001-04-01"), ("2001-12-01", "2001-12-31")):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=SOL, opacity=.10, line_width=0)
    fig.add_trace(go.Scatter(x=x, y=q[0.9], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q[0.1], fill="tonexty", fillcolor="rgba(58,124,165,0.18)", line=dict(width=0),
                             name="80 % de los años (P10–P90)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q[0.5], name="Mediana", line=dict(color=LLUVIA, width=3),
                             hovertemplate="semana del %{x|%d %b}: %{y:.1f} mm<extra>mediana</extra>"))
    fig.add_trace(go.Scatter(x=x, y=media.values, name="Media", line=dict(color=TINTA, width=1.5, dash="dot"),
                             hovertemplate="semana del %{x|%d %b}: %{y:.1f} mm<extra>media</extra>"))
    fig.add_annotation(x="2001-02-15", y=float(q[0.9].max()) * .92, text="temporada seca", showarrow=False,
                       font=dict(color=TIERRA, size=12))
    fig.update_layout(title="Lluvia semanal en un año típico (1981–2020): dos temporadas de lluvia",
                      yaxis_title="mm por semana", height=380, legend=LEYENDA_ARRIBA)
    fig.update_xaxes(tickvals=pd.date_range("2001-01-01", periods=12, freq="MS"), ticktext=MESES)
    return fig


# --------------------------------------------------------------------------
# Exploración de datos (solo variables del Excel)
# --------------------------------------------------------------------------
def fig_climatologia(d):
    g = d.set_index("fecha")
    mensual = g[OBJ].resample("MS").sum()
    m = mensual.groupby(mensual.index.month).mean().reindex(range(1, 13))
    dias = (g[OBJ] >= 1).groupby(g.index.month).mean().reindex(range(1, 13))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=MESES, y=m.values, name="Lluvia media mensual", marker_color=LLUVIA,
                         hovertemplate="%{x}: %{y:.0f} mm<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=MESES, y=dias.values, name="Proporción de días con lluvia ≥ 1 mm", mode="lines+markers",
                             line=dict(color=SOL, width=3), marker=dict(size=8),
                             hovertemplate="%{x}: %{y:.0%}<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title_text="mm por mes", secondary_y=False)
    fig.update_yaxes(title_text="días con lluvia", tickformat=".0%", range=[0, 1], showgrid=False, secondary_y=True)
    fig.update_layout(title="Climatología mensual de la lluvia", height=400, legend=LEYENDA_ARRIBA)
    return fig


def fig_distribucion(d):
    y = semanal(d, [OBJ])[OBJ]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.1,
                        subplot_titles=(f"Lluvia semanal · asimetría {es(y.skew(), 2)}",
                                        f"log(1 + lluvia) · asimetría {es(np.log1p(y).skew(), 2)}"))
    fig.add_trace(go.Histogram(x=y, nbinsx=60, marker_color=LLUVIA, opacity=.85, name="mm",
                               hovertemplate="%{x} mm: %{y} semanas<extra></extra>"), 1, 1)
    fig.add_trace(go.Histogram(x=np.log1p(y), nbinsx=40, marker_color=RIO, opacity=.85, name="log1p",
                               hovertemplate="%{x}: %{y} semanas<extra></extra>"), 1, 2)
    fig.add_vline(x=y.mean(), line_dash="dot", line_color=TINTA, row=1, col=1,
                  annotation_text=f"media {es(y.mean(), 1)}", annotation_position="top right")
    fig.add_vline(x=y.median(), line_dash="dash", line_color=SOL, row=1, col=1,
                  annotation_text=f"mediana {es(y.median(), 1)}", annotation_position="bottom right")
    fig.update_xaxes(title_text="mm por semana", row=1, col=1)
    fig.update_xaxes(title_text="log(1 + mm)", row=1, col=2)
    fig.update_yaxes(title_text="semanas", row=1, col=1)
    fig.update_layout(title="Distribución de la variable objetivo", height=400, showlegend=False, bargap=.05)
    return fig


def fig_cajas_mes(d, var):
    fig = go.Figure()
    for i, m in enumerate(MESES, start=1):
        fig.add_trace(go.Box(y=d.loc[d.fecha.dt.month == i, var], name=m, boxpoints=False, line_width=1.5,
                             marker_color=SOL if i in (12, 1, 2, 3) else RIO, showlegend=False))
    fig.update_layout(title="Estacionalidad por mes (ocre = temporada seca)", yaxis_title=_etiqueta(var), height=400)
    return fig


def fig_calendario(d):
    t = d.groupby([d.fecha.dt.year, d.fecha.dt.month])[OBJ].sum().unstack().reindex(columns=range(1, 13))
    fig = go.Figure(go.Heatmap(z=t.values, x=MESES, y=t.index, colorscale=ESCALA_LLUVIA,
                               colorbar=dict(title="mm", thickness=12), xgap=2, ygap=1,
                               hovertemplate="%{y} · %{x}: %{z:.0f} mm<extra></extra>"))
    fig.update_layout(title="Lluvia mensual, año por año", height=520, plot_bgcolor="white",
                      yaxis=dict(autorange="reversed", dtick=5, gridcolor="rgba(0,0,0,0)"),
                      xaxis=dict(side="top", gridcolor="rgba(0,0,0,0)"))
    return fig


def fig_correlaciones(d):
    cols = [c for c in VARIABLES if c in d]
    c = semanal(d, cols).corr(method="spearman")
    nombres = [VARIABLES[v][0] for v in cols]
    fig = go.Figure(go.Heatmap(z=c.values, x=nombres, y=nombres, zmin=-1, zmax=1, colorscale=ESCALA_DIVERGENTE,
                               text=np.round(c.values, 2), texttemplate="%{text}", textfont=dict(size=10),
                               colorbar=dict(title="ρ", thickness=12),
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Correlación de Spearman entre variables (semanal)", height=560, plot_bgcolor="white",
                      xaxis=dict(tickangle=-35, gridcolor="rgba(0,0,0,0)"),
                      yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"), margin=dict(l=190, b=150))
    return fig


def fig_relacion_objetivo(anios):
    t = tabla_modelado(1)
    fin_obj = t.fecha + pd.Timedelta(days=DIAS)
    t = t[(fin_obj < INICIO_TEST) & t.fecha.dt.year.between(*anios)]
    cols = columnas_predictoras(tabla_modelado(1))
    rho = t[cols].corrwith(t["y"], method="spearman").dropna()
    rho = rho.reindex(rho.abs().sort_values(ascending=False).index).head(15)[::-1]
    fig = go.Figure(go.Bar(x=rho.values, y=rho.index, orientation="h",
                           marker_color=[RIO if v > 0 else TIERRA for v in rho.values],
                           hovertemplate="%{y}: ρ = %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=TINTA, line_width=1)
    fig.update_layout(title="Predictoras (solo pasado) vs lluvia de la semana siguiente", height=480,
                      xaxis_title="ρ de Spearman", xaxis_range=[-1, 1], margin=dict(l=230))
    return fig


def fig_acf(d, rezagos=110):
    w = semanal(d, [OBJ])[OBJ]
    r = np.r_[1.0, acf(w.values, rezagos)]
    banda = 1.96 / np.sqrt(len(w))
    fig = go.Figure()
    fig.add_hrect(y0=-banda, y1=banda, fillcolor=SALVIA, opacity=.18, line_width=0)
    fig.add_trace(go.Bar(x=np.arange(len(r)), y=r, marker_color=np.where(np.abs(r) > banda, RIO, "#AFC3BA"),
                         hovertemplate="rezago %{x} semanas: %{y:.2f}<extra></extra>"))
    for sem in (52, 104):
        fig.add_vline(x=sem, line_dash="dot", line_color=SOL, annotation_text=f"{sem // 52} año", annotation_font_color=SOL)
    fig.update_layout(title="Autocorrelación de la lluvia semanal", xaxis_title="rezago (semanas)", yaxis_title="ACF",
                      height=400, showlegend=False)
    return fig


def tendencias(d):
    g = d.set_index("fecha")
    a = g.resample("YE").agg({OBJ: "sum", "T2M": "mean"})
    a = a[g[OBJ].resample("YE").size().values >= 365]
    a.index = a.index.year
    if len(a) < 3:
        return a, None, None
    return a, stats.linregress(a.index, a[OBJ]), stats.linregress(a.index, a["T2M"])


def fig_tendencia(d):
    a, rl, rt = tendencias(d)
    if rl is None:
        return vacia("Se necesitan al menos 3 años completos.")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=a.index, y=a[OBJ], name="Lluvia anual", marker_color="#9CC2B4",
                         hovertemplate="%{x}: %{y:.0f} mm<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=a.index, y=rl.intercept + rl.slope * a.index, name="Tendencia lineal",
                             line=dict(color=RIO, width=3, dash="dash")), secondary_y=False)
    fig.add_trace(go.Scatter(x=a.index, y=a["T2M"], name="Temperatura media", mode="lines+markers",
                             line=dict(color=TIERRA, width=2), marker=dict(size=5)), secondary_y=True)
    fig.update_yaxes(title_text="mm/año", secondary_y=False)
    fig.update_yaxes(title_text="°C", secondary_y=True, showgrid=False)
    fig.update_layout(title=f"Lluvia {es(rl.slope * 10, 0)} mm/década (p = {es(rl.pvalue, 3)}) · "
                            f"temperatura {es(rt.slope * 10, 2)} °C/década (p = {es(rt.pvalue, 3)})",
                      height=400, legend=LEYENDA_ARRIBA)
    fig.update_layout(title_font_size=15)
    return fig


def fig_faltantes(d):
    cols = [c for c in VARIABLES if c in d]
    t = d[cols].isna().groupby(d.fecha.dt.year).mean() * 100
    fig = go.Figure(go.Heatmap(z=t.T.values, x=t.index, y=[VARIABLES[c][0] for c in cols], zmin=0, zmax=100,
                               colorscale=[[0, "#F6F9F7"], [0.01, "#E3CE98"], [1, TIERRA]],
                               colorbar=dict(title="%", thickness=12), xgap=1, ygap=2,
                               hovertemplate="%{y}, %{x}: %{z:.1f} % faltante<extra></extra>"))
    total = d[cols].isna().mean() * 100
    peor = total.idxmax()
    fig.update_layout(title=f"Datos faltantes por año · máximo: {VARIABLES[peor][0].lower()} ({es(total[peor], 2)} %)",
                      height=400, plot_bgcolor="white", margin=dict(l=210),
                      xaxis=dict(gridcolor="rgba(0,0,0,0)"), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig




# ==========================================================================
# MODELOS DE CLASIFICACIÓN: REGRESIÓN LOGÍSTICA Y KNN
# ==========================================================================
# Objetivo binario: semana seca = lluvia de la semana h < umbral (mm).
# Mismas predictoras, partición y validación que el modelo de regresión del libro.
NOMBRES = {"logistica": "Regresión logística", "knn": "KNN", "climatologia": "Climatología (referencia)"}


def _pipeline(estimador):
    # imputación y escalado DENTRO del pipeline: se ajustan solo con los datos de entrenamiento de cada fold
    return Pipeline([("imputar", SimpleImputer(strategy="median")),
                     ("escalar", StandardScaler()),
                     ("modelo", estimador)])


MODELOS = {
    "logistica": (
        _pipeline(LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)),
        {"modelo__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
        "modelo__C",
    ),
    "knn": (
        _pipeline(KNeighborsClassifier(weights="uniform")),
        {"modelo__n_neighbors": [3, 11, 25, 51, 101, 151, 251]},
        "modelo__n_neighbors",
    ),
}


class Climatologia:
    """Probabilidad de semana seca = frecuencia histórica (solo entrenamiento) de esa semana del año."""
    def __init__(self, h):
        self.h = h

    def _semana(self, t):
        return (t["fecha"] + pd.to_timedelta(DIAS * (self.h - 1) + 1, unit="D")).dt.isocalendar().week.clip(upper=52)

    def fit(self, t):
        self.tabla = t.groupby(self._semana(t).values)["seca"].mean()
        self.global_ = t["seca"].mean()
        return self

    def predict_proba(self, t):
        return self._semana(t).map(self.tabla).fillna(self.global_).astype(float).values


def _bootstrap_auc(y, p, n_boot=500, bloque=12):
    """IC 95 % del AUC con bootstrap por bloques (respeta la autocorrelación semanal)."""
    rng = np.random.default_rng(SEED)
    n, vals = len(y), []
    for _ in range(n_boot):
        ini = rng.integers(0, max(1, n - bloque), int(np.ceil(n / bloque)))
        idx = np.concatenate([np.arange(s, s + bloque) for s in ini])[:n]
        if y[idx].min() != y[idx].max():
            vals.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


def _metricas(y, p):
    lo, hi = _bootstrap_auc(y, p)
    return {"AUC": roc_auc_score(y, p), "AUC IC95 inf": lo, "AUC IC95 sup": hi,
            "AP": average_precision_score(y, p), "Brier": brier_score_loss(y, p)}


@lru_cache(maxsize=32)
def entrenar(h=1, umbral=10.0):
    t = tabla_modelado(h).copy()
    t["seca"] = (t["y"] < umbral).astype(int)
    cols = columnas_predictoras(t)
    assert not {"y", "seca"} & set(cols), "fuga: el objetivo quedó entre las predictoras"
    train, test = particion_cronologica(t, h)
    purgadas = len(t) - len(train) - len(test)
    train = train.sort_values("fecha").reset_index(drop=True)
    test = test.sort_values("fecha").reset_index(drop=True)
    gap = DIAS * h + LATENCIA
    splits = list(cv_temporal(train["fecha"], N_SPLITS, gap))
    y_te = test["seca"].values

    res = {"h": h, "umbral": umbral, "cols": cols, "n_train": len(train), "n_test": len(test), "purgadas": purgadas,
           "gap": gap, "prop_seca_train": train["seca"].mean(), "prop_seca_test": test["seca"].mean(),
           "fechas_compartidas": len(set(train.fecha) & set(test.fecha)),
           "train_fin_objetivo": train.fecha.max() + pd.Timedelta(days=DIAS * h), "test_ini": test.fecha.min(),
           "modelos": {}}

    for clave, (pipe, grid, param) in MODELOS.items():
        gs = GridSearchCV(pipe, grid, cv=splits, scoring="roc_auc", return_train_score=True, n_jobs=1)
        gs.fit(train[cols], train["seca"])
        cvr = pd.DataFrame(gs.cv_results_)
        curva = pd.DataFrame({"valor": cvr[f"param_{param}"].astype(float), "train": cvr.mean_train_score,
                              "cv": cvr.mean_test_score, "cv_std": cvr.std_test_score}).sort_values("valor")
        p_tr = gs.best_estimator_.predict_proba(train[cols])[:, 1]
        p_te = gs.best_estimator_.predict_proba(test[cols])[:, 1]
        info = {"mejor": gs.best_params_[param], "param": param, "curva": curva,
                "auc_train": roc_auc_score(train["seca"], p_tr), "auc_cv": gs.best_score_,
                "p_test": p_te, "metricas": _metricas(y_te, p_te)}
        if clave == "logistica":
            coef = gs.best_estimator_.named_steps["modelo"].coef_[0]
            info["coeficientes"] = pd.Series(coef, index=cols).sort_values(key=np.abs, ascending=False)
        res["modelos"][clave] = info

    clim = Climatologia(h).fit(train)
    p_cl = clim.predict_proba(test)
    res["modelos"]["climatologia"] = {"p_test": p_cl, "metricas": _metricas(y_te, p_cl),
                                      "auc_train": roc_auc_score(train["seca"], clim.predict_proba(train))}
    res["y_test"] = y_te
    res["test"] = test[["fecha", "y", "seca"]]
    return res


def curvas(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    frac, media = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {"roc": (fpr, tpr), "pr": (rec, prec), "cal": (media, frac)}


# --------------------------------------------------------------------------
# Figuras de los modelos
# --------------------------------------------------------------------------
COLOR_MODELO = {"logistica": RIO, "knn": SOL, "climatologia": GRIS}


def fig_roc_pr(res):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Curva ROC", "Curva precisión-recall"), horizontal_spacing=.12)
    y = res["y_test"]
    for clave, m in res["modelos"].items():
        c = curvas(y, m["p_test"])
        estilo = dict(color=COLOR_MODELO[clave], width=2 if clave == "climatologia" else 3,
                      dash="dash" if clave == "climatologia" else "solid")
        fig.add_trace(go.Scatter(x=c["roc"][0], y=c["roc"][1], name=f"{NOMBRES[clave]} (AUC {es(m['metricas']['AUC'], 3)})",
                                 line=estilo, legendgroup=clave), 1, 1)
        fig.add_trace(go.Scatter(x=c["pr"][0], y=c["pr"][1], name=NOMBRES[clave], line=estilo,
                                 legendgroup=clave, showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#B9C9C1", dash="dot"), showlegend=False,
                             hoverinfo="skip"), 1, 1)
    fig.add_hline(y=y.mean(), line_dash="dot", line_color="#B9C9C1", row=1, col=2)
    fig.update_xaxes(title_text="tasa de falsos positivos", row=1, col=1)
    fig.update_yaxes(title_text="tasa de verdaderos positivos", row=1, col=1)
    fig.update_xaxes(title_text="recall", row=1, col=2)
    fig.update_yaxes(title_text="precisión", row=1, col=2)
    fig.update_layout(title="Discriminación en el periodo de prueba 2021–2025", height=440,
                      legend=dict(orientation="h", y=-0.22, x=0))
    return fig


def fig_curva_validacion(res, clave):
    m = res["modelos"][clave]
    c = m["curva"]
    es_log = clave == "logistica"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c.valor, y=c.cv + c.cv_std, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=c.valor, y=c.cv - c.cv_std, fill="tonexty", fillcolor="rgba(31,111,120,0.15)",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=c.valor, y=c.train, name="Entrenamiento", mode="lines+markers", line=dict(color=SOL, width=3)))
    fig.add_trace(go.Scatter(x=c.valor, y=c.cv, name="Validación cruzada temporal", mode="lines+markers",
                             line=dict(color=RIO, width=3)))
    elegido = m["mejor"]
    fig.add_vline(x=np.log10(elegido) if es_log else elegido, line_dash="dot", line_color=TINTA,
                  annotation_text=f"elegido: {elegido:g}")
    fig.update_xaxes(type="log" if es_log else "linear",
                     title_text="C (inverso de la regularización)" if es_log else "k (número de vecinos)")
    fig.update_layout(title=f"Curva de validación ({NOMBRES[clave]}): dónde empieza el sobreajuste", yaxis_title="AUC",
                      height=420, legend=LEYENDA_ARRIBA, title_font_size=15)
    return fig


def fig_brecha(res):
    filas = []
    for clave in ("logistica", "knn"):
        m = res["modelos"][clave]
        for etapa, v in [("Entrenamiento", m["auc_train"]), ("Validación", m["auc_cv"]), ("Prueba", m["metricas"]["AUC"])]:
            filas.append({"modelo": NOMBRES[clave], "etapa": etapa, "AUC": v})
    d = pd.DataFrame(filas)
    fig = go.Figure()
    for etapa, color in (("Entrenamiento", SOL), ("Validación", RIO), ("Prueba", TINTA)):
        s = d[d.etapa == etapa]
        fig.add_trace(go.Bar(x=s.modelo, y=s.AUC, name=etapa, marker_color=color, text=[es(v, 3) for v in s.AUC],
                             textposition="outside"))
    fig.update_yaxes(range=[max(0.4, d.AUC.min() - .08), 1])
    fig.update_layout(title="Brecha entrenamiento → validación → prueba", height=420, barmode="group",
                      yaxis_title="AUC", legend=LEYENDA_ARRIBA)
    return fig


def fig_confusion(y, p, umbral_prob):
    cm = confusion_matrix(y, (p >= umbral_prob).astype(int), labels=[0, 1])
    etiquetas = ["Con lluvia", "Seca"]
    total = cm.sum()
    texto = [[f"{v}<br>{v / total:.0%}" for v in fila] for fila in cm]
    fig = go.Figure(go.Heatmap(z=cm, x=etiquetas, y=etiquetas, text=texto, texttemplate="%{text}",
                               textfont=dict(size=15), colorscale=[[0, "#F6F9F7"], [1, RIO]], showscale=False,
                               hovertemplate="real %{y}, predicha %{x}: %{z}<extra></extra>", xgap=3, ygap=3))
    fig.update_layout(title=f"Matriz de confusión (prueba, umbral {es(umbral_prob, 2)})", height=400, plot_bgcolor="white",
                      xaxis=dict(title="predicción", gridcolor="rgba(0,0,0,0)"),
                      yaxis=dict(title="real", autorange="reversed", gridcolor="rgba(0,0,0,0)"))
    return fig


def fig_calibracion(res):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#B9C9C1", dash="dot"), name="Calibración perfecta"))
    for clave, m in res["modelos"].items():
        media, frac = curvas(res["y_test"], m["p_test"])["cal"]
        fig.add_trace(go.Scatter(x=media, y=frac, mode="lines+markers", name=NOMBRES[clave],
                                 line=dict(color=COLOR_MODELO[clave], width=2.5,
                                           dash="dash" if clave == "climatologia" else "solid")))
    fig.update_layout(title="Calibración: ¿la probabilidad dice la verdad?", xaxis_title="probabilidad predicha",
                      yaxis_title="frecuencia observada de semana seca", height=400,
                      legend=dict(orientation="h", y=-0.3, x=0))
    return fig


def fig_coeficientes(res, n=12):
    c = res["modelos"]["logistica"]["coeficientes"].head(n)[::-1]
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h",
                           marker_color=[SOL if v > 0 else LLUVIA for v in c.values],
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=TINTA, line_width=1)
    fig.update_layout(title="Coeficientes estandarizados (ocre = más probable semana seca)", height=440,
                      xaxis_title="coeficiente (log-odds por desviación estándar)", margin=dict(l=230), title_font_size=15)
    return fig


def fig_linea_tiempo(res, clave):
    t = res["test"].assign(p=res["modelos"][clave]["p_test"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=t.fecha, y=t.y, name="Lluvia observada en la semana objetivo",
                         marker_color=np.where(t.seca == 1, "#E3CE98", LLUVIA)), secondary_y=False)
    fig.add_trace(go.Scatter(x=t.fecha, y=t.p, name="Probabilidad de semana seca", line=dict(color=TINTA, width=2)),
                  secondary_y=True)
    fig.update_yaxes(title_text="mm", secondary_y=False)
    fig.update_yaxes(title_text="probabilidad", range=[0, 1], secondary_y=True, showgrid=False)
    fig.update_layout(title=f"Periodo de prueba · {NOMBRES[clave]} (beige = semana seca real)", height=400, bargap=0.1,
                      legend=LEYENDA_ARRIBA)
    return fig


# ==========================================================================
# APP
# ==========================================================================
FUENTES = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700"
           "&family=Figtree:wght@400;500;600;700&display=swap")

print("Cargando datos y entrenando los modelos iniciales...")
DF, FUENTE = cargar_datos()
DF_TRAIN = DF[DF.fecha < INICIO_TEST]
ANIO_MIN, ANIO_MAX = int(DF.fecha.dt.year.min()), int(DF_TRAIN.fecha.dt.year.max())
OPC_VARIABLES = [{"label": f"{n} ({u})", "value": v} for v, (n, u) in VARIABLES.items() if v in DF]
CONFIG_GRAF = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

app = Dash(__name__, external_stylesheets=[FUENTES], title="Agua para la finca · Sabanagrande",
           suppress_callback_exceptions=True)
server = app.server  # <- For Render

# ==========================================================================
# ESTILOS E ILUSTRACIÓN (incrustados para que todo quede en un solo archivo)
# ==========================================================================
CSS = r"""/* Paleta: agua de jagüey, tinta de río, sol de temporada seca */
:root {
  --fondo: #E6EDE9;
  --panel: #FBFCFB;
  --tinta: #17332E;
  --tinta-suave: #4E6A63;
  --rio: #1F6F78;
  --lluvia: #3A7CA5;
  --sol: #D69E2E;
  --tierra: #A5673F;
  --borde: #CBD8D2;
  --radio-panel: 14px;
  --radio-control: 8px;
  --titulo: "Bricolage Grotesque", Figtree, "Segoe UI", system-ui, sans-serif;
  --texto: Figtree, "Segoe UI", system-ui, sans-serif;
}

* { box-sizing: border-box; }
body { margin: 0; background: var(--fondo); color: var(--tinta); font-family: var(--texto); font-size: 16px; line-height: 1.55; }

/* ---------- Encabezado con la ilustración ---------- */
.hero {
  position: relative;
  aspect-ratio: 1600 / 560;
  min-height: 340px;
  max-height: 540px;
  width: 100%;
  background: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAwIDU2MCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQgc2xpY2UiIHJvbGU9ImltZyIgYXJpYS1sYWJlbD0iRmluY2EgZW4gdGVtcG9yYWRhIHNlY2EganVudG8gYWwgcsOtbywgY29uIHVuIHJlc2Vydm9yaW8gZGUgYWd1YSB5IHVuYSBudWJlIGRlIGxsdXZpYSBhIGxvIGxlam9zIj4KPGRlZnM+CjxsaW5lYXJHcmFkaWVudCBpZD0iY2llbG8iIHgxPSIwIiB5MT0iMCIgeDI9IjAiIHkyPSIxIj4KIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzlGQzdDRiIvPjxzdG9wIG9mZnNldD0iLjU1IiBzdG9wLWNvbG9yPSIjRDhFNERDIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjRUZFM0M0Ii8+PC9saW5lYXJHcmFkaWVudD4KPHJhZGlhbEdyYWRpZW50IGlkPSJzb2wiIGN4PSIuNSIgY3k9Ii41IiByPSIuNSI+CiA8c3RvcCBvZmZzZXQ9IjAiIHN0b3AtY29sb3I9IiNGNkM0NTMiLz48c3RvcCBvZmZzZXQ9Ii41NSIgc3RvcC1jb2xvcj0iI0U4QTkzQSIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iI0U4QTkzQSIgc3RvcC1vcGFjaXR5PSIwIi8+PC9yYWRpYWxHcmFkaWVudD4KPGxpbmVhckdyYWRpZW50IGlkPSJ0aWVycmEiIHgxPSIwIiB5MT0iMCIgeDI9IjAiIHkyPSIxIj4KIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iI0Q3QjI3QSIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iI0I5OEE1MiIvPjwvbGluZWFyR3JhZGllbnQ+CjxsaW5lYXJHcmFkaWVudCBpZD0iYWd1YSIgeDE9IjAiIHkxPSIwIiB4Mj0iMCIgeTI9IjEiPgogPHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjM0Y4Qzk1Ii8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMUY1RTY2Ii8+PC9saW5lYXJHcmFkaWVudD4KPGxpbmVhckdyYWRpZW50IGlkPSJyaW8iIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIwIj4KIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzVEOUNBMyIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzdGQjJCNSIvPjwvbGluZWFyR3JhZGllbnQ+CjxjbGlwUGF0aCBpZD0iamFndWV5Ij48ZWxsaXBzZSBjeD0iMTE4MCIgY3k9IjQ1NSIgcng9IjIzMCIgcnk9IjU4Ii8+PC9jbGlwUGF0aD4KPC9kZWZzPgo8cmVjdCB3aWR0aD0iMTYwMCIgaGVpZ2h0PSI1NjAiIGZpbGw9InVybCgjY2llbG8pIi8+CjxjaXJjbGUgY3g9IjEzMzAiIGN5PSIxMjAiIHI9IjE1MCIgZmlsbD0idXJsKCNzb2wpIiBvcGFjaXR5PSIuNTUiLz4KPGNpcmNsZSBjeD0iMTMzMCIgY3k9IjEyMCIgcj0iNTgiIGZpbGw9IiNGMkI1NDQiLz4KPGcgb3BhY2l0eT0iLjg1IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2NDAgLTMwKSI+PHBhdGggZD0iTTE1MCAxNTAgcTEwLTQ4IDYyLTQ0IHEyMi00MCA3Mi0yMiBxNDQtMTYgNjYgMjQgcTQ0IDQgNDAgNDQgeiIgZmlsbD0iIzZGOEY5OSIvPgo8bGluZSB4MT0iMTc1IiB5MT0iMTYwIiB4Mj0iMTYxIiB5Mj0iMjQ1IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMTkxIiB5MT0iMTYwIiB4Mj0iMTc3IiB5Mj0iMjY1IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjA3IiB5MT0iMTYwIiB4Mj0iMTkzIiB5Mj0iMjM5IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjIzIiB5MT0iMTYwIiB4Mj0iMjA5IiB5Mj0iMjQ3IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjM5IiB5MT0iMTYwIiB4Mj0iMjI1IiB5Mj0iMjU1IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjU1IiB5MT0iMTYwIiB4Mj0iMjQxIiB5Mj0iMjM2IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjcxIiB5MT0iMTYwIiB4Mj0iMjU3IiB5Mj0iMjM3IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjg3IiB5MT0iMTYwIiB4Mj0iMjczIiB5Mj0iMjYxIiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzAzIiB5MT0iMTYwIiB4Mj0iMjg5IiB5Mj0iMjUyIiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzE5IiB5MT0iMTYwIiB4Mj0iMzA1IiB5Mj0iMjM4IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzM1IiB5MT0iMTYwIiB4Mj0iMzIxIiB5Mj0iMjQ2IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzUxIiB5MT0iMTYwIiB4Mj0iMzM3IiB5Mj0iMjUzIiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzY3IiB5MT0iMTYwIiB4Mj0iMzUzIiB5Mj0iMjM2IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzgzIiB5MT0iMTYwIiB4Mj0iMzY5IiB5Mj0iMjY0IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8L2c+CjxwYXRoIGQ9Ik0wIDI4NSBDIDE4MCAyNDAgMzMwIDI2MiA1MjAgMjUwIFMgODgwIDIyMCAxMDYwIDI0OCBTIDE0MDAgMjM2IDE2MDAgMjU4IEwxNjAwIDMzMCBMMCAzMzAgWiIgZmlsbD0iIzlEQjM5QSIgb3BhY2l0eT0iLjc1Ii8+CjxwYXRoIGQ9Ik0wIDMwNSBDIDIyMCAyODAgNDIwIDMwMCA2NDAgMjg4IFMgMTAyMCAyNzAgMTI0MCAyOTIgUyAxNDgwIDI4NCAxNjAwIDI5NiBMMTYwMCAzNDAgTDAgMzQwIFoiIGZpbGw9IiM3RjlDODAiIG9wYWNpdHk9Ii44Ii8+CjxwYXRoIGQ9Ik0wIDMzMCBDIDMwMCAzMTggNjIwIDM0NCA5MDAgMzMyIFMgMTQwMCAzMjIgMTYwMCAzMzQgTDE2MDAgMzU2IEMgMTM4MCAzNDggMTEwMCAzNjAgODYwIDM1NiBTIDI4MCAzNDYgMCAzNTYgWiIgZmlsbD0idXJsKCNyaW8pIi8+CjxwYXRoIGQ9Ik02MCAzNDAgaDEyMCBNNDIwIDM0NSBoOTAgTTc2MCAzNDIgaDE0MCBNMTE4MCAzMzYgaDExMCBNMTQ1MCAzNDAgaDgwIiBzdHJva2U9IiNFOEYyRjEiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNyIvPgo8cGF0aCBkPSJNMCAzNTIgQyA0MDAgMzYyIDkwMCAzNDggMTYwMCAzNTYgTDE2MDAgNTYwIEwwIDU2MCBaIiBmaWxsPSJ1cmwoI3RpZXJyYSkiLz4KPHBhdGggZD0iTTEwMzkgNDU0IEwxMDE2IDQ1NyBMMTAxNCA0NDkiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0xODUgNTQxIEwxNjAgNTQ5IEwxMzkgNTQ2IEwxNTEgNTU2IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTEyNiA1NDcgTDEyMyA1MzggTDEwOSA1MjkgTDExNiA1MjMgTDEwNiA1MjYiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0xMTA3IDQzMCBMMTA5OCA0MzcgTDExMjIgNDQ4IEwxMTA1IDQ0MSBMMTExNCA0NDkiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNMzg0IDQ5NSBMMzkxIDUwNyBMMzY3IDUxNSIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTEyNjcgNDUyIEwxMjgyIDQ1OSBMMTI4MSA0NzMgTDEyNzMgNDc3IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTkyOCA0OTIgTDkxNSA0ODcgTDkzMSA1MDEgTDkxOCA0OTMiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNNjE0IDUzNCBMNjQyIDUzNCBMNjYwIDUzOCBMNjUwIDU0NyIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTI0MSA1MzEgTDIyMyA1NDUgTDIxNiA1MzkgTDIxOSA1NDIiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0xMzY4IDQxOSBMMTM3NiA0MTkgTDEzNjkgNDMxIEwxMzYzIDQ0MCBMMTM2NiA0NDgiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjYiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0xNDAgNDIzIEwxNDIgNDM1IEwxNTYgNDI3IEwxMzEgNDQwIiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTYzNCA1NDcgTDY1OCA1NTEgTDY0OCA1NjMgTDY0NCA1NzQgTDYzOCA1NjQiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjYiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik03MjcgNDQzIEw3MDYgNDQ4IEw2ODEgNDQ0IEw3MDIgNDQzIEw2ODIgNDU2IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNODE0IDUwMCBMNzkxIDQ5NSBMNzkxIDQ5NyBMNzk4IDQ5NSIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTg4MSA1NDAgTDg5OCA1NDMgTDg5MiA1NTQgTDkyMCA1NTYiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0zMDkgNDIxIEwyOTAgNDE4IEwzMDQgNDE1IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNOTkzIDU1MCBMOTgxIDU0OSBMOTUzIDU0MyIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuNiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTEwOTQgNDk0IEwxMTAyIDQ5NCBMMTA4MiA1MDYgTDExMDggNTEyIEwxMTE5IDUyMiIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0xNTE1IDQxMyBMMTU0MiA0MjcgTDE1NjkgNDM4IEwxNTkyIDQ0NSIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuNiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTgxNSA1MDIgTDc5MyA1MDcgTDgwNSA1MDkgTDc4MCA1MDUiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik00MjcgNTEyIEw0MDYgNTEyIEw0MTYgNTAzIiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNMCA1NDUgTDYgNTM4IEwxIDU0NyIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTE0NCA0NTMgTDE0MCA0NDcgTDE1MiA0NDUgTDE0NiA0NTQgTDE0MSA0NTkiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0yMzYgNTI0IEwyMzggNTI5IEwyMjkgNTIxIEwyMTAgNTE0IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTcwMSA0NjcgTDcyNiA0NzkgTDcwOCA0ODUgTDY4MSA0ODEiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNNzQwIDQzNyBMNzQ2IDQyNyBMNzY2IDQzMyBMNzU3IDQ0MyBMNzg0IDQzNSIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik01MzQgNTMyIEw1MTYgNTMzIEw1MzcgNTMwIEw1NDMgNTM3IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTY3NSA0NTcgTDY5OCA0NzEgTDcyNCA0NjcgTDc0NyA0NjQgTDc3MSA0NjYiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNNDY0IDQ1MSBMNDY3IDQ1MiBMNDg1IDQ0MiBMNDU4IDQ0MCBMNDYwIDQzOCIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTE0MTggNDg4IEwxNDQxIDUwMSBMMTQzNSA1MDIgTDE0MTIgNDk5IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNNDY0IDUyMCBMNDU3IDUxNiBMNDU5IDUyNSIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0zIDUyMiBMLTMgNTMyIEwtMjYgNTQzIEwtNDcgNTQ1IEwtMjUgNTU3IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNOTc5IDQ0NSBMMTAwMSA0NTUgTDk5NCA0NDcgTDEwMTcgNDYwIiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS42IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNOTQ4IDUwMiBMOTI1IDUxNSBMOTA3IDUxMCBMODg3IDUwMCBMODY4IDUwOCIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuNiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTEzNDMgNDM3IEwxMzY3IDQ0NiBMMTM2OSA0NTcgTDEzNjMgNDUxIEwxMzcwIDQ1OCIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTQzIDQwMyBMNTYgMzk2IEw2MSA0MDkgTDQxIDQxMiBMNjggNDA4IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNNTcgNDY0IEw0NyA0NzAgTDM0IDQ4NCIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik02NjcgNDY2IEw2NjUgNDYwIEw2NDAgNDczIEw2MzQgNDc3IEw2NDggNDg1IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDAgMzgwKSBzY2FsZSgwLjQ1KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzhEOEEzRSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC00IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTgiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0xMiIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0xMCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMzUgMzgwKSBzY2FsZSgwLjQ1KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIzMCAzODApIHNjYWxlKDAuNDUpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjNEY3RDQ1IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTE4IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTIyIiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMjYiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMjQiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzI1IDM4MCkgc2NhbGUoMC40NSkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM4RDhBM0UiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtNCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC04IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMTIiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMTAiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDIwIDM4MCkgc2NhbGUoMC40NSkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg1MTUgMzgwKSBzY2FsZSgwLjQ1KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDYxMCAzODApIHNjYWxlKDAuNDUpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjOEQ4QTNFIiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTQiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtOCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTEyIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTEwIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDcwNSAzODApIHNjYWxlKDAuNDUpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjNEY3RDQ1IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTE4IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTIyIiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMjYiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMjQiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODAwIDM4MCkgc2NhbGUoMC40NSkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg1OCA0MjApIHNjYWxlKDAuNikiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxODEgNDIwKSBzY2FsZSgwLjYpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjNEY3RDQ1IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTE4IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTIyIiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMjYiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMjQiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzA0IDQyMCkgc2NhbGUoMC42KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzhEOEEzRSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC00IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTgiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0xMiIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0xMCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0MjcgNDIwKSBzY2FsZSgwLjYpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjNEY3RDQ1IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTE4IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTIyIiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMjYiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMjQiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNTUwIDQyMCkgc2NhbGUoMC42KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDY3MyA0MjApIHNjYWxlKDAuNikiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM4RDhBM0UiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtNCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC04IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMTIiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMTAiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNzk2IDQyMCkgc2NhbGUoMC42KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDc2IDQ3MCkgc2NhbGUoMC44KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIyNyA0NzApIHNjYWxlKDAuOCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM4RDhBM0UiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtNCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC04IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMTIiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMTAiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzc4IDQ3MCkgc2NhbGUoMC44KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDUyOSA0NzApIHNjYWxlKDAuOCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2ODAgNDcwKSBzY2FsZSgwLjgpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjOEQ4QTNFIiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTQiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtOCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTEyIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTEwIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDgzMSA0NzApIHNjYWxlKDAuOCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg5NCA1MzApIHNjYWxlKDEuMCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM4RDhBM0UiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtNCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC04IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMTIiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMTAiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjczIDUzMCkgc2NhbGUoMS4wKSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ1MiA1MzApIHNjYWxlKDEuMCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MzEgNTMwKSBzY2FsZSgxLjApIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjOEQ4QTNFIiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTQiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtOCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTEyIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTEwIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDgxMCA1MzApIHNjYWxlKDEuMCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZWxsaXBzZSBjeD0iMTE4MCIgY3k9IjQ1NSIgcng9IjI0NiIgcnk9IjY2IiBmaWxsPSIjOUM3NDQzIi8+CjxlbGxpcHNlIGN4PSIxMTgwIiBjeT0iNDU1IiByeD0iMjMwIiByeT0iNTgiIGZpbGw9IiNDOUE2NkUiLz4KPGcgY2xpcC1wYXRoPSJ1cmwoI2phZ3VleSkiPjxyZWN0IHg9Ijk0MCIgeT0iNDYyIiB3aWR0aD0iNDgwIiBoZWlnaHQ9IjYwIiBmaWxsPSJ1cmwoI2FndWEpIi8+CjxwYXRoIGQ9Ik05NjAgNDcwIHEgMzAgLTUgNjAgMCB0IDYwIDAgTTExMTAgNDgwIHEgMzAgLTUgNjAgMCB0IDYwIDAgTTEyNjAgNDcyIHEgMzAgLTUgNjAgMCIgc3Ryb2tlPSIjQTlEMkQ0IiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiIG9wYWNpdHk9Ii43Ii8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMzMwIDM5MikiPjxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSIxMiIgaGVpZ2h0PSIxMTgiIHJ4PSIzIiBmaWxsPSIjRjRFRkU0IiBzdHJva2U9IiM1QjQ2MzAiIHN0cm9rZS13aWR0aD0iMiIvPgo8bGluZSB4MT0iMCIgeTE9IjYiIHgyPSIxMiIgeTI9IjYiIHN0cm9rZT0iIzVCNDYzMCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz4KPGxpbmUgeDE9IjAiIHkxPSIyMCIgeDI9IjciIHkyPSIyMCIgc3Ryb2tlPSIjNUI0NjMwIiBzdHJva2Utd2lkdGg9IjEuNiIvPgo8bGluZSB4MT0iMCIgeTE9IjM0IiB4Mj0iMTIiIHkyPSIzNCIgc3Ryb2tlPSIjNUI0NjMwIiBzdHJva2Utd2lkdGg9IjEuNiIvPgo8bGluZSB4MT0iMCIgeTE9IjQ4IiB4Mj0iNyIgeTI9IjQ4IiBzdHJva2U9IiM1QjQ2MzAiIHN0cm9rZS13aWR0aD0iMS42Ii8+CjxsaW5lIHgxPSIwIiB5MT0iNjIiIHgyPSIxMiIgeTI9IjYyIiBzdHJva2U9IiM1QjQ2MzAiIHN0cm9rZS13aWR0aD0iMS42Ii8+CjxsaW5lIHgxPSIwIiB5MT0iNzYiIHgyPSI3IiB5Mj0iNzYiIHN0cm9rZT0iIzVCNDYzMCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz4KPGxpbmUgeDE9IjAiIHkxPSI5MCIgeDI9IjEyIiB5Mj0iOTAiIHN0cm9rZT0iIzVCNDYzMCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz4KPGxpbmUgeDE9IjAiIHkxPSIxMDQiIHgyPSI3IiB5Mj0iMTA0IiBzdHJva2U9IiM1QjQ2MzAiIHN0cm9rZS13aWR0aD0iMS42Ii8+CjxsaW5lIHgxPSIwIiB5MT0iMTE4IiB4Mj0iMTIiIHkyPSIxMTgiIHN0cm9rZT0iIzVCNDYzMCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz4KPHJlY3QgeD0iMiIgeT0iNzIiIHdpZHRoPSI4IiBoZWlnaHQ9IjQ0IiBmaWxsPSIjMkY3Qzg2Ii8+PC9nPgo8cGF0aCBkPSJNOTU1IDQ1NSBDIDkwMCA0NzAgODgwIDUwMCA4MjAgNTA1IiBzdHJva2U9IiMyRTNGM0EiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtZGFzaGFycmF5PSIxIDkiIG9wYWNpdHk9Ii44Ii8+CjxwYXRoIGQ9Ik04MDAgNDk4IHEgNSA4IDAgMTIgcSAtNSAtNCAwIC0xMiB6IiBmaWxsPSIjM0E3Q0E1Ii8+CjxwYXRoIGQ9Ik03ODUgNTEwIHEgNSA4IDAgMTIgcSAtNSAtNCAwIC0xMiB6IiBmaWxsPSIjM0E3Q0E1Ii8+CjxwYXRoIGQ9Ik03NzAgNTAwIHEgNSA4IDAgMTIgcSAtNSAtNCAwIC0xMiB6IiBmaWxsPSIjM0E3Q0E1Ii8+Cjwvc3ZnPg==") center bottom / cover no-repeat, #9FC7CF;
  display: flex;
  align-items: flex-start;
}
.hero__texto { max-width: 640px; padding: clamp(24px, 3.2vw, 48px) 5vw 0; }
.hero h1 {
  font-family: var(--titulo);
  font-weight: 700;
  font-size: clamp(2rem, 4.2vw, 3.4rem);
  line-height: 1.02;
  letter-spacing: -0.02em;
  margin: 0 0 12px;
  color: var(--tinta);
}
.hero__bajada { font-size: 1.12rem; margin: 0 0 14px; max-width: 34em; color: #21443D; }
.hero__fuente {
  display: inline-block; font-size: .85rem; padding: 4px 12px; border-radius: 999px;
  background: rgba(251, 252, 251, .78); border: 1px solid rgba(23, 51, 46, .18);
}

/* ---------- Pestañas ---------- */
.pestanas-contenedor { max-width: 1320px; margin: -34px auto 0; padding: 0 24px; position: relative; z-index: 2; }
.pestanas { border: none !important; gap: 6px; background: transparent; }
.pestana {
  font-family: var(--titulo) !important; font-size: 1.05rem !important; font-weight: 500;
  background: rgba(251, 252, 251, .86) !important; color: var(--tinta-suave) !important;
  border: 1px solid var(--borde) !important; border-bottom: none !important;
  border-radius: 12px 12px 0 0 !important; padding: 14px 22px !important;
}
.pestana:hover { color: var(--tinta) !important; background: var(--panel) !important; }
.pestana--activa {
  background: var(--fondo) !important; color: var(--tinta) !important; font-weight: 700 !important;
  border-top: 4px solid var(--rio) !important; border-left: 1px solid var(--borde) !important;
  border-right: 1px solid var(--borde) !important;
}
.pestana:focus-visible { outline: 3px solid var(--sol); outline-offset: -3px; }

/* ---------- Contenido ---------- */
.contenido, .con-rail { max-width: 1320px; margin: 0 auto; padding: 28px 24px 48px; }
.panel { background: var(--panel); border: 1px solid var(--borde); border-radius: var(--radio-panel); padding: 12px 14px; margin-bottom: 18px; }
.titulo-seccion { font-family: var(--titulo); font-size: 1.8rem; letter-spacing: -0.01em; margin: 0 0 10px; }
.subtitulo { font-family: var(--titulo); font-size: 1.2rem; margin: 6px 0 12px; font-weight: 700; }
p { max-width: 68ch; }

.contexto { display: grid; grid-template-columns: 1.25fr 1fr; gap: 28px; align-items: start; margin-bottom: 18px; }
.contexto__texto { padding: 6px 4px; }
.pasos { padding-left: 1.3em; margin: 0; }
.pasos li { margin-bottom: 10px; padding-left: 4px; }
.pasos li::marker { font-family: var(--titulo); font-weight: 700; color: var(--rio); }

/* Indicadores con la "regla de nivel" del reservorio */
.indicadores { display: flex; flex-direction: column; gap: 16px; }
.indicador { display: grid; grid-template-columns: 14px 1fr; gap: 14px; align-items: stretch; }
.regla {
  position: relative; border: 1.5px solid #5B4630; border-radius: 3px; background:
    repeating-linear-gradient(to top, transparent 0 9px, rgba(91,70,48,.35) 9px 10px), #F4EFE4;
  min-height: 56px; overflow: hidden;
}
.regla__nivel { position: absolute; left: 0; right: 0; bottom: 0; background: var(--rio); }
.indicador__valor { display: block; font-family: var(--titulo); font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.indicador__texto { display: block; color: var(--tinta-suave); font-size: .95rem; }

.tabla-ficha { width: 100%; border-collapse: collapse; }
.tabla-ficha th { text-align: left; vertical-align: top; width: 180px; padding: 8px 12px 8px 0; color: var(--tinta-suave); font-weight: 600; }
.tabla-ficha td { padding: 8px 0; border-bottom: 1px solid #E3EAE6; }

/* ---------- Diseño con barra de controles ---------- */
.con-rail { display: grid; grid-template-columns: 270px 1fr; gap: 22px; align-items: start; }
.rail {
  position: sticky; top: 16px; background: var(--tinta); color: #EAF2EE;
  border-radius: var(--radio-panel); padding: 20px 18px;
}
.control { margin-bottom: 20px; }
.control__etiqueta { display: block; font-weight: 600; margin-bottom: 6px; font-size: .95rem; }
.control__ayuda { font-size: .82rem; color: #A9C3BA; margin: 6px 0 0; }
.nota { font-size: .85rem; color: var(--tinta-suave); margin: 8px 0 0; }
.rail .nota { color: #A9C3BA; border-top: 1px solid rgba(234,242,238,.18); padding-top: 12px; }

.grilla { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.grilla > .panel { margin-bottom: 0; min-width: 0; }
.grilla .ancho { grid-column: 1 / -1; }
.modelos > div > .grilla, .modelos > div > .panel { margin-bottom: 18px; }
.grilla--resumen { grid-template-columns: 1fr 1.25fr; }

/* ---------- Dropdowns (react-select de Dash 2) ---------- */
.rail .Select-control {
  background: #F3F7F5; border: 1px solid transparent; border-radius: var(--radio-control);
  min-height: 40px; box-shadow: none; cursor: pointer;
}
.rail .Select-control:hover { border-color: var(--sol); }
.rail .is-focused:not(.is-open) > .Select-control { border-color: var(--sol); box-shadow: 0 0 0 3px rgba(214,158,46,.35); }
.rail .Select-value-label, .rail .Select-placeholder { color: var(--tinta) !important; font-weight: 500; line-height: 38px; }
.rail .Select-arrow { border-top-color: var(--rio); }
.Select-menu-outer { border: 1px solid var(--borde); border-radius: 0 0 var(--radio-control) var(--radio-control); box-shadow: 0 10px 24px rgba(23,51,46,.18); z-index: 20; }
.Select-option { color: var(--tinta); }
.Select-option.is-focused { background: #E1ECE7; }
.Select-option.is-selected { background: var(--rio); color: white; }

/* ---------- Sliders ---------- */
.rail .rc-slider-rail { background: rgba(234,242,238,.25); height: 6px; }
.rail .rc-slider-track { background: var(--sol); height: 6px; }
.rail .rc-slider-handle { border: 3px solid var(--sol); background: var(--tinta); width: 18px; height: 18px; margin-top: -6px; opacity: 1; }
.rail .rc-slider-handle:focus-visible { box-shadow: 0 0 0 4px rgba(214,158,46,.45); }
.rail .rc-slider-mark-text { color: #A9C3BA; font-size: .75rem; }
.rail .rc-slider-dot { border-color: rgba(234,242,238,.35); background: var(--tinta); }

/* ---------- Control segmentado (horizonte) ---------- */
.segmento { display: flex; background: rgba(234,242,238,.12); border-radius: var(--radio-control); padding: 3px; gap: 3px; }
.segmento__opcion {
  flex: 1; text-align: center; padding: 7px 4px; border-radius: 6px; cursor: pointer; font-size: .88rem;
  color: #CFE0D9; margin: 0 !important; white-space: nowrap;
}
.segmento__opcion:has(input:checked) { background: var(--sol); color: var(--tinta); font-weight: 700; }
.segmento__opcion:has(input:focus-visible) { outline: 2px solid #F6C453; }
.segmento__radio { position: absolute; opacity: 0; width: 1px; height: 1px; }

/* ---------- Tablas del tab de modelos ---------- */
.tabla-scroll { overflow-x: auto; }
.tabla-metricas { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: .93rem; }
.tabla-metricas th { text-align: right; font-weight: 600; color: var(--tinta-suave); padding: 6px 8px; border-bottom: 2px solid var(--borde); }
.tabla-metricas th:first-child, .tabla-metricas td:first-child { text-align: left; }
.tabla-metricas td { text-align: right; padding: 7px 8px; border-bottom: 1px solid #E3EAE6; }
.tabla-metricas td.ic { color: var(--tinta-suave); font-size: .85rem; }
.tabla-metricas .fila-trivial td { color: var(--tinta-suave); font-style: italic; }

.lista-fuga { list-style: none; padding: 0; margin: 0; }
.lista-fuga li { position: relative; padding: 6px 0 6px 30px; border-bottom: 1px dashed #E3EAE6; font-size: .93rem; }
.lista-fuga .check {
  position: absolute; left: 0; top: 7px; width: 20px; height: 20px; border-radius: 50%;
  background: var(--rio); color: white; font-size: .75rem; display: grid; place-items: center;
}

.pie { text-align: center; font-size: .85rem; color: var(--tinta-suave); padding: 20px 24px 36px; }

/* ---------- Responsivo ---------- */
@media (max-width: 980px) {
  .con-rail, .contexto, .grilla, .grilla--resumen { grid-template-columns: 1fr; }
  .rail { position: static; }
  .hero { min-height: 300px; aspect-ratio: auto; }
  .pestana { padding: 10px 12px !important; font-size: .95rem !important; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }

/* ---------- Entregable 1: piezas nuevas ---------- */
.rail input[type=number] {
  width: 100%; background: #F3F7F5; border: 1px solid transparent; border-radius: var(--radio-control);
  padding: 7px 10px; font: inherit; font-weight: 500; color: var(--tinta); min-height: 38px;
}
.rail input[type=number]:hover { border-color: var(--sol); }
.rail input[type=number]:focus-visible { outline: none; border-color: var(--sol); box-shadow: 0 0 0 3px rgba(214,158,46,.35); }
.par { display: grid; grid-template-columns: 1fr 1fr; gap: 0 10px; }
.par--3 { grid-template-columns: repeat(3, 1fr); }
.par .control { margin-bottom: 12px; }
.par .control__etiqueta { font-size: .85rem; }
.rail__grupo {
  font-family: var(--titulo); font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: var(--sol);
  margin: 16px 0 10px; padding-top: 12px; border-top: 1px solid rgba(234,242,238,.18);
}
.rail__grupo:first-child { margin-top: 0; padding-top: 0; border-top: none; }
.rail .segmento { flex-wrap: wrap; }
.rail .segmento__opcion { flex: 1 0 22%; }

.tarjetas { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 18px; }
.tarjeta { background: var(--panel); border: 1px solid var(--borde); border-top: 5px solid var(--rio);
  border-radius: var(--radio-panel); padding: 16px 20px; }
.tarjeta--conservador { border-top-color: var(--tierra); }
.tarjeta__etiqueta { display: block; font-weight: 600; color: var(--tinta-suave); font-size: .92rem; }
.tarjeta__valor { display: block; font-family: var(--titulo); font-weight: 700; font-size: clamp(2.2rem, 4vw, 3rem); line-height: 1.05; margin: 4px 0; }
.tarjeta__valor small { font-size: 1.05rem; font-weight: 500; color: var(--tinta-suave); margin-left: 6px; }
.tarjeta__detalle { font-size: .9rem; color: var(--tinta-suave); margin: 4px 0 0; }

.chip { display: inline-block; font-size: .74rem; font-weight: 700; padding: 2px 9px; border-radius: 999px; white-space: nowrap; }
.chip--ml { background: #DCEBE6; color: var(--rio); }
.chip--cte { background: #F1E4D3; color: #7A4A2C; }
.chip--calc { background: #E9E4F0; color: #4F3F6B; }
.chip--fuera { background: #ECEEEC; color: #5F6B66; }
.tabla-metricas .fila-modelo td { font-weight: 700; color: var(--tinta); }
.tabla-metricas td.texto { text-align: left; color: var(--tinta-suave); font-size: .88rem; }
.resumen-eda { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px 18px; margin: 4px 0 0; }
.resumen-eda div { border-left: 3px solid var(--rio); padding: 2px 0 2px 10px; }
.resumen-eda b { display: block; font-family: var(--titulo); font-size: 1.35rem; }
.resumen-eda span { font-size: .85rem; color: var(--tinta-suave); }
@media (max-width: 980px) {
  .tarjetas, .resumen-eda { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 560px) {
  .tarjetas, .resumen-eda, .par--3 { grid-template-columns: 1fr; }
}
.modelos .grilla, .modelos .panel, .modelos .tarjetas { margin-bottom: 18px; }
.modelos .grilla > .panel { margin-bottom: 0; }
.boton {
  width: 100%; margin-top: 8px; padding: 10px 12px; border-radius: var(--radio-control); cursor: pointer;
  background: transparent; color: #EAF2EE; border: 1px solid rgba(234,242,238,.45); font: inherit; font-weight: 600;
}
.boton:hover { border-color: var(--sol); color: #F6C453; }
.boton:focus-visible { outline: 2px solid #F6C453; outline-offset: 2px; }
"""

app.index_string = """<!DOCTYPE html>
<html lang="es">
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}<style>""" + CSS + """</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""


def grafica(id_, clase=""):
    return html.Div(dcc.Graph(id=id_, config=CONFIG_GRAF), className=f"panel {clase}".strip())


def control(etiqueta, componente, ayuda=None):
    hijos = [html.Label(etiqueta, className="control__etiqueta"), componente]
    if ayuda:
        hijos.append(html.P(ayuda, className="control__ayuda"))
    return html.Div(hijos, className="control")


# --------------------------------------------------------------------------
# Contexto
# --------------------------------------------------------------------------
def indicadores():
    w = semanal(DF_TRAIN, [OBJ])[OBJ]
    lluvia = DF_TRAIN[OBJ].mean() * 365.25
    secas = (w < UMBRAL_DEFECTO).mean()
    mensual = DF_TRAIN.set_index("fecha")[OBJ].resample("MS").sum()
    m = mensual.groupby(mensual.index.month).mean()
    meses_secos = int((m < 50).sum())
    res = entrenar(1, UMBRAL_DEFECTO)
    auc = res["modelos"]["logistica"]["metricas"]["AUC"]
    datos = [
        (f"{es(lluvia)} mm", f"lluvia al año en {ANIO_MIN}–{ANIO_MAX}, concentrada de abril a noviembre", min(lluvia / 1500, 1)),
        (f"{meses_secos} de 12", "meses con menos de 50 mm de lluvia en promedio", meses_secos / 12),
        (f"{secas:.0%}", f"de las semanas llueven menos de {UMBRAL_DEFECTO:g} mm (semana seca)", secas),
        (es(auc, 2), "AUC de la regresión logística en la prueba 2021–2025, una semana adelante", auc),
    ]
    return html.Div([
        html.Div([
            html.Div(html.Div(className="regla__nivel", style={"height": f"{min(max(v, 0.03), 1) * 100:.0f}%"}),
                     className="regla"),
            html.Div([html.Span(valor, className="indicador__valor"), html.Span(texto, className="indicador__texto")]),
        ], className="indicador") for valor, texto, v in datos
    ], className="indicadores")


def tab_contexto():
    variables = ", ".join(VARIABLES[v][0].lower() for v in VARIABLES if v in DF)
    return html.Div([
        html.Div([
            html.Div([
                html.H2("El problema", className="titulo-seccion"),
                html.P("En Sabanagrande (Atlántico) la lluvia llega en dos temporadas y de diciembre a marzo casi no llueve. "
                       "Una finca que siembra sin saber cuánta agua tendrá corre el riesgo de quedarse sin riego justo "
                       "cuando el cultivo más lo necesita."),
                html.P(["El ", html.Strong("Machine Learning"), " se usa solo para estimar el agua que aporta la naturaleza: "
                        "la lluvia de las próximas semanas. Los factores de la finca (reservorio, hectáreas, cultivo, riego) "
                        "no están en el Excel y se tratan como constantes en el análisis final del libro."]),
                html.H3("Qué muestra este tablero", className="subtitulo"),
                html.Ol([
                    html.Li([html.Strong("Datos. "), "Excel del entregable 1: clima diario de NASA POWER en la celda de la "
                             "finca, 1981–2025."]),
                    html.Li([html.Strong("Exploración. "), f"Solo las variables del Excel y solo {ANIO_MIN}–{ANIO_MAX}: "
                             "los años desde 2021 quedan reservados para la prueba."]),
                    html.Li([html.Strong("Modelos. "), "Regresión logística y KNN estiman la probabilidad de que la semana h "
                             "sea seca, y se comparan con la climatología y con el clasificador trivial."]),
                ], className="pasos"),
            ], className="contexto__texto"),
            html.Div([
                html.H3("En pocas cifras", className="subtitulo"),
                indicadores(),
            ], className="contexto__indicadores panel"),
        ], className="contexto"),
        html.Div(dcc.Graph(figure=fig_anio_tipico(DF_TRAIN), config=CONFIG_GRAF), className="panel"),
        html.Div([
            html.H3("La base de datos", className="subtitulo"),
            html.Table(html.Tbody([
                html.Tr([html.Th("Fuente"), html.Td(f"NASA POWER, comunidad AG (archivo: {FUENTE})")]),
                html.Tr([html.Th("Ubicación"), html.Td(f"Celda de la finca en Sabanagrande, Atlántico "
                                                       f"({es(LAT_FINCA, 2)}° N, {es(-LON_FINCA, 2)}° O)")]),
                html.Tr([html.Th("Periodo"), html.Td(f"{DF.fecha.min():%d/%m/%Y} a {DF.fecha.max():%d/%m/%Y}, diario "
                                                     f"({es(len(DF))} días)")]),
                html.Tr([html.Th("Variables"), html.Td(f"{len(OPC_VARIABLES)}: {variables}")]),
                html.Tr([html.Th("Variable objetivo"), html.Td("Lluvia acumulada en la semana h después de la emisión "
                                                               "del pronóstico (los lunes); para los clasificadores, si esa "
                                                               "lluvia queda por debajo de un umbral (semana seca)")]),
                html.Tr([html.Th("Partición"), html.Td(f"Entrenamiento 1981–2020; prueba desde {INICIO_TEST:%d/%m/%Y}, "
                                                       "que no se usa en la exploración ni en el ajuste")]),
            ]), className="tabla-ficha"),
        ], className="panel"),
    ], className="contenido")


# --------------------------------------------------------------------------
# Exploración de datos
# --------------------------------------------------------------------------
def tab_eda():
    rail = html.Aside([
        control("Variable", dcc.Dropdown(OPC_VARIABLES, "PRECTOTCORR", id="eda-var", clearable=False),
                "Se usa en la estacionalidad por mes."),
        control("Años", dcc.RangeSlider(ANIO_MIN, ANIO_MAX, 1, value=[ANIO_MIN, ANIO_MAX], id="eda-anios",
                                         marks={a: str(a) for a in range(ANIO_MIN, ANIO_MAX + 1, 10)},
                                         tooltip={"placement": "bottom"})),
        html.P(f"Solo variables del Excel. Los años desde {INICIO_TEST.year} quedan reservados para evaluar el modelo, "
               "así que no aparecen aquí. La ET0 no se analiza: es una construcción que se usa solo en el análisis final.",
               className="nota"),
    ], className="rail")
    grilla = html.Div([
        html.Div(id="eda-resumen", className="panel ancho"),
        grafica("eda-clima"), grafica("eda-dist"),
        grafica("eda-cajas"), grafica("eda-acf"),
        grafica("eda-calendario"), grafica("eda-corr"),
        grafica("eda-relacion"), grafica("eda-tendencia"),
        grafica("eda-faltantes", "ancho"),
    ], className="grilla")
    return html.Div([rail, grilla], className="con-rail")


def resumen_eda(d):
    w = semanal(d, [OBJ])[OBJ]
    a, rl, _ = tendencias(d)
    mensual = d.set_index("fecha")[OBJ].resample("MS").sum()
    m = mensual.groupby(mensual.index.month).mean()
    secos = [MESES[i - 1] for i in m.index if m[i] < 50]
    datos = [
        (f"{es(a[OBJ].mean())} mm" if len(a) else "–", "lluvia media anual"),
        (f"{es(w.mean(), 1)} / {es(w.median(), 1)} mm", "media / mediana semanal"),
        (es(w.skew(), 2), "asimetría semanal (log1p: " + es(np.log1p(w).skew(), 2) + ")"),
        (f"{es((w == 0).mean() * 100, 1)} %", "semanas sin lluvia"),
        (es(acf(w.values, 1)[0], 2), "autocorrelación a 1 semana"),
        (es(a[OBJ].std() / a[OBJ].mean(), 2) if len(a) > 1 else "–", "coeficiente de variación anual"),
        (f"{a[OBJ].idxmin()} · {a[OBJ].idxmax()}" if len(a) else "–",
         f"año más seco ({es(a[OBJ].min())} mm) y más lluvioso ({es(a[OBJ].max())} mm)" if len(a) else "años completos"),
        (f"{len(secos)} meses", "con menos de 50 mm en promedio: " + (", ".join(secos) or "ninguno")),
    ]
    return [html.H3(f"Resumen de {d.fecha.dt.year.min()}–{d.fecha.dt.year.max()}", className="subtitulo"),
            html.Div([html.Div([html.B(v), html.Span(t)]) for v, t in datos], className="resumen-eda"),
            html.P("La lluvia semanal es muy asimétrica (muchas semanas secas y pocos aguaceros grandes): por eso la "
                   "métrica principal es el RMSE y la referencia exigente es la climatología, no la media.", className="nota")]


@app.callback(
    [Output("eda-resumen", "children")] +
    [Output(i, "figure") for i in ["eda-clima", "eda-dist", "eda-cajas", "eda-acf", "eda-calendario", "eda-corr",
                                  "eda-relacion", "eda-tendencia", "eda-faltantes"]],
    Input("eda-var", "value"), Input("eda-anios", "value"))
def actualizar_eda(var, anios):
    anios = sorted(anios or [ANIO_MIN, ANIO_MAX])
    d = DF_TRAIN[DF_TRAIN.fecha.dt.year.between(anios[0], anios[1])]
    if d.fecha.dt.year.nunique() < 2:
        f = vacia("Elija un rango de al menos 2 años.")
        return [html.P("Elija un rango de al menos 2 años.", className="nota")] + [f] * 9
    return (resumen_eda(d), fig_climatologia(d), fig_distribucion(d), fig_cajas_mes(d, var), fig_acf(d),
            fig_calendario(d), fig_correlaciones(d), fig_relacion_objetivo(anios), fig_tendencia(d), fig_faltantes(d))


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def tab_modelos():
    rail = html.Aside([
        control("Horizonte (semanas)", dcc.RadioItems(
            [{"label": str(h), "value": h} for h in (1, 2, 4, 8)], 1, id="mod-h", className="segmento",
            labelClassName="segmento__opcion", inputClassName="segmento__radio"),
            "Semanas entre la emisión del pronóstico (lunes) y la semana que se predice."),
        control("Semana seca si llueve menos de", dcc.Dropdown(
            [{"label": f"{u:g} mm", "value": u} for u in (5.0, 10.0, 20.0)], UMBRAL_DEFECTO, id="mod-umbral",
            clearable=False)),
        control("Modelo en detalle", dcc.Dropdown(
            [{"label": NOMBRES["logistica"], "value": "logistica"}, {"label": NOMBRES["knn"], "value": "knn"}],
            "logistica", id="mod-modelo", clearable=False)),
        control("Umbral de probabilidad", dcc.Slider(0.1, 0.9, 0.05, value=0.5, id="mod-prob",
                                                    marks={0.1: "0,1", 0.5: "0,5", 0.9: "0,9"},
                                                    tooltip={"placement": "bottom"}),
                "Desde qué probabilidad se declara la semana como seca."),
        html.P("La primera vez que se elige una combinación, los modelos se entrenan con validación cruzada; "
               "tarda unos segundos.", className="nota"),
    ], className="rail")
    cuerpo = dcc.Loading(html.Div([
        html.Div([html.Div(id="mod-tabla", className="panel"), html.Div(id="mod-fuga", className="panel")],
                 className="grilla grilla--resumen"),
        grafica("mod-roc"),
        html.Div([grafica("mod-validacion"), grafica("mod-brecha")], className="grilla"),
        html.Div([grafica("mod-confusion"), grafica("mod-calibracion")], className="grilla"),
        html.Div([grafica("mod-extra"), grafica("mod-tiempo")], className="grilla"),
    ]), type="circle", color=RIO)
    return html.Div([rail, html.Div(cuerpo, className="modelos")], className="con-rail")


def lista_fuga(res):
    m = res["modelos"]
    items = [
        ("Partición cronológica", f"entrenamiento con semanas objetivo hasta {res['train_fin_objetivo']:%d/%m/%Y}; "
                                  f"prueba con emisiones desde {res['test_ini']:%d/%m/%Y}. "
                                  f"Fechas compartidas: {res['fechas_compartidas']}."),
        ("Purga", f"{res['purgadas']} semanas eliminadas porque su ventana objetivo cruzaba al periodo de prueba."),
        ("Solo pasado", f"las predictoras usan datos con al menos {LATENCIA} días de antigüedad; "
                        "los armónicos del ciclo anual son calendario."),
        ("Validación temporal", f"{N_SPLITS} folds de ventana creciente con {res['gap']} días de separación."),
        ("Pipeline", "imputación y escalado se ajustan dentro de cada fold, nunca con validación ni prueba."),
        ("Sobreajuste", f"C = {m['logistica']['mejor']:g} y k = {m['knn']['mejor']:g} se eligen por validación; "
                        "la prueba se usa una sola vez al final."),
    ]
    return [html.H3("Controles contra fuga y sobreajuste", className="subtitulo"),
            html.Ul([html.Li([html.Span("✓", className="check"), html.Strong(t + ": "), d]) for t, d in items],
                    className="lista-fuga"),
            html.P(f"Entrenamiento: {es(res['n_train'])} semanas ({res['prop_seca_train']:.0%} secas) · "
                   f"prueba: {es(res['n_test'])} ({res['prop_seca_test']:.0%} secas) · {len(res['cols'])} predictoras.",
                   className="nota")]


def tabla_metricas(res, prob):
    y = res["y_test"]
    filas = []
    for clave in ("climatologia", "logistica", "knn"):
        p = res["modelos"][clave]["p_test"]
        yhat = (p >= prob).astype(int)
        m = res["modelos"][clave]["metricas"]
        filas.append((NOMBRES[clave], es(m["AUC"], 3), f"[{es(m['AUC IC95 inf'], 3)}, {es(m['AUC IC95 sup'], 3)}]",
                      accuracy_score(y, yhat), precision_score(y, yhat, zero_division=0),
                      recall_score(y, yhat, zero_division=0), f1_score(y, yhat, zero_division=0), m["Brier"]))
    todo_seco = np.ones_like(y)
    filas.append(("Trivial: siempre seca", "0,500", "–", accuracy_score(y, todo_seco), precision_score(y, todo_seco),
                  1.0, f1_score(y, todo_seco), float(np.mean((1 - y) ** 2))))
    enc = ["Modelo", "AUC", "IC 95 %", "Accuracy", "Precisión", "Recall", "F1", "Brier"]
    cuerpo = [html.Tr([html.Td(f[0]), html.Td(f[1]), html.Td(f[2], className="ic")] +
                      [html.Td(es(v, 3)) for v in f[3:]],
                      className="fila-trivial" if f[0].startswith("Trivial") else "") for f in filas]
    return [html.H3(f"Desempeño en prueba 2021–2025 · h = {res['h']}, semana seca < {res['umbral']:g} mm",
                    className="subtitulo"),
            html.Div(html.Table([html.Thead(html.Tr([html.Th(e) for e in enc])), html.Tbody(cuerpo)],
                                className="tabla-metricas"), className="tabla-scroll"),
            html.P("Un modelo solo aporta si supera a la climatología, no solo al clasificador trivial. El IC del AUC se "
                   "calcula con bootstrap por bloques para respetar la autocorrelación. Brier: menor es mejor.",
                   className="nota")]


@app.callback(
    Output("mod-tabla", "children"), Output("mod-fuga", "children"), Output("mod-roc", "figure"),
    Output("mod-validacion", "figure"), Output("mod-brecha", "figure"), Output("mod-confusion", "figure"),
    Output("mod-calibracion", "figure"), Output("mod-extra", "figure"), Output("mod-tiempo", "figure"),
    Input("mod-h", "value"), Input("mod-umbral", "value"), Input("mod-modelo", "value"), Input("mod-prob", "value"))
def actualizar_modelos(h, umbral, clave, prob):
    res = entrenar(int(h or 1), float(umbral or UMBRAL_DEFECTO))
    clave = clave if clave in ("logistica", "knn") else "logistica"
    prob = 0.5 if prob is None else float(prob)
    if clave == "logistica":
        extra = fig_coeficientes(res)
    else:
        extra = fig_curva_validacion(res, "logistica")
        extra.update_layout(title="Referencia: curva de validación de la regresión logística")
    return (tabla_metricas(res, prob), lista_fuga(res), fig_roc_pr(res), fig_curva_validacion(res, clave),
            fig_brecha(res), fig_confusion(res["y_test"], res["modelos"][clave]["p_test"], prob), fig_calibracion(res),
            extra, fig_linea_tiempo(res, clave))


# --------------------------------------------------------------------------
# Layout general
# --------------------------------------------------------------------------
app.layout = html.Div([
    html.Header([
        html.Div([
            html.H1("¿Cuánta agua va a tener la finca?"),
            html.P("Pronóstico de la lluvia semanal con Machine Learning para una finca de Sabanagrande, Atlántico.",
                   className="hero__bajada"),
            html.Span("Entregable 1 · datos de NASA POWER, 1981–2025", className="hero__fuente"),
        ], className="hero__texto"),
    ], className="hero"),
    dcc.Tabs(id="pestanas", value="contexto", className="pestanas", parent_className="pestanas-contenedor",
             children=[
                 dcc.Tab(tab_contexto(), label="Contexto", value="contexto", className="pestana",
                         selected_className="pestana--activa"),
                 dcc.Tab(tab_eda(), label="Exploración de datos", value="eda", className="pestana",
                         selected_className="pestana--activa"),
                 dcc.Tab(tab_modelos(), label="Modelos", value="modelos", className="pestana",
                         selected_className="pestana--activa"),
             ]),
    html.Footer("Edwin Yunis y Jairo Serrano · Maestría en Ingeniería Industrial · Machine Learning, Dr. Lihki Rubio. "
                "Fuente: NASA POWER (NASA Langley Research Center).", className="pie"),
], className="app")


if __name__ == "__main__":
    app.run(debug=False, port=8050)
