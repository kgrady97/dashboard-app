# =============================================================================
# NYC Motor Vehicle Collisions — Unified Dash Dashboard
# =============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans, DBSCAN

# ── Color Palette ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#F5F7FA",
    "card":      "#FFFFFF",
    "border":    "#E2E8F0",
    "accent1":   "#FF4C6A",   # red-coral
    "accent2":   "#F7A325",   # amber
    "accent3":   "#3ECFCF",   # teal
    "accent4":   "#6C63FF",   # purple
    "text":      "#1A202C",
    "subtext":   "#718096",
    "grid":      "#E2E8F0",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=COLORS["text"]),
    margin=dict(l=40, r=20, t=40, b=40),
)

def _ax(**kw): return dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], **kw)

LEGEND = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11))

# ── Data Loading & Preprocessing ──────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    file_id = "1vKDGNyDM1BKhXYL_SSj0FP_JOlqrtrhP"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    df = pd.read_parquet(url)
    st.write(df.columns)   # 👈 ADD THIS LINE
    return df

    # ── Normalise dtypes ───────────────────────────────────────────────────────
    # Arrow-backed StringDtype uses pd.NA which breaks groupby/str ops → cast to object
    for col in df.select_dtypes(include="string").columns:
        df[col] = df[col].astype(object)
    # Also catch any remaining pyarrow / pandas StringDtype columns
    for col in df.columns:
        if hasattr(df[col], "dtype") and str(df[col].dtype) in ("string", "String"):
            df[col] = df[col].astype(object)
    # Nullable Int64 → plain int64 (avoids pd.NA issues in arithmetic)
    for col in df.select_dtypes(include="Int64").columns:
        df[col] = df[col].astype("int64")
    # int32 → int64 for consistency
    for col in df.select_dtypes(include="int32").columns:
        df[col] = df[col].astype("int64")

    # Severity flags
    df["injury_flag"] = (df["total_injured"] > 0).astype(int)
    df["fatal_flag"]  = (df["total_killed"]  > 0).astype(int)

    # Vehicle type normalisation
    vehicle_map = {
        "SEDAN": "Car", "PASSENGER VEHICLE": "Car",
        "STATION WAGON/SPORT UTILITY VEHICLE": "SUV",
        "SPORT UTILITY / STATION WAGON": "SUV",
        "TAXI": "Taxi", "TAXI CAB": "Taxi",
        "PICK-UP TRUCK": "Truck", "PICKUP TRUCK": "Truck", "BOX TRUCK": "Truck",
        "BUS": "Bus", "BIKE": "Bicycle", "BICYCLE": "Bicycle",
        "MOTORCYCLE": "Motorcycle",
    }
    vehicle_cols = [
        "vehicle_type_code1", "vehicle_type_code2",
        "vehicle_type_code_3", "vehicle_type_code_4", "vehicle_type_code_5",
    ]
    for col in vehicle_cols:
        if col in df.columns:
            df[col] = df[col].str.upper().replace(vehicle_map)

    return df


print("Loading data…")
df = load_data()

# ── Borough list for dropdown ──────────────────────────────────────────────────
BOROUGHS = sorted([b for b in df["borough"].dropna().unique()])
ALL_KEYS  = ["ALL"] + BOROUGHS

# ── Pre-aggregate everything keyed by borough at startup ──────────────────────
# Callbacks then just do O(1) dict lookups — no per-request groupby on large df.

def _build_cache(d, key):
    """Build all aggregations for a given slice of df."""
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    # vehicle counts
    vcols = [c for c in ["vehicle_type_code1","vehicle_type_code2",
                          "vehicle_type_code_3","vehicle_type_code_4",
                          "vehicle_type_code_5"] if c in d.columns]
    vehicle_counts = (pd.concat([d[c] for c in vcols])
                        .dropna().value_counts().head(10))

    # contributing factors
    cf = d.dropna(subset=["contributing_factor_vehicle_1"])
    factor_inj = (cf.groupby("contributing_factor_vehicle_1")["total_injured"]
                   .mean().sort_values(ascending=False).head(10).sort_values())
    factor_fat = (cf.groupby("contributing_factor_vehicle_1")["total_killed"]
                   .mean().sort_values(ascending=False).head(10).sort_values())

    # ── Clustering (on lat/lon clean subset) ──────────────────────────────────
    # Sample BEFORE fitting — DBSCAN is O(n²) in memory; fitting on 400k+ rows
    # will exhaust RAM and hang/crash the process.  10k points is more than
    # enough to capture the spatial cluster structure for a map visualisation.
    GEO_SAMPLE = 10_000
    d_geo_full = d.dropna(subset=["latitude", "longitude"])
    d_geo = (d_geo_full.sample(GEO_SAMPLE, random_state=42)
             if len(d_geo_full) > GEO_SAMPLE else d_geo_full.copy())
    X = d_geo[["latitude", "longitude"]].values

    # K-Means (k=5)
    if len(X) >= 5:
        km = KMeans(n_clusters=5, random_state=42, n_init="auto")
        km_labels = km.fit_predict(X).astype(str)
    else:
        km_labels = np.full(len(X), "0")

    # DBSCAN — eps=0.01 ≈ ~1 km in lat/lon degrees, safe on the sampled subset
    if len(X) >= 50:
        db = DBSCAN(eps=0.01, min_samples=10)
        db_labels = db.fit_predict(X).astype(str)
    else:
        db_labels = np.full(len(X), "-1")

    # Store only the three columns needed by the map callback — avoids keeping
    # full-width DataFrame copies (all 50+ columns) in memory for every borough.
    d_geo = pd.DataFrame({
        "latitude":   d_geo["latitude"].values,
        "longitude":  d_geo["longitude"].values,
        "cluster_km": km_labels,
        "cluster_db": db_labels,
    })

    # Heatmap sample — density maps benefit from more points than clustering.
    # 30k gives smooth density contours without serialising huge JSON to browser.
    HEAT_SAMPLE = 30_000
    heat_src = d_geo_full[["latitude", "longitude"]]
    heatmap_geo = (heat_src.sample(HEAT_SAMPLE, random_state=0).reset_index(drop=True)
                   if len(heat_src) > HEAT_SAMPLE else heat_src.reset_index(drop=True))

    # Top 10 most dangerous intersections by collision count
    top_intersections = (
        d_geo_full
        .groupby(["latitude", "longitude"])
        .size()
        .reset_index(name="collision_count")
        .sort_values("collision_count", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return {
        # KPIs
        "total":   len(d),
        "injured": int(d["injury_flag"].sum()),
        "fatal":   int(d["fatal_flag"].sum()),
        "avg_inj": float(d["total_injured"].mean()),
        "avg_fat": float(d["total_killed"].mean()),
        # hourly
        "hour_counts":   d.groupby("hour").size(),
        "hour_inj_mean": d.groupby("hour")["total_injured"].mean(),
        "hour_fat_mean": d.groupby("hour")["total_killed"].mean(),
        # borough
        "borough_counts": d["borough"].value_counts(),
        # severity by day
        "day_inj": d.groupby("day_of_week")["total_injured"].mean().reindex(day_order),
        "day_fat": d.groupby("day_of_week")["total_killed"].mean().reindex(day_order),
        # severity by month
        "month_inj": d.groupby("month")["total_injured"].mean(),
        "month_fat": d.groupby("month")["total_killed"].mean(),
        # vehicle
        "vehicle_counts": vehicle_counts,
        # road user
        "ped_inj":  int(d["number_of_pedestrians_injured"].sum()),
        "cyc_inj":  int(d["number_of_cyclist_injured"].sum()),
        "mot_inj":  int(d["number_of_motorist_injured"].sum()),
        "ped_fat":  int(d["number_of_pedestrians_killed"].sum()),
        "cyc_fat":  int(d["number_of_cyclist_killed"].sum()),
        "mot_fat":  int(d["number_of_motorist_killed"].sum()),
        # donut
        "no_inj":   int((d["injury_flag"] == 0).sum()),
        "inj_only": int(((d["injury_flag"] == 1) & (d["fatal_flag"] == 0)).sum()),
        "fatal_ct": int((d["fatal_flag"] == 1).sum()),
        # factors
        "factor_inj": factor_inj,
        "factor_fat": factor_fat,
        # stats table
        "stats_df": d,
        # severe
        "severe_df": d,
        # clustering
        "cluster_geo_df": d_geo,
        # heatmap
        "heatmap_geo": heatmap_geo,
        # top 10 dangerous intersections
        "top_intersections": top_intersections,
    }

print("Pre-aggregating cache…")
CACHE = {}
CACHE["ALL"] = _build_cache(df, "ALL")
for b in BOROUGHS:
    CACHE[b] = _build_cache(df[df["borough"] == b], b)
print("Cache ready.")

def get(borough):
    return CACHE.get(borough, CACHE["ALL"])


# ── App Bootstrap ─────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap",
    ],
    title="NYC Collision Intel",
)

# ── KPI Card factory ──────────────────────────────────────────────────────────
def kpi_card(card_id, label, accent):
    return html.Div(
        [
            html.P(label, style={"fontSize": "11px", "letterSpacing": "1.5px",
                                 "textTransform": "uppercase", "color": COLORS["subtext"],
                                 "marginBottom": "6px"}),
            html.H3(id=card_id, style={"fontSize": "28px", "fontWeight": "700",
                                        "color": accent, "margin": 0,
                                        "fontFamily": "'Space Mono', monospace"}),
        ],
        style={
            "background": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderTop": f"3px solid {accent}",
            "borderRadius": "8px",
            "padding": "20px 24px",
            "flex": "1",
        },
    )


# ── Layout ────────────────────────────────────────────────────────────────────
def chart_card(title, graph_id, height=340):
    return html.Div(
        [
            html.P(title, style={"fontSize": "11px", "letterSpacing": "1.5px",
                                 "textTransform": "uppercase", "color": COLORS["subtext"],
                                 "marginBottom": "4px", "fontWeight": "600"}),
            dcc.Graph(
                id=graph_id,
                config={"displayModeBar": False},
                style={"height": f"{height}px"},
            ),
        ],
        style={
            "background": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "padding": "20px",
        },
    )


app.layout = html.Div(
    style={"background": COLORS["bg"], "minHeight": "100vh",
           "fontFamily": "'DM Sans', sans-serif", "color": COLORS["text"],
           "padding": "32px 40px"},
    children=[

        # ── Header ────────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H1("NYC COLLISION INTEL",
                        style={"fontSize": "26px", "fontWeight": "700",
                               "letterSpacing": "3px", "margin": 0,
                               "fontFamily": "'Space Mono', monospace"}),
                html.P("Motor Vehicle Collision Analysis Dashboard",
                       style={"color": COLORS["subtext"], "margin": "4px 0 0 0",
                               "fontSize": "13px"}),
            ]),
            html.Div([
                html.Label("FILTER BY BOROUGH",
                           style={"fontSize": "10px", "letterSpacing": "1px",
                                  "color": COLORS["subtext"], "display": "block",
                                  "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="borough-filter",
                    options=[{"label": "All Boroughs", "value": "ALL"}] +
                            [{"label": b, "value": b} for b in BOROUGHS],
                    value="ALL",
                    clearable=False,
                    style={"width": "220px", "fontSize": "13px"},
                ),
            ]),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "flex-end", "marginBottom": "28px"}),

        # ── KPI Row ───────────────────────────────────────────────────────────
        html.Div(
            [
                kpi_card("kpi-total",    "Total Crashes",           COLORS["accent3"]),
                kpi_card("kpi-injured",  "Crashes w/ Injury",       COLORS["accent2"]),
                kpi_card("kpi-fatal",    "Crashes w/ Fatality",     COLORS["accent1"]),
                kpi_card("kpi-avg-inj",  "Avg Injuries / Crash",    COLORS["accent4"]),
                kpi_card("kpi-avg-fat",  "Avg Fatalities / Crash",  COLORS["accent1"]),
            ],
            style={"display": "flex", "gap": "16px", "marginBottom": "20px"},
        ),

        # ── Row 1: Hour + Borough ─────────────────────────────────────────────
        html.Div([
            html.Div(chart_card("Collisions by Hour of Day",   "chart-hour"),
                     style={"flex": "1"}),
            html.Div(chart_card("Collisions by Borough",       "chart-borough"),
                     style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # ── Row 2: Severity by Hour + Severity by Day ─────────────────────────
        html.Div([
            html.Div(chart_card("Avg Injuries & Fatalities by Hour", "chart-sev-hour"),
                     style={"flex": "1"}),
            html.Div(chart_card("Avg Severity by Day of Week",       "chart-sev-day"),
                     style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # ── Row 3: Severity by Month + Vehicle Types ──────────────────────────
        html.Div([
            html.Div(chart_card("Avg Severity by Month",   "chart-sev-month"),
                     style={"flex": "1"}),
            html.Div(chart_card("Top Vehicle Types Involved", "chart-vehicle"),
                     style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # ── Row 4: Road user breakdown + Injury/Fatal donut ───────────────────
        html.Div([
            html.Div(chart_card("Injuries by Road User Type",   "chart-user-inj"),
                     style={"flex": "2"}),
            html.Div(chart_card("Crash Severity Split",         "chart-donut"),
                     style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # ── Row 5: Contributing factors injury + fatality ─────────────────────
        html.Div([
            html.Div(chart_card("Top Factors — Avg Injury Severity",    "chart-factor-inj", 380),
                     style={"flex": "1"}),
            html.Div(chart_card("Top Factors — Avg Fatality Severity",  "chart-factor-fat", 380),
                     style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # ── Row 6: Collision Density Heatmap (full-width) ─────────────────────
        html.Div([
            html.P("COLLISION DENSITY HEATMAP",
                   style={"fontSize": "11px", "letterSpacing": "1.5px",
                           "textTransform": "uppercase", "color": COLORS["subtext"],
                           "marginBottom": "10px", "fontWeight": "600"}),
            dcc.Graph(
                id="chart-heatmap",
                config={"displayModeBar": False},
                style={"height": "480px"},
            ),
        ], style={
            "background": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "padding": "20px",
            "marginBottom": "16px",
        }),

        # ── Row 7: Geospatial Clustering (full-width) ─────────────────────────
        html.Div([
            html.P("GEOSPATIAL COLLISION CLUSTERING",
                   style={"fontSize": "11px", "letterSpacing": "1.5px",
                           "textTransform": "uppercase", "color": COLORS["subtext"],
                           "marginBottom": "10px", "fontWeight": "600"}),
            html.Div([
                html.Label("Algorithm:",
                           style={"fontSize": "11px", "color": COLORS["subtext"],
                                  "marginRight": "12px", "alignSelf": "center"}),
                dcc.RadioItems(
                    id="cluster-method",
                    options=[
                        {"label": " K-Means (k=5)", "value": "kmeans"},
                        {"label": " DBSCAN (eps=0.01, min_samples=10)", "value": "dbscan"},
                    ],
                    value="kmeans",
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "20px", "fontSize": "12px",
                                "color": COLORS["text"]},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            dcc.Graph(
                id="chart-cluster",
                config={"displayModeBar": False},
                style={"height": "480px"},
            ),
        ], style={
            "background": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "padding": "20px",
            "marginBottom": "16px",
        }),

        # ── Row 8: Top 10 Most Dangerous Intersections ────────────────────────
        html.Div([
            html.P("TOP 10 MOST DANGEROUS INTERSECTIONS",
                   style={"fontSize": "11px", "letterSpacing": "1.5px",
                           "textTransform": "uppercase", "color": COLORS["subtext"],
                           "marginBottom": "10px", "fontWeight": "600"}),
            dcc.Graph(
                id="chart-hotspots",
                config={"displayModeBar": False},
                style={"height": "480px"},
            ),
        ], style={
            "background": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "padding": "20px",
            "marginBottom": "16px",
        }),

        # ── Row 9: Statistics table ───────────────────────────────────────────
        html.Div([
            html.Div([
                html.P("SUMMARY STATISTICS",
                       style={"fontSize": "11px", "letterSpacing": "1.5px",
                               "textTransform": "uppercase", "color": COLORS["subtext"],
                               "marginBottom": "12px", "fontWeight": "600"}),
                html.Div(id="stats-table"),
            ], style={
                "background": COLORS["card"],
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "8px",
                "padding": "20px",
                "flex": "1",
            }),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # ── Row 10: Severe crashes table ──────────────────────────────────────
        html.Div([
            html.P("MOST SEVERE CRASHES  (total_injured ≥ 10)",
                   style={"fontSize": "11px", "letterSpacing": "1.5px",
                           "textTransform": "uppercase", "color": COLORS["subtext"],
                           "marginBottom": "12px", "fontWeight": "600"}),
            html.Div(id="severe-table"),
        ], style={
            "background": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "padding": "20px",
        }),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

GC = dict(gridcolor=COLORS["grid"])  # shorthand grid color

@app.callback(
    Output("kpi-total","children"), Output("kpi-injured","children"),
    Output("kpi-fatal","children"), Output("kpi-avg-inj","children"),
    Output("kpi-avg-fat","children"),
    Input("borough-filter","value"),
)
def update_kpis(borough):
    c = get(borough)
    return (f"{c['total']:,}", f"{c['injured']:,}", f"{c['fatal']:,}",
            f"{c['avg_inj']:.3f}", f"{c['avg_fat']:.4f}")


@app.callback(Output("chart-hour","figure"), Input("borough-filter","value"))
def chart_hour(borough):
    c = get(borough)
    counts = c["hour_counts"].reset_index()
    counts.columns = ["hour", "count"]
    fig = go.Figure(go.Bar(
        x=counts["hour"], y=counts["count"],
        marker_color=COLORS["accent3"], marker_line_width=0,
        hovertemplate="Hour %{x}:00 — %{y:,} crashes<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        showlegend=False,
        xaxis=dict(title="Hour of Day", tickmode="linear", **GC),
        yaxis=dict(title="Crashes", **GC),
    )
    return fig


@app.callback(Output("chart-borough","figure"), Input("borough-filter","value"))
def chart_borough(borough):
    c = get(borough)
    bc = c["borough_counts"].reset_index()
    bc.columns = ["borough", "count"]
    fig = go.Figure(go.Bar(
        x=bc["count"], y=bc["borough"], orientation="h",
        marker_color=COLORS["accent2"], marker_line_width=0,
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        showlegend=False,
        xaxis=dict(title="Crashes", **GC),
        yaxis=dict(**GC),
    )
    return fig


@app.callback(Output("chart-sev-hour","figure"), Input("borough-filter","value"))
def chart_sev_hour(borough):
    c = get(borough)
    hi, hk = c["hour_inj_mean"], c["hour_fat_mean"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hi.index, y=hi.values, name="Avg Injured",
                             line=dict(color=COLORS["accent2"], width=2),
                             mode="lines+markers"))
    fig.add_trace(go.Scatter(x=hk.index, y=hk.values, name="Avg Killed",
                             line=dict(color=COLORS["accent1"], width=2),
                             mode="lines+markers", yaxis="y2"))
    fig.update_layout(
        **PLOT_LAYOUT,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h", x=1, y=1.02, xanchor="right", yanchor="bottom"),
        xaxis=dict(title="Hour", tickmode="linear", **GC),
        yaxis=dict(title="Avg Injured", **GC),
        yaxis2=dict(title="Avg Killed", overlaying="y", side="right",
                    showgrid=False, color=COLORS["accent1"]),
    )
    return fig


@app.callback(Output("chart-sev-day","figure"), Input("borough-filter","value"))
def chart_sev_day(borough):
    c = get(borough)
    di, dk = c["day_inj"], c["day_fat"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Avg Injured", x=di.index, y=di.values,
                         marker_color=COLORS["accent2"], offsetgroup=0))
    fig.add_trace(go.Bar(name="Avg Killed", x=dk.index, y=dk.values,
                         marker_color=COLORS["accent1"], offsetgroup=1))
    fig.update_layout(
        **PLOT_LAYOUT,
        barmode="group",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h", x=1, y=1.02, xanchor="right", yanchor="bottom"),
        xaxis=dict(**GC),
        yaxis=dict(title="Avg per Crash", **GC),
    )
    return fig


@app.callback(Output("chart-sev-month","figure"), Input("borough-filter","value"))
def chart_sev_month(borough):
    c = get(borough)
    mi, mk = c["month_inj"], c["month_fat"]
    months_abbr = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mi.index, y=mi.values, name="Avg Injured",
                             line=dict(color=COLORS["accent2"], width=2),
                             mode="lines+markers"))
    fig.add_trace(go.Scatter(x=mk.index, y=mk.values, name="Avg Killed",
                             line=dict(color=COLORS["accent1"], width=2),
                             mode="lines+markers", yaxis="y2"))
    fig.update_layout(
        **PLOT_LAYOUT,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h", x=1, y=1.02, xanchor="right", yanchor="bottom"),
        xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
                   ticktext=months_abbr, **GC),
        yaxis=dict(title="Avg Injured", **GC),
        yaxis2=dict(title="Avg Killed", overlaying="y", side="right",
                    showgrid=False, color=COLORS["accent1"]),
    )
    return fig


@app.callback(Output("chart-vehicle","figure"), Input("borough-filter","value"))
def chart_vehicle(borough):
    c = get(borough)
    counts = c["vehicle_counts"]
    palette = [COLORS["accent3"], COLORS["accent4"], COLORS["accent2"],
               COLORS["accent1"], "#A0E4CB", "#FFD89B", "#C9B8FF",
               "#FF8FA3", "#80E8E8", "#FFB347"]
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker_color=palette[:len(counts)], marker_line_width=0,
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        showlegend=False,
        xaxis=dict(title="Occurrences", **GC),
        yaxis=dict(**GC),
    )
    return fig


@app.callback(Output("chart-user-inj","figure"), Input("borough-filter","value"))
def chart_user_inj(borough):
    c = get(borough)
    labels = ["Pedestrians", "Cyclists", "Motorists"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Injured", x=labels,
                         y=[c["ped_inj"], c["cyc_inj"], c["mot_inj"]],
                         marker_color=COLORS["accent2"]))
    fig.add_trace(go.Bar(name="Killed", x=labels,
                         y=[c["ped_fat"], c["cyc_fat"], c["mot_fat"]],
                         marker_color=COLORS["accent1"]))
    fig.update_layout(
        **PLOT_LAYOUT,
        barmode="group",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h", x=1, y=1.02, xanchor="right", yanchor="bottom"),
        xaxis=dict(**GC),
        yaxis=dict(title="Count", **GC),
    )
    return fig


@app.callback(Output("chart-donut","figure"), Input("borough-filter","value"))
def chart_donut(borough):
    c = get(borough)
    fig = go.Figure(go.Pie(
        labels=["No Injury", "Injury", "Fatal"],
        values=[c["no_inj"], c["inj_only"], c["fatal_ct"]],
        hole=0.55,
        marker=dict(colors=[COLORS["accent3"], COLORS["accent2"], COLORS["accent1"]]),
        textfont=dict(size=12, color=COLORS["text"]),
        insidetextorientation="radial",
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=COLORS["text"]),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11),
                    orientation="h", x=1, y=1.02, xanchor="right", yanchor="bottom"),
    )
    return fig


@app.callback(Output("chart-factor-inj","figure"), Input("borough-filter","value"))
def chart_factor_inj(borough):
    fi = get(borough)["factor_inj"]
    fig = go.Figure(go.Bar(
        x=fi.values, y=fi.index, orientation="h",
        marker=dict(color=fi.values,
                    colorscale=[[0, COLORS["accent4"]], [1, COLORS["accent3"]]],
                    line_width=0),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        showlegend=False,
        xaxis=dict(title="Avg Injuries per Crash", **GC),
        yaxis=dict(**GC),
    )
    return fig


@app.callback(Output("chart-factor-fat","figure"), Input("borough-filter","value"))
def chart_factor_fat(borough):
    ff = get(borough)["factor_fat"]
    fig = go.Figure(go.Bar(
        x=ff.values, y=ff.index, orientation="h",
        marker=dict(color=ff.values,
                    colorscale=[[0, "#FF8C8C"], [1, COLORS["accent1"]]],
                    line_width=0),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        showlegend=False,
        xaxis=dict(title="Avg Fatalities per Crash", **GC),
        yaxis=dict(**GC),
    )
    return fig


@app.callback(Output("stats-table","children"), Input("borough-filter","value"))
def stats_table(borough):
    d = get(borough)["stats_df"]
    num_cols = [
        "number_of_persons_injured","number_of_persons_killed",
        "number_of_pedestrians_injured","number_of_pedestrians_killed",
        "number_of_cyclist_injured","number_of_cyclist_killed",
        "number_of_motorist_injured","number_of_motorist_killed",
        "total_injured","total_killed",
    ]
    existing = [c for c in num_cols if c in d.columns]
    stats = d[existing].describe().T.reset_index().rename(columns={"index": "column"})
    stats = stats[["column","count","mean","std","min","50%","max"]]
    stats.columns = ["Column","Count","Mean","Std","Min","Median","Max"]
    for col in ["Mean","Std","Min","Median","Max"]:
        stats[col] = stats[col].map("{:.3f}".format)
    stats["Count"] = stats["Count"].map("{:.0f}".format)
    return dash_table.DataTable(
        data=stats.to_dict("records"),
        columns=[{"name": c, "id": c} for c in stats.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": COLORS["border"], "color": COLORS["text"],
                      "fontWeight": "600", "fontSize": "11px", "border": "none"},
        style_data={"backgroundColor": COLORS["card"], "color": COLORS["text"],
                    "fontSize": "12px", "border": f"1px solid {COLORS['border']}",
                    "fontFamily": "'Space Mono', monospace"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg"]}
        ],
    )


@app.callback(Output("severe-table","children"), Input("borough-filter","value"))
def severe_table(borough):
    d = get(borough)["severe_df"]
    cols = ["borough","crash_datetime","hour","day_of_week","total_injured",
            "total_killed","contributing_factor_vehicle_1","vehicle_type_code1"]
    cols = [c for c in cols if c in d.columns]
    severe = (d[d["total_injured"] >= 10][cols]
               .sort_values("total_injured", ascending=False)
               .head(20).reset_index(drop=True))
    severe.columns = [c.replace("_"," ").title() for c in severe.columns]
    return dash_table.DataTable(
        data=severe.to_dict("records"),
        columns=[{"name": c, "id": c} for c in severe.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": COLORS["border"], "color": COLORS["text"],
                      "fontWeight": "600", "fontSize": "11px", "border": "none"},
        style_data={"backgroundColor": COLORS["card"], "color": COLORS["text"],
                    "fontSize": "12px", "border": f"1px solid {COLORS['border']}"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg"]},
            {"if": {"filter_query": "{Total Injured} >= 15", "column_id": "Total Injured"},
             "color": COLORS["accent1"], "fontWeight": "700"},
        ],
        page_size=10,
    )


@app.callback(Output("chart-heatmap", "figure"), Input("borough-filter", "value"))
def chart_heatmap(borough):
    h = get(borough)["heatmap_geo"]
    fig = go.Figure(go.Densitymap(
        lat=h["latitude"],
        lon=h["longitude"],
        radius=8,
        colorscale=[
            [0.0, "rgba(0,0,0,0)"],
            [0.2, "#6C63FF"],
            [0.5, "#F7A325"],
            [1.0, "#FF4C6A"],
        ],
        showscale=True,
        colorbar=dict(thickness=10, len=0.6, title=dict(text="Density", side="right"),
                      tickfont=dict(size=10)),
        hovertemplate="Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
    ))
    map_layout = {**PLOT_LAYOUT, "margin": dict(l=0, r=0, t=0, b=0)}
    fig.update_layout(
        **map_layout,
        map=dict(
            style="carto-positron",
            center=dict(lat=40.73, lon=-73.93),
            zoom=10,
        ),
        height=460,
    )
    return fig


@app.callback(
    Output("chart-cluster", "figure"),
    Input("borough-filter", "value"),
    Input("cluster-method", "value"),
)
def chart_cluster(borough, method):
    d_geo = get(borough)["cluster_geo_df"]

    CLUSTER_PALETTE = [
        "#6C63FF", "#3ECFCF", "#F7A325", "#FF4C6A", "#A0E4CB",
        "#FFD89B", "#C9B8FF", "#FF8FA3", "#80E8E8", "#FFB347",
    ]

    if method == "kmeans":
        col = "cluster_km"
        title_suffix = "K-Means (k=5)"
        noise_label = None
    else:
        col = "cluster_db"
        title_suffix = "DBSCAN (eps=0.01, min_samples=50)"
        noise_label = "-1"

    clusters = sorted(d_geo[col].unique(), key=lambda x: int(x))
    fig = go.Figure()

    for i, cid in enumerate(clusters):
        subset = d_geo[d_geo[col] == cid]
        is_noise = (cid == noise_label)
        color = "#CCCCCC" if is_noise else CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        name = "Noise" if is_noise else f"Cluster {cid}"
        sample = subset
        fig.add_trace(go.Scattermap(
            lat=sample["latitude"],
            lon=sample["longitude"],
            mode="markers",
            marker=dict(size=4, color=color, opacity=0.6),
            name=name,
            hovertemplate=f"{name}<br>Lat: %{{lat:.4f}}<br>Lon: %{{lon:.4f}}<extra></extra>",
        ))

    map_layout = {**PLOT_LAYOUT, "margin": dict(l=0, r=0, t=0, b=0)}
    fig.update_layout(
        **map_layout,
        map=dict(
            style="carto-positron",
            center=dict(lat=40.73, lon=-73.93),
            zoom=10,
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(size=11),
                    x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bordercolor=COLORS["border"], borderwidth=1),
        height=460,
    )
    return fig


@app.callback(Output("chart-hotspots", "figure"), Input("borough-filter", "value"))
def chart_hotspots(borough):
    ti = get(borough)["top_intersections"]
    fig = go.Figure()

    # Bubble size scaled to collision count
    max_count = ti["collision_count"].max()
    sizes = (ti["collision_count"] / max_count * 30 + 12).tolist()

    fig.add_trace(go.Scattermap(
        lat=ti["latitude"],
        lon=ti["longitude"],
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=ti["collision_count"],
            colorscale=[[0, COLORS["accent2"]], [0.5, COLORS["accent1"]], [1, "#8B0000"]],
            showscale=True,
            colorbar=dict(thickness=10, len=0.6,
                          title=dict(text="Collisions", side="right"),
                          tickfont=dict(size=10)),
            opacity=0.85,
        ),
        text=[str(i + 1) for i in range(len(ti))],
        textfont=dict(size=10, color="white"),
        customdata=ti[["collision_count"]].values,
        hovertemplate=(
            "<b>Rank #%{text}</b><br>"
            "Lat: %{lat:.5f}<br>"
            "Lon: %{lon:.5f}<br>"
            "Collisions: %{customdata[0]}<extra></extra>"
        ),
    ))

    map_layout = {**PLOT_LAYOUT, "margin": dict(l=0, r=0, t=0, b=0)}
    fig.update_layout(
        **map_layout,
        map=dict(
            style="carto-positron",
            center=dict(lat=float(ti["latitude"].mean()),
                        lon=float(ti["longitude"].mean())),
            zoom=11,
        ),
        showlegend=False,
        height=460,
    )
    return fig


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
