"""Plotly chart builders for the dashboard.

Colour roles (validated reference palette, light theme):
  SERIES_1 blue    - actual / main series
  SERIES_2 orange  - predicted / revised series
  CRITICAL red     - reserved for anomalies (status colour, always with a label)
  NEUTRAL grey     - context (normal transactions)
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
NEUTRAL = "#b5b3ab"
SEQ_BLUE = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig: go.Figure, title: str = "", height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        height=height, margin=dict(l=8, r=8, t=48 if title else 16, b=8 if not legend else 40),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, color=INK_2, size=12),
        showlegend=legend, legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0,
                                       font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False,
                     tickprefix="₹", separatethousands=True)
    return fig


def category_bar(tx: pd.DataFrame, month: str | None = None) -> go.Figure:
    """Horizontal bars: spending per category (single hue = magnitude)."""
    exp = tx[tx["type"] == "Expense"].copy()
    if month:
        exp = exp[exp["date"].str[:7] == month]
    s = exp.groupby("category")["amount"].sum().sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker=dict(color=SERIES_1, cornerradius=4),
                           text=[f"₹{v:,.0f}" for v in s.values], textposition="outside",
                           textfont=dict(color=INK_2), cliponaxis=False,
                           hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>"))
    fig = _layout(fig, f"Spending by category{' - ' + month if month else ''}", legend=False,
                  height=max(260, 36 * len(s) + 70))
    fig.update_xaxes(tickprefix="₹", separatethousands=True, showgrid=True,
                     range=[0, (s.max() if len(s) else 1) * 1.22])  # room for the value labels
    fig.update_yaxes(tickprefix="", showgrid=False)
    return fig


def monthly_trend(tx: pd.DataFrame) -> go.Figure:
    d = tx.copy()
    d["month"] = d["date"].str[:7]
    m = d.pivot_table(index="month", columns="type", values="amount", aggfunc="sum", fill_value=0)
    fig = go.Figure()
    if "Income" in m:
        fig.add_trace(go.Scatter(x=m.index, y=m["Income"], name="Income", mode="lines+markers",
                                 line=dict(color=SERIES_1, width=2), marker=dict(size=8),
                                 hovertemplate="%{x}<br>Income ₹%{y:,.0f}<extra></extra>"))
    if "Expense" in m:
        fig.add_trace(go.Scatter(x=m.index, y=m["Expense"], name="Expenses", mode="lines+markers",
                                 line=dict(color=SERIES_2, width=2), marker=dict(size=8),
                                 hovertemplate="%{x}<br>Expenses ₹%{y:,.0f}<extra></extra>"))
    fig = _layout(fig, "Monthly income and expenses")
    fig.update_layout(hovermode="x unified")
    return fig


def forecast_chart(forecast: dict) -> go.Figure:
    hist = pd.DataFrame(forecast.get("history", []))
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["month"], y=hist["actual"], name="Actual (all spending)",
                                 mode="lines+markers", line=dict(color=NEUTRAL, width=2, dash="dot"),
                                 marker=dict(size=8), hovertemplate="%{x}<br>All spending ₹%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=hist["month"], y=hist["regular"], name="Actual (regular spending)",
                                 mode="lines+markers", line=dict(color=SERIES_1, width=2), marker=dict(size=8),
                                 hovertemplate="%{x}<br>Regular ₹%{y:,.0f}<extra></extra>"))
        pred = hist.dropna(subset=["predicted"])
        if not pred.empty:
            fig.add_trace(go.Scatter(x=pred["month"], y=pred["predicted"], name="Predicted (backtest)",
                                     mode="lines+markers", line=dict(color=SERIES_2, width=2),
                                     marker=dict(size=8, symbol="diamond"),
                                     hovertemplate="%{x}<br>Predicted ₹%{y:,.0f}<extra></extra>"))
    if forecast.get("predicted") is not None:
        fig.add_trace(go.Scatter(x=[forecast["target_month"]], y=[forecast["predicted"]], name="Next-month forecast",
                                 mode="markers+text", marker=dict(size=14, color=SERIES_2, symbol="star",
                                                                  line=dict(color=SURFACE, width=2)),
                                 text=[f"₹{forecast['predicted']:,.0f}"], textposition="top center",
                                 textfont=dict(color=INK),
                                 hovertemplate="%{x}<br>Forecast ₹%{y:,.0f}<extra></extra>"))
    fig = _layout(fig, "Actual vs predicted monthly expenses", height=380)
    fig.update_layout(hovermode="x unified")
    return fig


def anomaly_scatter(scored: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    normal = scored[scored["is_anomaly"] == 0]
    flagged = scored[scored["is_anomaly"] == 1]
    fig.add_trace(go.Scatter(x=normal["date"], y=normal["amount"], mode="markers", name="Normal",
                             marker=dict(size=8, color=NEUTRAL, opacity=0.7),
                             customdata=normal[["description", "category"]],
                             hovertemplate="%{x}<br>%{customdata[0]}<br>%{customdata[1]} ₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=flagged["date"], y=flagged["amount"], mode="markers+text", name="Flagged as unusual",
                             marker=dict(size=13, color=CRITICAL, symbol="diamond", line=dict(color=SURFACE, width=2)),
                             text=[f"₹{v:,.0f}" for v in flagged["amount"]], textposition="top center",
                             textfont=dict(color=INK),
                             customdata=flagged[["description", "category", "anomaly_score"]],
                             hovertemplate="%{x}<br>%{customdata[0]}<br>%{customdata[1]} ₹%{y:,.0f}"
                                           "<br>score %{customdata[2]:.3f}<extra></extra>"))
    fig = _layout(fig, "Expenses over time (log scale)", height=380)
    fig.update_yaxes(type="log")
    return fig


def budget_comparison(revised_budget: list) -> go.Figure:
    b = pd.DataFrame(revised_budget)
    fig = go.Figure()
    if b.empty:
        return _layout(fig, "Recommended budget allocation")
    b = b.sort_values("planned")
    fig.add_trace(go.Bar(y=b["category"], x=b["planned"], name="Predicted (no change)", orientation="h",
                         marker=dict(color=NEUTRAL, cornerradius=4),
                         hovertemplate="%{y}<br>Predicted ₹%{x:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(y=b["category"], x=b["revised"], name="Recommended", orientation="h",
                         marker=dict(color=SERIES_1, cornerradius=4),
                         hovertemplate="%{y}<br>Recommended ₹%{x:,.0f}<extra></extra>"))
    fig = _layout(fig, "Recommended budget allocation for next month", height=max(300, 44 * len(b) + 80))
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_xaxes(tickprefix="₹", separatethousands=True)
    fig.update_yaxes(tickprefix="")
    return fig


def savings_gauge(expected: float, goal: float) -> go.Figure:
    top = max(goal * 1.5, expected * 1.1, 1)
    color = GOOD if expected >= goal - 1 else CRITICAL   # ₹1 tolerance for rounding
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=max(expected, 0), number=dict(prefix="₹", valueformat=",.0f", font=dict(color=INK)),
        gauge=dict(axis=dict(range=[0, top], tickprefix="₹", tickfont=dict(color=MUTED)),
                   bar=dict(color=color, thickness=0.35), bgcolor="#f0efec", borderwidth=0,
                   threshold=dict(line=dict(color=INK, width=3), thickness=0.8, value=goal))))
    fig.update_layout(height=230, margin=dict(l=48, r=48, t=24, b=8), paper_bgcolor=SURFACE,
                      font=dict(family=FONT))
    return fig


def confusion_heatmap(cm: dict) -> go.Figure:
    df = pd.DataFrame(cm)  # columns = predicted, index = true
    df = df.loc[df.columns]  # same order
    fig = go.Figure(go.Heatmap(z=df.values, x=df.columns, y=df.index, colorscale=[[i / 7, c] for i, c in
                                                                                  enumerate(SEQ_BLUE)],
                               text=df.values, texttemplate="%{text}", showscale=False,
                               xgap=2, ygap=2,
                               hovertemplate="true %{y}<br>predicted %{x}<br>%{z} rows<extra></extra>"))
    fig = _layout(fig, "Confusion matrix (test set, rows = true class)", height=440, legend=False)
    fig.update_yaxes(tickprefix="", autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig
