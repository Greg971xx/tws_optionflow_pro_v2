# core/plotly_template.py
import plotly.graph_objects as go

DARK_BG = "#1e1e1e"
GRID = "#444"
FG = "#e0e0e0"

def apply_standard_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=FG),
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
        margin=dict(l=50, r=20, t=50, b=40),
        hovermode="x unified",
        legend=dict(bgcolor=DARK_BG)
    )
    return fig
