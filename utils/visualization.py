from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str | Sequence[str],
    title: str,
    labels: dict[str, str] | None = None,
    color_sequence: list[str] | None = None,
    barmode: str = "relative",
) -> go.Figure:
    """Create a Plotly Express bar chart.

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    x, y : str | Sequence[str]
        Column name(s) for x-axis and y-axis.
    title : str
        Chart title.
    labels : dict | None
        Axis label overrides passed to ``px.bar``.
    color_sequence : list[str] | None
        Custom colour palette.
    barmode : str
        One of ``"relative"``, ``"group"``, ``"overlay"``, ``"stack"``.

    Returns
    -------
    go.Figure
    """
    kwargs: dict = dict(x=x, y=y, title=title, barmode=barmode)
    if labels is not None:
        kwargs["labels"] = labels
    if color_sequence is not None:
        kwargs["color_discrete_sequence"] = color_sequence
    return px.bar(data, **kwargs)


def create_pie_chart(
    data: pd.DataFrame,
    values: str,
    names: str,
    title: str,
    hole: float = 0.0,
) -> go.Figure:
    """Create a Plotly Express pie / donut chart.

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    values : str
        Column used for slice sizes.
    names : str
        Column used for slice labels.
    title : str
        Chart title.
    hole : float
        Size of the donut hole (0 = full pie).

    Returns
    -------
    go.Figure
    """
    return px.pie(data, values=values, names=names, title=title, hole=hole)


def create_scatter_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    labels: dict[str, str] | None = None,
    trendline: str | None = None,
) -> go.Figure:
    """Create a Plotly Express scatter chart.

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    x, y : str
        Column names.
    title : str
        Chart title.
    labels : dict | None
        Axis label overrides.
    trendline : str | None
        Trendline method (e.g. ``"ols"``).

    Returns
    -------
    go.Figure
    """
    kwargs: dict = dict(x=x, y=y, title=title)
    if labels is not None:
        kwargs["labels"] = labels
    if trendline is not None:
        kwargs["trendline"] = trendline
    return px.scatter(data, **kwargs)
