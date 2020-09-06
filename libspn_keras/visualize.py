from collections import OrderedDict
import functools
import itertools
import operator
from typing import Any, List

import colorlover as cl
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

from libspn_keras.layers import (
    DenseProduct,
    DenseSum,
    FlatToRegions,
    PermuteAndPadScopes,
    RootSum,
)


def visualize_dense_spn(
    dense_spn: tf.keras.Model,
    show_legend: bool = False,
    show_padding: bool = True,
    transparent: bool = False,
    node_size: int = 30,
) -> go.Figure:
    """
    Visualize dense SPN, consisting of ``DenseSum``, ``DenseProduct``, ``RootSum`` and leaf layers.

    Args:
        dense_spn: An SPN of type ``tensorflow.keras.Sequential``
        show_legend: Whether to show legend of scopes and layers on the right
        show_padding: Whether to show padded nodes
        transparent: If ``True``, the background is transparent.
        node_size: Size of the nodes drawn in the graph. Adjust to avoid clutter.

    Returns:
        A ``plotly.graph_objects.Figure`` instance. Use ``.show()`` to render the visualization.
    """
    (
        nodes,
        edges,
        colors,
        symbols,
        scopes,
        node_group_sizes,
        names,
    ) = _assemble_dense_spn_figure(dense_spn, show_padding=show_padding)

    hovertext = [
        "{" + ", ".join(str(s) for s in sorted(scope)) + "}"
        for scope in scopes.values()
    ]

    y_n, x_n = zip(*nodes.values())

    # Build figure
    fig = go.Figure(
        layout=go.Layout(
            plot_bgcolor="rgb(255,255,255,0)",
            xaxis=go.layout.XAxis(ticks="", tickvals=[]),
            yaxis=go.layout.YAxis(ticks="", tickvals=[]),
            legend=dict(traceorder="reversed", font=dict(size=18)),
        )
    )
    if transparent:
        fig.layout.paper_bgcolor = "rgb(255,255,255,0)"

    y_e, x_e = zip(*np.asarray(edges).transpose((0, 2, 1)))

    x_e_flat = list(x_e[0]) + functools.reduce(
        operator.concat, [[None] + list(x_ei) for x_ei in x_e]  # type: ignore
    )
    y_e_flat = list(y_e[0]) + functools.reduce(
        operator.concat, [[None] + list(y_ei) for y_ei in y_e]  # type: ignore
    )

    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=x_e_flat,
            y=y_e_flat,
            showlegend=False,
            line=dict(color="gray"),
        )
    )

    # Add scatter trace with medium sized markers
    offset = 0
    for group_size, name in zip(node_group_sizes, names):
        group_ind = slice(offset, offset + group_size)
        offset += group_size
        if not show_padding and all(s == "asterisk" for s in symbols[group_ind]):
            continue
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=x_n[group_ind],
                y=y_n[group_ind],
                hovertext=hovertext[group_ind],
                hoverinfo="text",
                marker=dict(
                    color=colors[group_ind],
                    size=node_size,
                    symbol=symbols[group_ind],
                    line=dict(width=2),
                ),
                showlegend=show_legend,
                name=name,
            )
        )
    return fig


def _assemble_dense_spn_figure(  # noqa: C901
    dense_spn: tf.keras.Model, show_padding: bool = True
):  # noqa: C901, ANN202
    color_palette = itertools.cycle(cl.scales["12"]["qual"]["Set3"])
    max_total_nodes = 0
    for layer in dense_spn.layers:
        if isinstance(layer, RootSum):
            num_scopes = 1
            num_nodes = 1
        else:
            _, num_scopes, _, num_nodes = layer.output_shape
        total_nodes = num_scopes * num_nodes
        max_total_nodes = max(total_nodes, max_total_nodes)

    node_yx = OrderedDict()
    edges_yx: List[Any] = []

    colors = []
    symbols = []
    scopes = OrderedDict()

    last_width = 0.0
    last_num_nodes = None
    prev_layer = None
    node_group_sizes = []
    names = []
    for layer_index, layer in enumerate(dense_spn.layers):

        if isinstance(layer, FlatToRegions):
            continue

        if isinstance(layer, RootSum):
            num_scopes = 1
            num_nodes = 1
        else:
            _, num_scopes, _, num_nodes = layer.output_shape
        current_width = float(num_scopes * num_nodes)
        if current_width < last_width:
            if last_num_nodes == num_nodes * num_scopes:
                current_width = last_width
            else:
                current_width = (last_width + current_width) / 2 + 1
        xs = (
            np.linspace(
                -current_width / 2 + 0.5,
                current_width / 2 - 0.5,
                num=num_scopes * num_nodes,
            )
            if num_scopes * num_nodes != 1
            else [0]
        )
        for scope_index in range(num_scopes):
            ci = next(color_palette)
            for nj, xj in enumerate(
                xs[num_nodes * scope_index : num_nodes * (scope_index + 1)]
            ):
                node_yx[(layer_index, scope_index, nj)] = (layer_index, xj)
                colors.append(ci)
                if isinstance(layer, (DenseSum, RootSum)):
                    symbols.append("circle-cross")
                elif isinstance(layer, DenseProduct):
                    symbols.append("circle-x")
                elif isinstance(layer, PermuteAndPadScopes):
                    if layer.permutations[0][scope_index] == -1:  # type: ignore
                        symbols.append("asterisk")
                    else:
                        symbols.append("circle")
                else:
                    symbols.append("circle")

                if prev_layer is not None:
                    scope = set()
                    num_nodes_prev = prev_layer.output_shape[-1]
                    if isinstance(layer, (DenseSum, RootSum)):
                        edges_yx.extend(
                            [
                                (
                                    (layer_index, xj),
                                    node_yx[
                                        (layer_index - 1, scope_index, node_offset)
                                    ],
                                )
                                for node_offset in range(num_nodes_prev)
                            ]
                        )
                        scope = scopes[(layer_index - 1, scope_index, 0)]
                    elif isinstance(layer, DenseProduct):
                        nk = nj
                        for factor in range(layer.num_factors):
                            child_key = (
                                layer_index - 1,
                                scope_index * layer.num_factors + factor,
                                nk % num_nodes_prev,
                            )

                            if scopes[child_key] != set() or show_padding:
                                edges_yx.append(((layer_index, xj), node_yx[child_key]))
                            scope |= scopes[child_key]

                            nk //= num_nodes_prev

                    elif isinstance(layer, PermuteAndPadScopes):
                        prev_scope_index = layer.permutations[0, scope_index]
                        if prev_scope_index != -1:
                            child_key = (
                                layer_index - 1,
                                prev_scope_index,
                                nj % num_nodes,
                            )
                            edges_yx.append(((layer_index, xj), node_yx[child_key]))
                            scope = {int(prev_scope_index)}

                        else:
                            scope = set()
                else:
                    scope = {scope_index}

                scopes[(layer_index, scope_index, nj)] = scope

            node_group_sizes.append(num_nodes)

            if scope == set():
                scope_suffix = r"\emptyset"
            else:
                scope_suffix = (
                    "{" + ", ".join("X_{}".format(s) for s in sorted(scope)) + "}"
                )
            names.append(
                "$\\text{" + layer.__class__.__name__ + " }" + scope_suffix + "$"
            )

        prev_layer = layer
        last_width = current_width
        last_num_nodes = num_scopes * num_nodes

    return node_yx, edges_yx, colors, symbols, scopes, node_group_sizes, names
