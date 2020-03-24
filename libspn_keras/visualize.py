import functools
import itertools
import operator
from collections import OrderedDict
import numpy as np

from libspn_keras.layers import DenseSum, DenseProduct, NormalLeaf, RootSum, \
    ToRegions, PermuteAndPadScopes

from libspn_keras.region import RegionVariable, RegionNode, \
    region_graph_to_dense_spn
import colorlover as cl
import plotly.graph_objects as go


def assemble_dense_spn_figure(dense_spn, show_padding=True):
    color_palette = itertools.cycle(
        cl.scales['12']['qual']['Set3']
    )
    max_total_nodes = 0
    for layer in dense_spn.layers:
        if isinstance(layer, RootSum):
            num_scopes = 1
            num_nodes = 1
        else:
            num_scopes, _, _, num_nodes = layer.output_shape
        total_nodes = num_scopes * num_nodes
        max_total_nodes = max(total_nodes, max_total_nodes)

    node_yx = OrderedDict()
    edges_yx = []

    colors = []
    symbols = []
    scopes = OrderedDict()

    last_width = 0
    last_num_nodes = None
    prev_layer = None
    node_group_sizes = []
    names = []
    for layer_index, layer in enumerate(dense_spn.layers):

        if isinstance(layer, ToRegions):
            continue

        if isinstance(layer, RootSum):
            num_scopes = 1
            num_nodes = 1
        else:
            num_scopes, _, _, num_nodes = layer.output_shape
        current_width = num_scopes * num_nodes
        if current_width < last_width:
            if last_num_nodes == num_nodes * num_scopes:
                current_width = last_width
            else:
                current_width = (last_width + current_width) / 2 + 1
        xs = np.linspace(
            -current_width / 2 + 0.5, current_width / 2 - 0.5,
            num=num_scopes * num_nodes) if num_scopes * num_nodes != 1 else [0]
        for scope_index in range(num_scopes):
            ci = next(color_palette)
            for nj, xj in enumerate(xs[num_nodes * scope_index:num_nodes * (scope_index + 1)]):
                node_yx[(layer_index, scope_index, nj)] = (layer_index, xj)
                colors.append(ci)
                if isinstance(layer, (DenseSum, RootSum)):
                    symbols.append("circle-cross")
                elif isinstance(layer, DenseProduct):
                    symbols.append("circle-x")
                elif isinstance(layer, PermuteAndPadScopes):
                    if layer.permutations[0][scope_index] == -1:
                        symbols.append("asterisk")
                    else:
                        symbols.append("circle")
                else:
                    symbols.append("circle")

                if prev_layer is not None:
                    scope = set()
                    num_nodes_prev = prev_layer.output_shape[-1]
                    if isinstance(layer, (DenseSum, RootSum)):
                        edges_yx.extend([
                            ((layer_index, xj), node_yx[(layer_index - 1, scope_index, node_offset)])
                            for node_offset in range(num_nodes_prev)
                        ])
                        scope = scopes[(layer_index - 1, scope_index, 0)]
                    elif isinstance(layer, DenseProduct):
                        nk = nj
                        for factor in range(layer.num_factors):
                            child_key = (
                                layer_index - 1,
                                scope_index * layer.num_factors + factor,
                                nk % num_nodes_prev
                            )

                            if scopes[child_key] != set() or show_padding:
                                edges_yx.append(
                                    ((layer_index, xj), node_yx[child_key])
                                )
                            scope |= scopes[child_key]

                            nk //= num_nodes_prev

                    elif isinstance(layer, PermuteAndPadScopes):
                        prev_scope_index = layer.permutations.numpy()[0, scope_index]
                        if prev_scope_index != -1:
                            child_key = (layer_index - 1, prev_scope_index, nj % num_nodes)
                            edges_yx.append(
                                ((layer_index, xj), node_yx[child_key])
                            )
                            scope = {int(prev_scope_index)}

                        else:
                            scope = set()
                else:
                    scope = {scope_index}

                scopes[(layer_index, scope_index, nj)] = scope

            node_group_sizes.append(num_nodes)

            if scope == set():
                scope_suffix = '\emptyset'
            else:
                scope_suffix = '{' + ', '.join('X_{}'.format(s) for s in sorted(scope)) + '}'
            names.append('$\\text{' + layer.__class__.__name__ + ' }' + scope_suffix + '$')

        prev_layer = layer
        last_width = current_width
        last_num_nodes = num_scopes * num_nodes

    return node_yx, edges_yx, colors, symbols, scopes, node_group_sizes, names


def visualize_dense_spn(dense_spn, show_legend=False, show_padding=True, node_size=30):

    nodes, edges, colors, symbols, scopes, node_group_sizes, names = \
        assemble_dense_spn_figure(dense_spn, show_padding=show_padding)

    hovertext = ['{' + ', '.join(str(s) for s in sorted(scope)) + '}'
                 for scope in scopes.values()]

    y_n, x_n = zip(*nodes.values())

    # Build figure
    fig = go.Figure(
        layout=go.Layout(
            plot_bgcolor='rgb(255,255,255,0)',
            paper_bgcolor='rgb(255,255,255,0)',
            xaxis=go.layout.XAxis(ticks="", tickvals=[]),
            yaxis=go.layout.YAxis(ticks="", tickvals=[]),
            legend=dict(traceorder='reversed', font=dict(size=18))
        ))

    y_e, x_e = zip(*np.asarray(edges).transpose(0, 2, 1))

    x_e_flat = list(x_e[0]) + functools.reduce(operator.concat, [[None] + list(x_ei) for x_ei in x_e])
    y_e_flat = list(y_e[0]) + functools.reduce(operator.concat, [[None] + list(y_ei) for y_ei in y_e])

    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=x_e_flat,
            y=y_e_flat,
            showlegend=False,
            line=dict(color='gray')
        )
    )

    # Add scatter trace with medium sized markers
    offset = 0
    for group_size, name in zip(node_group_sizes, names):
        group_ind = slice(offset, offset + group_size)
        offset += group_size
        if not show_padding and all(s == 'asterisk' for s in symbols[group_ind]):
            continue
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=x_n[group_ind],
                y=y_n[group_ind],
                hovertext=hovertext[group_ind],
                hoverinfo='text',
                marker=dict(
                    color=colors[group_ind],
                    size=node_size,
                    symbol=symbols[group_ind],
                    line=dict(
                        width=2
                    )
                ),
                showlegend=show_legend,
                name=name
            )
        )
    return fig


if __name__ == "__main__":

    x0 = RegionVariable(index=0)
    x1 = RegionVariable(index=1)
    x2 = RegionVariable(index=2)
    x3 = RegionVariable(index=3)
    x4 = RegionVariable(index=4)

    x0_x1 = RegionNode([x0, x1])

    x2_x3_x4 = RegionNode([x2, x3, x4])
    # x2_x3_x4 = RegionGraphNode([x2_x3, x4])

    # root = RegionGraphNode([x0_x1, x2_x3_x4])
    root = RegionNode([x2, x0_x1])

    dense_spn = region_graph_to_dense_spn(root, leaf_node=NormalLeaf(num_components=2),
                                          num_sums_iterable=itertools.cycle([2]))

    import itertools

    nodes = [RegionVariable(i) for i in range(64)]

    while len(nodes) > 1:
        next_nodes = []
        for i in range(0, len(nodes), 2):
            next_nodes.append(RegionNode([nodes[i], nodes[i + 1]]))
        nodes = next_nodes

    leaf = NormalLeaf(num_components=2)
    dense_spn = region_graph_to_dense_spn(
        nodes[0],
        num_sums_iterable=itertools.cycle([2]),
        leaf_node=leaf,
        return_weighted_child_logits=False,
    )
    dense_spn.summary()

    visualize_dense_spn(dense_spn=dense_spn, node_size=12).show()

    dense_spn.summary()
