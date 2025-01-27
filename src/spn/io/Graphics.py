"""
Created on March 29, 2018

@author: Alejandro Molina
"""
from matplotlib.ticker import NullLocator
from networkx.drawing.nx_agraph import graphviz_layout

# import matplotlib
# matplotlib.use('Agg')
import logging

logger = logging.getLogger(__name__)


def _get_networkx_obj(spn):
    import networkx as nx
    from spn.structure.Base import Sum, Product, Leaf, Placeholder, get_nodes_by_type
    import numpy as np

    all_nodes = get_nodes_by_type(spn)
    logger.info(all_nodes)

    g = nx.Graph()

    labels = {}
    for n in all_nodes:

        if isinstance(n, Sum):
            label = "+"
        elif isinstance(n, Product):
            label = "x"
        elif isinstance(n, Placeholder):
            label = f"<P{n.placeholder_id}>"
        else:
            label = "V" + str(n.scope[0])
        if hasattr(n, "var_name"):
            label += f"\n(VN: {n.var_name}, VID: {n.var_id})"
        if hasattr(n, "node_id"):
            label += f"\n(ID {n.node_id})"
        g.add_node(n.id)
        labels[n.id] = label

        if isinstance(n, Leaf):
            continue
        for i, c in enumerate(n.children):
            edge_label = ""
            if isinstance(n, Sum):
                edge_label = np.round(n.weights[i], 2)
            g.add_edge(c.id, n.id, weight=edge_label)

    return g, labels


def draw_spn(spn, width=8, height=6):

    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout
    import matplotlib.pyplot as plt

    plt.clf()

    g, labels = _get_networkx_obj(spn)

    pos = graphviz_layout(g, prog="dot")
    ax = plt.gca()
    plt.gcf().set_size_inches(width, height)

    nx.draw(
        g,
        pos,
        with_labels=True,
        arrows=False,
        node_color="#DDDDDD",
        edge_color="#888888",
        width=1,
        node_size=700,
        labels=labels,
        font_size=9,
        node_shape="",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
    )
    ax.collections[0].set_edgecolor("#333333")
    edge_labels = nx.draw_networkx_edge_labels(
        g, pos=pos, edge_labels=nx.get_edge_attributes(g, "weight"), font_size=16, alpha=0.6
    )

    xpos = list(map(lambda p: p[0], pos.values()))
    ypos = list(map(lambda p: p[1], pos.values()))

    ax.set_xlim(min(xpos) - 20, max(xpos) + 20)
    ax.set_ylim(min(ypos) - 20, max(ypos) + 20)
    plt.tight_layout()
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    return plt


def plot_spn(spn, fname="plot.pdf", width=8, height=6):
    plt = draw_spn(spn, width=width, height=height)
    plt.savefig(fname, bbox_inches="tight", pad_inches=0)


def plot_spn2(spn, fname="plot.pdf"):
    import networkx as nx
    import matplotlib.pyplot as plt

    g, _ = _get_networkx_obj(spn)

    pos = graphviz_layout(g, prog="dot")
    nx.draw(g, pos, with_labels=False, arrows=False)
    plt.savefig(fname)


def plot_spn_to_svg(root_node, fname="plot.svg"):
    import networkx.drawing.nx_pydot as nxpd

    g, _ = _get_networkx_obj(root_node)

    pdG = nxpd.to_pydot(g)
    svg_string = pdG.create_svg()

    f = open(fname, "wb")
    f.write(svg_string)
