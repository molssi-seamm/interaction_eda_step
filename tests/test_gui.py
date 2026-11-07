# -*- coding: utf-8 -*-

"""Smoke test of the Tk dialog: create it and re-lay it out for every choice
that drives the layout. Skipped when no display is available."""

import pytest


@pytest.fixture()
def tk_node():
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    import Pmw
    import seamm

    Pmw.initialise(root)
    flowchart = seamm.Flowchart(namespace="org.molssi.seamm", directory=".")
    tk_flowchart = seamm.TkFlowchart(
        master=root, flowchart=flowchart, namespace="org.molssi.seamm.tk"
    )
    node = flowchart.create_node("Interaction EDA")
    flowchart.add_node(node)
    plugin = tk_flowchart.plugin_manager.get("Interaction EDA")
    tk_node = plugin.create_tk_node(
        tk_flowchart=tk_flowchart, node=node, canvas=tk_flowchart.canvas, x=100, y=100
    )
    yield tk_node
    root.destroy()


def test_dialog_layouts(tk_node):
    tk_node.create_dialog()
    tk_node.reset_dialog()
    # defaults: molecules, cluster and pairs -> no column, no cutoff
    assert tk_node["fragment column"].grid_info() == {}
    assert tk_node["triples cutoff"].grid_info() == {}
    assert tk_node["functional"].grid_info() != {}
    assert tk_node["low fp64 gpus"].grid_info() != {}
    assert tk_node["low fp64 gpus"].get() == "refuse"

    tk_node["fragments"].set("atom column")
    tk_node.reset_dialog()
    assert tk_node["fragment column"].grid_info() != {}
    tk_node["fragments"].set("molecules")
    tk_node.reset_dialog()
    assert tk_node["fragment column"].grid_info() == {}

    for calculations in ("cluster, pairs and triples", "cluster", "pairs"):
        tk_node["calculations"].set(calculations)
        tk_node.reset_dialog()
        shown = tk_node["triples cutoff"].grid_info() != {}
        assert shown == ("triples" in calculations)

    assert tk_node["vv10 in response"].grid_info() != {}
    tk_node["vv10"].set("off (whole functional)")
    tk_node.reset_dialog()
    assert tk_node["vv10 in response"].grid_info() == {}
    tk_node["vv10"].set("as in the functional")
    tk_node.reset_dialog()
    assert tk_node["vv10 in response"].grid_info() != {}


def test_dialog_psi4(tk_node):
    """Psi4 hides the GPU, VV10 and dispersion settings and offers no meta-GGAs."""
    tk_node.create_dialog()
    assert tk_node["functional"].get() == "wB97M-V"
    tk_node["backend"].set("Psi4")
    tk_node.reset_dialog()
    for key in ("vv10", "vv10 in response", "dispersion", "low fp64 gpus"):
        assert tk_node[key].grid_info() == {}, key
    # The default meta-GGA is replaced, and none are offered
    assert tk_node["functional"].get() == "wB97X-V"
    offered = list(tk_node["functional"].combobox.cget("values"))
    assert "wB97X-V" in offered and "wB97M-V" not in offered
    assert len(tk_node["ferf response"].combobox.cget("values")) == 2

    tk_node["backend"].set("GPU4PySCF")
    tk_node.reset_dialog()
    for key in ("vv10", "vv10 in response", "dispersion", "low fp64 gpus"):
        assert tk_node[key].grid_info() != {}, key
    assert "wB97M-V" in list(tk_node["functional"].combobox.cget("values"))
