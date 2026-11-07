# -*- coding: utf-8 -*-

"""The graphical part of a Interaction EDA step"""

import pprint  # noqa: F401
import tkinter as tk  # noqa: F401
import tkinter.ttk as ttk

import interaction_eda_step  # noqa: F401
import seamm
from seamm_util import ureg, Q_, units_class  # noqa: F401
import seamm_widgets as sw

#: The settings of the method, in display order.
METHOD = (
    "backend",
    "functional",
    "basis",
    "density fitting",
    "vv10",
    "vv10 in response",
    "dispersion",
    "ferf response",
    "svd threshold",
)
#: The settings for the fragments and calculations.
FRAGMENTS = ("fragments", "fragment column", "calculations", "triples cutoff")


class TkInteractionEda(seamm.TkNode):
    """
    The graphical part of a Interaction EDA step in a flowchart.

    Attributes
    ----------
    tk_flowchart : TkFlowchart = None
        The flowchart that we belong to.
    node : Node = None
        The corresponding node of the non-graphical flowchart
    namespace : str
        The namespace of the current step.
    tk_subflowchart : TkFlowchart
        A graphical Flowchart representing a subflowchart
    canvas: tkCanvas = None
        The Tk Canvas to draw on
    dialog : Dialog
        The Pmw dialog object
    x : int = None
        The x-coordinate of the center of the picture of the node
    y : int = None
        The y-coordinate of the center of the picture of the node
    w : int = 200
        The width in pixels of the picture of the node
    h : int = 50
        The height in pixels of the picture of the node
    self[widget] : dict
        A dictionary of tk widgets built using the information
        contained in Interaction EDA_parameters.py

    See Also
    --------
    InteractionEda, TkInteractionEda,
    InteractionEdaParameters,
    """

    def __init__(
        self,
        tk_flowchart=None,
        node=None,
        canvas=None,
        x=None,
        y=None,
        w=200,
        h=50,
    ):
        """
        Initialize a graphical node.

        Parameters
        ----------
        tk_flowchart: Tk_Flowchart
            The graphical flowchart that we are in.
        node: Node
            The non-graphical node for this step.
        namespace: str
            The stevedore namespace for finding sub-nodes.
        canvas: Canvas
           The Tk canvas to draw on.
        x: float
            The x position of the nodes center on the canvas.
        y: float
            The y position of the nodes cetner on the canvas.
        w: float
            The nodes graphical width, in pixels.
        h: float
            The nodes graphical height, in pixels.

        Returns
        -------
        None
        """
        self.dialog = None

        super().__init__(
            tk_flowchart=tk_flowchart,
            node=node,
            canvas=canvas,
            x=x,
            y=y,
            w=w,
            h=h,
        )

    def create_dialog(self):
        """
        Create the dialog. A set of widgets will be chosen by default
        based on what is specified in the Interaction EDA_parameters
        module.

        Parameters
        ----------
        None

        Returns
        -------
        None

        See Also
        --------
        TkInteractionEda.reset_dialog
        """

        frame = super().create_dialog(title="Interaction EDA")
        P = self.node.parameters

        self["method frame"] = ttk.LabelFrame(
            frame, borderwidth=4, relief="sunken", text="Method", labelanchor="n"
        )
        self["fragments frame"] = ttk.LabelFrame(
            frame,
            borderwidth=4,
            relief="sunken",
            text="Fragments and calculations",
            labelanchor="n",
        )
        for key in METHOD:
            self[key] = P[key].widget(self["method frame"])
        for key in FRAGMENTS:
            self[key] = P[key].widget(self["fragments frame"])

        # Re-lay out when a choice changes which other settings apply
        for key in ("vv10", "fragments", "calculations"):
            self[key].combobox.bind("<<ComboboxSelected>>", self.reset_dialog)
            self[key].combobox.bind("<Return>", self.reset_dialog)
            self[key].combobox.bind("<FocusOut>", self.reset_dialog)

        self.reset_dialog()

    def reset_dialog(self, widget=None):
        """Lay out the widgets, showing only the settings that apply.

        The atom column applies only when the fragments come from an atom
        column, and the triples cutoff only when triples are computed.
        """
        frame = self["frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()
        for key in ("method frame", "fragments frame"):
            for slave in self[key].grid_slaves():
                slave.grid_forget()

        self["method frame"].grid(row=0, column=0, sticky=tk.EW, pady=5)
        self["fragments frame"].grid(row=1, column=0, sticky=tk.EW, pady=5)

        # VV10 in the response only matters when the functional keeps VV10
        method = [
            key
            for key in METHOD
            if not (key == "vv10 in response" and self["vv10"].get().startswith("off"))
        ]
        widgets = []
        for row, key in enumerate(method):
            self[key].grid(row=row, column=0, sticky=tk.EW)
            widgets.append(self[key])
        sw.align_labels(widgets, sticky=tk.E)

        fragments = self["fragments"].get()
        calculations = self["calculations"].get()
        keys = ["fragments"]
        if fragments == "atom column":
            keys.append("fragment column")
        keys.append("calculations")
        if "triples" in calculations:
            keys.append("triples cutoff")
        widgets = []
        for row, key in enumerate(keys):
            self[key].grid(row=row, column=0, sticky=tk.EW)
            widgets.append(self[key])
        sw.align_labels(widgets, sticky=tk.E)

        frame.columnconfigure(0, weight=1)

        have_results = (
            "results" in self.node.metadata and len(self.node.metadata["results"]) > 0
        )
        if have_results and "results" in self.node.parameters:
            self.setup_results()

    def right_click(self, event):
        """
        Handles the right click event on the node.

        Parameters
        ----------
        event : Tk Event

        Returns
        -------
        None

        See Also
        --------
        TkInteractionEda.edit
        """

        super().right_click(event)
        self.popup_menu.add_command(label="Edit..", command=self.edit)

        self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
