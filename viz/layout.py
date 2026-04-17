"""
layout.py — GridLayout: manages matplotlib gridspec parameters for TWSLT plots.
"""
import matplotlib.pyplot as plt


class GridLayout:
    """
    Manages the 2-panel grid for TWSLT overview + super cluster plots.

    Layout:
        - GRID_ROWS × GRID_COLS grid
        - Top panel: overview (spans all COLS, height_ratios[0] rows)
        - Bottom panels: N super clusters in row-major order

    Usage:
        layout = GridLayout()
        fig, axes = layout.createfigure()           # returns plt.figure + axes dict
        fig, axes = layout.create_subplots()        # returns plt.figure + 2D axes array
    """

    def __init__(self,
                 n_rows=7,
                 n_cols=5,
                 height_ratios=None,
                 width_ratio=1.0,
                 wspace=0.15,
                 hspace=0.15,
                 fig_width=20,
                 fig_height=28):
        """
        Args:
            n_rows: total grid rows (default 7)
            n_cols: total grid cols (default 5)
            height_ratios: list of length n_rows. Default [3,1,1,1,1,1,1]
            width_ratio: width ratio per column (default 1.0 each)
            wspace, hspace: subplot spacing
            fig_width, fig_height: figure dimensions in inches
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.height_ratios = height_ratios or [3, 1, 1, 1, 1, 1, 1]
        self.width_ratio = width_ratio
        self.wspace = wspace
        self.hspace = hspace
        self.fig_width = fig_width
        self.fig_height = fig_height

    @property
    def overview_row_span(self):
        """Row indices occupied by the overview panel."""
        return slice(0, self.height_ratios[0])

    @property
    def sc_row_start(self):
        """First row of the super cluster grid."""
        return sum(self.height_ratios[:1])  # actually first SC row index

    def create_figure(self):
        """
        Create figure and gridspec.

        Returns:
            fig: plt.Figure
            gs: matplotlib GridSpec
        """
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        gs = fig.add_gridspec(
            self.n_rows,
            self.n_cols,
            height_ratios=self.height_ratios,
            width_ratios=[1.0] * self.n_cols,
            wspace=self.wspace,
            hspace=self.hspace,
        )
        return fig, gs

    def create_subplots(self):
        """
        Create figure and axes using plt.subplots.

        Returns:
            fig, axes where axes[0] is overview (spans all cols),
            axes[1:] are SC axes in row-major order
        """
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        gs = fig.add_gridspec(
            self.n_rows,
            self.n_cols,
            height_ratios=self.height_ratios,
            width_ratios=[1.0] * self.n_cols,
            wspace=self.wspace,
            hspace=self.hspace,
        )

        axes = []
        # Overview spans all cols in first GRID_ROWS portion
        ov_gs = fig.add_gridspec(1, 1)  # placeholder — use gs directly
        ax_ov = fig.add_subplot(gs[:self.height_ratios[0], :])
        axes.append(ax_ov)

        # SC panels — rows are accumulated after overview
        sc_rows = self.n_rows - self.height_ratios[0]
        for r in range(sc_rows):
            for c in range(self.n_cols):
                ax = fig.add_subplot(gs[self.height_ratios[0] + r, c])
                axes.append(ax)

        axes_arr = fig.subplots(2, self.n_cols,
                                gridspec_kw={'height_ratios': [self.height_ratios[0], 1]})
        return fig, axes

    def sc_index_to_rc(self, sc_idx):
        """Convert SC index (0-based) to (row, col) within the SC grid."""
        sc_rows = self.n_rows - self.height_ratios[0]
        sc_row = sc_idx // self.n_cols
        sc_col = sc_idx % self.n_cols
        return sc_row, sc_col

    def get_sc_gs(self, gs, sc_idx):
        """Get the gridspec slice for a given SC index."""
        sc_row, sc_col = self.sc_index_to_rc(sc_idx)
        row_offset = self.height_ratios[0]
        return gs[row_offset + sc_row, sc_col]
