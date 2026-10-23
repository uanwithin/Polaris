"""Regenerate the bundled Rerun 0.40 layout; not needed to run the C++ demo."""

from pathlib import Path

import rerun.blueprint as rrb

layout = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            name="MPPC | point-cloud navigation",
            origin="world",
            visual_bounds=rrb.VisualBounds2D(x_range=[0, 18], y_range=[0, 12]),
        ),
        rrb.Vertical(
            rrb.TimeSeriesView(name="Control | m/s, rad/s", origin="control"),
            rrb.TimeSeriesView(name="Clearance | m", origin="safety"),
            rrb.TimeSeriesView(name="Solver | ms", contents="solver/time_ms"),
            rrb.Tabs(
                rrb.TextDocumentView(name="Legend", origin="legend"),
                rrb.TimeSeriesView(name="Objective", contents="solver/objective"),
                rrb.TextLogView(name="Solver status", origin="solver/status"),
            ),
            row_shares=[1, 1, 1, 1],
        ),
        column_shares=[0.72, 0.28],
    ),
    rrb.BlueprintPanel(state="collapsed"),
    rrb.SelectionPanel(state="collapsed"),
    rrb.TimePanel(timeline="sim_time", state="collapsed"),
    auto_layout=False,
    auto_views=False,
)

layout.save("polaris_mppc", Path(__file__).resolve().parents[1] / "resources/polaris.rbl")
