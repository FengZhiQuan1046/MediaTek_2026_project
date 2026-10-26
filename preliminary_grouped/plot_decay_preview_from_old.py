"""Make layout previews with new decay labels and explicitly recorded old curves.

These PDFs are draft placeholders. The normal decay plotters replace them with
measured 0.5/0.7/1.0 curves when the pending experiments finish.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import plot_amazon_decay_grid as amazon
import plot_additional_decay_grid as additional

HERE = Path(__file__).resolve().parent
MAPPING = (("030", 0.3, 0.5), ("010", 0.1, 0.7), ("001", 0.01, 1.0))


def render(kind, module):
    archive = HERE / "outputs" / f"{kind}_decay_grid_030_010_001" / "plotted_data.json"
    output = HERE / "outputs" / f"{kind}_decay_grid"
    old = json.loads(archive.read_text())
    assert {row["decay"] for row in old["sources"]} == {0.3, 0.1, 0.01}

    if kind == "amazon":
        module.DECAYS = tuple((tag, display, color, marker, line)
                              for (tag, _, display), (_, _, color, marker, line)
                              in zip(MAPPING, module.DECAYS))
        arguments = argparse.Namespace(font_path=None, output=output,
                                       min_users=100, max_x=20)
    else:
        module.DECAYS = tuple((tag, display) for tag, _, display in MAPPING)
        arguments = argparse.Namespace(font_path=None, output=output,
                                       min_users=100, max_x=20)

    module.plot(old["statistics"], arguments)
    preview = {
        "figure_status": "draft_proxy_not_measured_at_displayed_decay_values",
        "display_to_source_decay": [
            {"display_decay": display, "measured_source_decay": measured,
             "source_tag": tag} for tag, measured, display in MAPPING
        ],
        "source_plotted_data": str(archive),
        "sources": old["sources"],
        "statistics": old["statistics"],
    }
    (output / "plotted_data.json").write_text(json.dumps(preview, indent=2) + "\n")
    print("DRAFT proxy figure:", output / f"{kind}_decay_item_period_grid.pdf", flush=True)


if __name__ == "__main__":
    render("amazon", amazon)
    render("additional", additional)
