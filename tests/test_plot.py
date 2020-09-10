import io

import matplotlib.pyplot as plt
import pytest

from pd_utils import plot_multi_axis
from tests.base import GeneratedTest

from tests.test_pandas import DataFrameTest


class TestPlot(DataFrameTest, GeneratedTest):
    generate = False
    plot_df = DataFrameTest.df_weight.copy()
    plot_df["third"] = [
        2.11,
        2.02,
        2.03,
        2.54,
        2.05,
        2.06,
        2.77,
        2.08,
        2.09,
        2.1,
        2.21,
        2.12,
    ]

    @pytest.mark.parametrize(
        "axis_locations, colored",
        [(False, False), (False, True), (True, False), (True, True),],
    )
    def test_multi_axis_plot(self, axis_locations: bool, colored: bool):
        file_name = f"multi_axis_plot"
        if axis_locations:
            file_name += "_axis_locations"
        if colored:
            file_name += "_colored"
        file_name += ".png"
        plt.figure()
        plot = plot_multi_axis(
            self.plot_df,
            cols=["RET", "weight", "third"],
            axis_locations_in_legend=axis_locations,
            colored_axes=colored,
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        content = buf.read()
        self.generate_or_check(content, file_name)
