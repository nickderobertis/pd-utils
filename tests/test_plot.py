import matplotlib.pyplot as plt

from pd_utils import plot_multi_axis


from tests.test_pandas import DataFrameTest


class TestPlot(DataFrameTest):
    plot_df = DataFrameTest.df_weight.copy()
    plot_df['third'] = [
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

    def test_multi_axis_plot(self):
        plot = plot_multi_axis(self.plot_df, cols=['RET', 'weight', 'third'])
        plt.tight_layout()
        plt.show()
