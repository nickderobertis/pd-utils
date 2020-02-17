import pandas as pd
from pandas.util.testing import assert_frame_equal
from pandas import Timestamp
from numpy import nan
import numpy

import pd_utils
import pd_utils.cum
import pd_utils.datetime_utils
import pd_utils.filldata
import pd_utils.merge
import pd_utils.port
import pd_utils.transform


class DataFrameTest:

    df = pd.DataFrame(
        [
            (10516, "a", "1/1/2000", 1.01),
            (10516, "a", "1/2/2000", 1.02),
            (10516, "a", "1/3/2000", 1.03),
            (10516, "a", "1/4/2000", 1.04),
            (10516, "b", "1/1/2000", 1.05),
            (10516, "b", "1/2/2000", 1.06),
            (10516, "b", "1/3/2000", 1.07),
            (10516, "b", "1/4/2000", 1.08),
            (10517, "a", "1/1/2000", 1.09),
            (10517, "a", "1/2/2000", 1.10),
            (10517, "a", "1/3/2000", 1.11),
            (10517, "a", "1/4/2000", 1.12),
        ],
        columns=["PERMNO", "byvar", "Date", "RET"],
    )

    df_duplicate_row = pd.DataFrame(
        [
            (10516, "a", "1/1/2000", 1.01),
            (10516, "a", "1/2/2000", 1.02),
            (10516, "a", "1/3/2000", 1.03),
            (10516, "a", "1/3/2000", 1.03),  # this is a duplicated row
            (10516, "a", "1/4/2000", 1.04),
            (10516, "b", "1/1/2000", 1.05),
            (10516, "b", "1/2/2000", 1.06),
            (10516, "b", "1/3/2000", 1.07),
            (10516, "b", "1/4/2000", 1.08),
            (10517, "a", "1/1/2000", 1.09),
            (10517, "a", "1/2/2000", 1.10),
            (10517, "a", "1/3/2000", 1.11),
            (10517, "a", "1/4/2000", 1.12),
        ],
        columns=["PERMNO", "byvar", "Date", "RET"],
    )

    df_weight = pd.DataFrame(
        data=[
            (10516, "a", "1/1/2000", 1.01, 0),
            (10516, "a", "1/2/2000", 1.02, 1),
            (10516, "a", "1/3/2000", 1.03, 1),
            (10516, "a", "1/4/2000", 1.04, 0),
            (10516, "b", "1/1/2000", 1.05, 1),
            (10516, "b", "1/2/2000", 1.06, 1),
            (10516, "b", "1/3/2000", 1.07, 1),
            (10516, "b", "1/4/2000", 1.08, 1),
            (10517, "a", "1/1/2000", 1.09, 0),
            (10517, "a", "1/2/2000", 1.1, 0),
            (10517, "a", "1/3/2000", 1.11, 0),
            (10517, "a", "1/4/2000", 1.12, 1),
        ],
        columns=["PERMNO", "byvar", "Date", "RET", "weight"],
    )

    df_nan_byvar = pd.DataFrame(
        data=[("a", 1), (nan, 2), ("b", 3), ("b", 4),], columns=["byvar", "val"]
    )

    df_nan_byvar_and_val = pd.DataFrame(
        data=[("a", 1), (nan, 2), ("b", nan), ("b", 4),], columns=["byvar", "val"]
    )

    single_ticker_df = pd.DataFrame(
        data=[("a", Timestamp("2000-01-01 00:00:00"), "ADM"),],
        columns=["byvar", "Date", "TICKER"],
    )

    df_datetime = df.copy()
    df_datetime["Date"] = pd.to_datetime(df_datetime["Date"])

    df_datetime_no_ret = df_datetime.copy()
    df_datetime_no_ret.drop("RET", axis=1, inplace=True)

    df_gvkey_str = pd.DataFrame(
        [
            ("001076", "3/1/1995"),
            ("001076", "4/1/1995"),
            ("001722", "1/1/2012"),
            ("001722", "7/1/2012"),
            ("001722", nan),
            (nan, "1/1/2012"),
        ],
        columns=["GVKEY", "Date"],
    )

    df_gvkey_str["Date"] = pd.to_datetime(df_gvkey_str["Date"])
    df_gvkey_num = df_gvkey_str.copy()
    df_gvkey_num["GVKEY"] = df_gvkey_num["GVKEY"].astype("float64")

    df_gvkey_str2 = pd.DataFrame(
        [
            ("001076", "2/1/1995"),
            ("001076", "3/2/1995"),
            ("001722", "11/1/2011"),
            ("001722", "10/1/2011"),
            ("001722", nan),
            (nan, "1/1/2012"),
        ],
        columns=["GVKEY", "Date"],
    )
    df_gvkey_str2["Date"] = pd.to_datetime(df_gvkey_str2["Date"])

    df_fill_data = pd.DataFrame(
        data=[
            (4, "c", nan, "a"),
            (1, "d", 3, "a"),
            (10, "e", 100, "a"),
            (2, nan, 6, "b"),
            (5, "f", 8, "b"),
            (11, "g", 150, "b"),
        ],
        columns=["y", "x1", "x2", "group"],
    )


class TestCumulate(DataFrameTest):

    expect_between_1_3 = pd.DataFrame(
        data=[
            (10516, "a", "1/1/2000", 1.01, 1.01),
            (10516, "a", "1/2/2000", 1.02, 1.02),
            (10516, "a", "1/3/2000", 1.03, 1.0506),
            (10516, "a", "1/4/2000", 1.04, 1.04),
            (10516, "b", "1/1/2000", 1.05, 1.05),
            (10516, "b", "1/2/2000", 1.06, 1.06),
            (10516, "b", "1/3/2000", 1.07, 1.1342),
            (10516, "b", "1/4/2000", 1.08, 1.08),
            (10517, "a", "1/1/2000", 1.09, 1.09),
            (10517, "a", "1/2/2000", 1.1, 1.1),
            (10517, "a", "1/3/2000", 1.11, 1.2210000000000003),
            (10517, "a", "1/4/2000", 1.12, 1.12),
        ],
        columns=["PERMNO", "byvar", "Date", "RET", "cum_RET"],
    )

    expect_first = pd.DataFrame(
        [
            (10516, "a", "1/1/2000", 1.01, 1.01),
            (10516, "a", "1/2/2000", 1.02, 1.02),
            (10516, "a", "1/3/2000", 1.03, 1.0506),
            (10516, "a", "1/4/2000", 1.04, 1.092624),
            (10516, "b", "1/1/2000", 1.05, 1.05),
            (10516, "b", "1/2/2000", 1.06, 1.06),
            (10516, "b", "1/3/2000", 1.07, 1.1342),
            (10516, "b", "1/4/2000", 1.08, 1.224936),
            (10517, "a", "1/1/2000", 1.09, 1.09),
            (10517, "a", "1/2/2000", 1.10, 1.10),
            (10517, "a", "1/3/2000", 1.11, 1.221),
            (10517, "a", "1/4/2000", 1.12, 1.36752),
        ],
        columns=["PERMNO", "byvar", "Date", "RET", "cum_RET"],
    )

    def test_method_between_1_3(self):
        cum_df = pd_utils.cumulate(
            self.df,
            "RET",
            "between",
            periodvar="Date",
            byvars=["PERMNO", "byvar"],
            time=[1, 3],
        )

        assert_frame_equal(self.expect_between_1_3, cum_df, check_dtype=False)

    def test_method_between_m2_0(self):
        cum_df = pd_utils.cumulate(
            self.df,
            "RET",
            "between",
            periodvar="Date",
            byvars=["PERMNO", "byvar"],
            time=[-2, 0],
        )

        # Actually same result as [1,3]
        assert_frame_equal(self.expect_between_1_3, cum_df, check_dtype=False)

    def test_shifted_index(self):
        df = self.df.copy()

        df.index = df.index + 10

        cum_df = pd_utils.cumulate(
            df,
            "RET",
            "between",
            periodvar="Date",
            byvars=["PERMNO", "byvar"],
            time=[-2, 0],
        )

        assert_frame_equal(self.expect_between_1_3, cum_df, check_dtype=False)

    def test_method_first(self):
        cum_df = pd_utils.cumulate(
            self.df, "RET", "first", periodvar="Date", byvars=["PERMNO", "byvar"]
        )

        assert_frame_equal(self.expect_first, cum_df, check_dtype=False)

    def test_grossify(self):
        df = self.df.copy()  # don't overwrite original
        df["RET"] -= 1  # ungrossify
        expect_first_grossify = self.expect_first.copy()
        expect_first_grossify["cum_RET"] -= 1
        expect_first_grossify["RET"] -= 1
        cum_df = pd_utils.cumulate(
            df,
            "RET",
            "first",
            periodvar="Date",
            byvars=["PERMNO", "byvar"],
            grossify=True,
        )

        assert_frame_equal(expect_first_grossify, cum_df, check_dtype=False)


class TestGroupbyMerge(DataFrameTest):
    def test_subset_max(self):
        byvars = ["PERMNO", "byvar"]
        out = pd_utils.groupby_merge(self.df, byvars, "max", subset="RET")
        expect_df = pd.DataFrame(
            [
                (10516, "a", "1/1/2000", 1.01, 1.04),
                (10516, "a", "1/2/2000", 1.02, 1.04),
                (10516, "a", "1/3/2000", 1.03, 1.04),
                (10516, "a", "1/4/2000", 1.04, 1.04),
                (10516, "b", "1/1/2000", 1.05, 1.08),
                (10516, "b", "1/2/2000", 1.06, 1.08),
                (10516, "b", "1/3/2000", 1.07, 1.08),
                (10516, "b", "1/4/2000", 1.08, 1.08),
                (10517, "a", "1/1/2000", 1.09, 1.12),
                (10517, "a", "1/2/2000", 1.10, 1.12),
                (10517, "a", "1/3/2000", 1.11, 1.12),
                (10517, "a", "1/4/2000", 1.12, 1.12),
            ],
            columns=["PERMNO", "byvar", "Date", "RET", "RET_max"],
        )

        assert_frame_equal(expect_df, out)

    def test_subset_std(self):
        byvars = ["PERMNO", "byvar"]
        out = pd_utils.merge.groupby_merge(self.df, byvars, "std", subset="RET")
        expect_df = pd.DataFrame(
            [
                (10516, "a", "1/1/2000", 1.01, 0.012909944487358068),
                (10516, "a", "1/2/2000", 1.02, 0.012909944487358068),
                (10516, "a", "1/3/2000", 1.03, 0.012909944487358068),
                (10516, "a", "1/4/2000", 1.04, 0.012909944487358068),
                (10516, "b", "1/1/2000", 1.05, 0.012909944487358068),
                (10516, "b", "1/2/2000", 1.06, 0.012909944487358068),
                (10516, "b", "1/3/2000", 1.07, 0.012909944487358068),
                (10516, "b", "1/4/2000", 1.08, 0.012909944487358068),
                (10517, "a", "1/1/2000", 1.09, 0.012909944487358068),
                (10517, "a", "1/2/2000", 1.10, 0.012909944487358068),
                (10517, "a", "1/3/2000", 1.11, 0.012909944487358068),
                (10517, "a", "1/4/2000", 1.12, 0.012909944487358068),
            ],
            columns=["PERMNO", "byvar", "Date", "RET", "RET_std"],
        )

        assert_frame_equal(expect_df, out)

    def test_nan_byvar_transform(self):
        expect_df = self.df_nan_byvar.copy()
        expect_df["val_transform"] = expect_df["val"]

        out = pd_utils.groupby_merge(
            self.df_nan_byvar, "byvar", "transform", (lambda x: x)
        )

        assert_frame_equal(expect_df, out)

    def test_nan_byvar_and_nan_val_transform_numeric(self):
        non_standard_index = self.df_nan_byvar_and_val.copy()
        non_standard_index.index = [5, 6, 7, 8]

        expect_df = self.df_nan_byvar_and_val.copy()
        expect_df["val_transform"] = expect_df["val"] + 1
        expect_df.index = [5, 6, 7, 8]

        out = pd_utils.groupby_merge(
            non_standard_index, "byvar", "transform", (lambda x: x + 1)
        )

        assert_frame_equal(expect_df, out)

    def test_nan_byvar_and_nan_val_and_nonstandard_index_transform_numeric(self):
        expect_df = self.df_nan_byvar_and_val.copy()
        expect_df["val_transform"] = expect_df["val"] + 1

    def test_nan_byvar_sum(self):
        expect_df = pd.DataFrame(
            data=[("a", 1, 1.0), (nan, 2, nan), ("b", 3, 7.0), ("b", 4, 7.0),],
            columns=["byvar", "val", "val_sum"],
        )

        out = pd_utils.groupby_merge(self.df_nan_byvar, "byvar", "sum")

        assert_frame_equal(expect_df, out)


class TestLongToWide:

    expect_df_with_colindex = pd.DataFrame(
        data=[
            (10516, "a", 1.01, 1.02, 1.03, 1.04),
            (10516, "b", 1.05, 1.06, 1.07, 1.08),
            (10517, "a", 1.09, 1.1, 1.11, 1.12),
        ],
        columns=[
            "PERMNO",
            "byvar",
            "RET1/1/2000",
            "RET1/2/2000",
            "RET1/3/2000",
            "RET1/4/2000",
        ],
    )

    expect_df_no_colindex = pd.DataFrame(
        data=[
            (10516, "a", "1/1/2000", 1.01, 1.02, 1.03, 1.04),
            (10516, "a", "1/2/2000", 1.01, 1.02, 1.03, 1.04),
            (10516, "a", "1/3/2000", 1.01, 1.02, 1.03, 1.04),
            (10516, "a", "1/4/2000", 1.01, 1.02, 1.03, 1.04),
            (10516, "b", "1/1/2000", 1.05, 1.06, 1.07, 1.08),
            (10516, "b", "1/2/2000", 1.05, 1.06, 1.07, 1.08),
            (10516, "b", "1/3/2000", 1.05, 1.06, 1.07, 1.08),
            (10516, "b", "1/4/2000", 1.05, 1.06, 1.07, 1.08),
            (10517, "a", "1/1/2000", 1.09, 1.1, 1.11, 1.12),
            (10517, "a", "1/2/2000", 1.09, 1.1, 1.11, 1.12),
            (10517, "a", "1/3/2000", 1.09, 1.1, 1.11, 1.12),
            (10517, "a", "1/4/2000", 1.09, 1.1, 1.11, 1.12),
        ],
        columns=["PERMNO", "byvar", "Date", "RET0", "RET1", "RET2", "RET3"],
    )
    input_data = DataFrameTest()

    ltw_no_dup_colindex = pd_utils.long_to_wide(
        input_data.df, ["PERMNO", "byvar"], "RET", colindex="Date"
    )
    ltw_dup_colindex = pd_utils.long_to_wide(
        input_data.df_duplicate_row, ["PERMNO", "byvar"], "RET", colindex="Date"
    )
    ltw_no_dup_no_colindex = pd_utils.long_to_wide(
        input_data.df, ["PERMNO", "byvar"], "RET"
    )
    ltw_dup_no_colindex = pd_utils.long_to_wide(
        input_data.df_duplicate_row, ["PERMNO", "byvar"], "RET"
    )
    df_list = [
        ltw_no_dup_colindex,
        ltw_dup_colindex,
        ltw_no_dup_no_colindex,
        ltw_dup_no_colindex,
    ]

    def test_no_duplicates_with_colindex(self):
        assert_frame_equal(self.expect_df_with_colindex, self.ltw_no_dup_colindex)

    def test_duplicates_with_colindex(self):
        assert_frame_equal(self.expect_df_with_colindex, self.ltw_dup_colindex)

    def test_no_duplicates_no_colindex(self):
        assert_frame_equal(self.expect_df_no_colindex, self.ltw_no_dup_no_colindex)

    def test_duplicates_no_colindex(self):
        assert_frame_equal(self.expect_df_no_colindex, self.ltw_dup_no_colindex)

    def test_no_extra_vars(self):
        for df in self.df_list:
            assert ("__idx__", "__key__") not in df.columns


class TestPortfolioAverages:

    input_data = DataFrameTest()

    expect_avgs_no_wt = pd.DataFrame(
        data=[
            (1, "a", 1.0250000000000001),
            (1, "b", 1.0550000000000002),
            (2, "a", 1.1050000000000002),
            (2, "b", 1.0750000000000002),
        ],
        columns=["portfolio", "byvar", "RET"],
    )

    expect_avgs_wt = pd.DataFrame(
        data=[
            (1, "a", 1.0250000000000001, 1.025),
            (1, "b", 1.0550000000000002, 1.0550000000000002),
            (2, "a", 1.1050000000000002, 1.12),
            (2, "b", 1.0750000000000002, 1.0750000000000002),
        ],
        columns=["portfolio", "byvar", "RET", "RET_wavg"],
    )

    expect_ports = pd.DataFrame(
        data=[
            (10516, "a", "1/1/2000", 1.01, 0, 1),
            (10516, "a", "1/2/2000", 1.02, 1, 1),
            (10516, "a", "1/3/2000", 1.03, 1, 1),
            (10516, "a", "1/4/2000", 1.04, 0, 1),
            (10516, "b", "1/1/2000", 1.05, 1, 1),
            (10516, "b", "1/2/2000", 1.06, 1, 1),
            (10516, "b", "1/3/2000", 1.07, 1, 2),
            (10516, "b", "1/4/2000", 1.08, 1, 2),
            (10517, "a", "1/1/2000", 1.09, 0, 2),
            (10517, "a", "1/2/2000", 1.1, 0, 2),
            (10517, "a", "1/3/2000", 1.11, 0, 2),
            (10517, "a", "1/4/2000", 1.12, 1, 2),
        ],
        columns=["PERMNO", "byvar", "Date", "RET", "weight", "portfolio"],
    )

    avgs, ports = pd_utils.portfolio_averages(
        input_data.df_weight, "RET", "RET", ngroups=2, byvars="byvar"
    )

    w_avgs, w_ports = pd_utils.portfolio_averages(
        input_data.df_weight, "RET", "RET", ngroups=2, byvars="byvar", wtvar="weight"
    )

    def test_simple_averages(self):
        assert_frame_equal(self.expect_avgs_no_wt, self.avgs, check_dtype=False)

    def test_weighted_averages(self):
        assert_frame_equal(self.expect_avgs_wt, self.w_avgs, check_dtype=False)

    def test_portfolio_construction(self):
        print(self.ports)
        assert_frame_equal(self.expect_ports, self.ports, check_dtype=False)
        assert_frame_equal(self.expect_ports, self.w_ports, check_dtype=False)


class TestWinsorize(DataFrameTest):
    def test_winsor_40_subset_byvars(self):

        expect_df = pd.DataFrame(
            data=[
                (10516, "a", "1/1/2000", 1.022624),
                (10516, "a", "1/2/2000", 1.022624),
                (10516, "a", "1/3/2000", 1.02672),
                (10516, "a", "1/4/2000", 1.02672),
                (10516, "b", "1/1/2000", 1.062624),
                (10516, "b", "1/2/2000", 1.062624),
                (10516, "b", "1/3/2000", 1.06672),
                (10516, "b", "1/4/2000", 1.06672),
                (10517, "a", "1/1/2000", 1.102624),
                (10517, "a", "1/2/2000", 1.102624),
                (10517, "a", "1/3/2000", 1.10672),
                (10517, "a", "1/4/2000", 1.10672),
            ],
            columns=["PERMNO", "byvar", "Date", "RET"],
        )

        wins = pd_utils.winsorize(
            self.df, 0.4, subset="RET", byvars=["PERMNO", "byvar"]
        )

        assert_frame_equal(expect_df, wins, check_less_precise=True)


class TestRegBy(DataFrameTest):
    def create_indf(self):
        indf = self.df_weight.copy()
        indf["key"] = indf["PERMNO"].astype(str) + "_" + indf["byvar"]
        return indf

    def test_regby_nocons(self):

        indf = self.create_indf()

        expect_df = pd.DataFrame(
            data=[
                (0.48774684748988806, "10516_a"),
                (0.9388636664168903, "10516_b"),
                (0.22929206076239614, "10517_a"),
            ],
            columns=["coef_RET", "key"],
        )

        rb = pd_utils.reg_by(indf, "weight", "RET", "key", cons=False)

        print("Reg by: ", rb)

        assert_frame_equal(expect_df, rb)

    def test_regby_cons(self):

        indf = self.create_indf()

        expect_df = pd.DataFrame(
            data=[
                (0.49999999999999645, 5.329070518200751e-15, "10516_a"),
                (0.9999999999999893, 1.0658141036401503e-14, "10516_b"),
                (-32.89999999999997, 29.999999999999982, "10517_a"),
            ],
            columns=["const", "coef_RET", "key"],
        )

        rb = pd_utils.reg_by(indf, "weight", "RET", "key")

        print("Reg by: ", rb)

        assert_frame_equal(expect_df, rb)

    def test_regby_cons_low_obs(self):

        indf = self.create_indf().loc[
            :8, :
        ]  # makes it so that one byvar only has one obs

        expect_df = pd.DataFrame(
            data=[
                (0.49999999999999645, 5.329070518200751e-15, "10516_a"),
                (0.9999999999999893, 1.0658141036401503e-14, "10516_b"),
                (nan, nan, "10517_a"),
            ],
            columns=["const", "coef_RET", "key"],
        )

        rb = pd_utils.reg_by(indf, "weight", "RET", "key")

        print("Reg by: ", rb)

        assert_frame_equal(expect_df, rb)


class TestExpandMonths(DataFrameTest):
    def test_expand_months_tradedays(self):

        expect_df = pd.DataFrame(
            data=[
                (
                    Timestamp("2000-01-03 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-04 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-05 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-06 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-07 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-10 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-11 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-12 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-13 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-14 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-18 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-19 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-20 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-21 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-24 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-25 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-26 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-27 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-28 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-31 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
            ],
            columns=["Daily Date", "byvar", "Date", "TICKER"],
        )

        em = pd_utils.expand_months(self.single_ticker_df)

        assert_frame_equal(expect_df.sort_index(axis=1), em.sort_index(axis=1))

    def test_expand_months_calendardays(self):

        expect_df = pd.DataFrame(
            data=[
                (
                    Timestamp("2000-01-01 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-02 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-03 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-04 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-05 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-06 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-07 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-08 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-09 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-10 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-11 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-12 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-13 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-14 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-15 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-16 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-17 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-18 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-19 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-20 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-21 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-22 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-23 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-24 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-25 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-26 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-27 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-28 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-29 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-30 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
                (
                    Timestamp("2000-01-31 00:00:00"),
                    "a",
                    Timestamp("2000-01-01 00:00:00"),
                    "ADM",
                ),
            ],
            columns=["Daily Date", "byvar", "Date", "TICKER"],
        )

        em = pd_utils.expand_months(self.single_ticker_df, trade_days=False)

        assert_frame_equal(expect_df.sort_index(axis=1), em.sort_index(axis=1))


class TestPortfolio(DataFrameTest):
    def test_portfolio_byvars(self):

        expect_df = pd.DataFrame(
            data=[
                (10516, "a", "1/1/2000", 1.01, 1),
                (10516, "a", "1/2/2000", 1.02, 1),
                (10516, "a", "1/3/2000", 1.03, 2),
                (10516, "a", "1/4/2000", 1.04, 2),
                (10516, "b", "1/1/2000", 1.05, 1),
                (10516, "b", "1/2/2000", 1.06, 1),
                (10516, "b", "1/3/2000", 1.07, 2),
                (10516, "b", "1/4/2000", 1.08, 2),
                (10517, "a", "1/1/2000", 1.09, 1),
                (10517, "a", "1/2/2000", 1.1, 1),
                (10517, "a", "1/3/2000", 1.11, 2),
                (10517, "a", "1/4/2000", 1.12, 2),
            ],
            columns=["PERMNO", "byvar", "Date", "RET", "portfolio"],
        )

        p = pd_utils.portfolio(self.df, "RET", ngroups=2, byvars=["PERMNO", "byvar"])

        assert_frame_equal(expect_df, p, check_dtype=False)

    def test_portfolio_with_nan_and_byvars(self):

        expect_df = pd.DataFrame(
            data=[
                (10516, "a", "1/1/2000", nan, 0),
                (10516, "a", "1/2/2000", 1.02, 1),
                (
                    10516,
                    "a",
                    "1/3/2000",
                    1.03,
                    1,
                ),  # changed from 2 to 1 when updated nan handling
                (10516, "a", "1/4/2000", 1.04, 2),
                (10516, "b", "1/1/2000", 1.05, 1),
                (10516, "b", "1/2/2000", 1.06, 1),
                (10516, "b", "1/3/2000", 1.07, 2),
                (10516, "b", "1/4/2000", 1.08, 2),
                (10517, "a", "1/1/2000", 1.09, 1),
                (10517, "a", "1/2/2000", 1.1, 1),
                (10517, "a", "1/3/2000", 1.11, 2),
                (10517, "a", "1/4/2000", 1.12, 2),
            ],
            columns=["PERMNO", "byvar", "Date", "RET", "portfolio"],
        )

        indf = self.df.copy()
        indf.loc[0, "RET"] = nan

        p = pd_utils.portfolio(indf, "RET", ngroups=2, byvars=["PERMNO", "byvar"])

        assert_frame_equal(expect_df, p, check_dtype=False)


class TestConvertSASDateToPandasDate:

    df_sasdate = pd.DataFrame(
        data=[
            ("011508", 16114.0),
            ("011508", 16482.0),
            ("011508", 17178.0),
            ("011508", 17197.0),
            ("011508", 17212.0),
        ],
        columns=["gvkey", "datadate"],
    )

    df_sasdate_nan = pd.DataFrame(
        data=[
            ("011508", 16114.0),
            ("011508", 16482.0),
            ("011508", 17178.0),
            ("011508", 17197.0),
            ("011508", nan),
            ("011508", 17212.0),
        ],
        columns=["gvkey", "datadate"],
    )

    def test_convert(self):

        expect_df = pd.DataFrame(
            data=[
                (numpy.datetime64("2004-02-13T00:00:00.000000000"),),
                (numpy.datetime64("2005-02-15T00:00:00.000000000"),),
                (numpy.datetime64("2007-01-12T00:00:00.000000000"),),
                (numpy.datetime64("2007-01-31T00:00:00.000000000"),),
                (numpy.datetime64("2007-02-15T00:00:00.000000000"),),
            ],
            columns=[0],
        )

        converted = pd.DataFrame(
            pd_utils.convert_sas_date_to_pandas_date(self.df_sasdate["datadate"])
        )

        assert_frame_equal(expect_df, converted)

    def test_convert_nan(self):

        expect_df = pd.DataFrame(
            data=[
                (numpy.datetime64("2004-02-13T00:00:00.000000000"),),
                (numpy.datetime64("2005-02-15T00:00:00.000000000"),),
                (numpy.datetime64("2007-01-12T00:00:00.000000000"),),
                (numpy.datetime64("2007-01-31T00:00:00.000000000"),),
                (numpy.datetime64("NaT"),),
                (numpy.datetime64("2007-02-15T00:00:00.000000000"),),
            ],
            columns=[0],
        )

        converted = pd.DataFrame(
            pd_utils.convert_sas_date_to_pandas_date(self.df_sasdate_nan["datadate"])
        )

        assert_frame_equal(expect_df, converted)


class TestMapWindows(DataFrameTest):

    times = [[-4, -2, 0], [-3, 1, 2], [4, 5, 6], [0, 1, 2], [-1, 0, 1]]

    df_period_str = pd.DataFrame(
        [
            (10516, "1/1/2000", 1.01),
            (10516, "1/2/2000", 1.02),
            (10516, "1/3/2000", 1.03),
            (10516, "1/4/2000", 1.04),
            (10516, "1/5/2000", 1.05),
            (10516, "1/6/2000", 1.06),
            (10516, "1/7/2000", 1.07),
            (10516, "1/8/2000", 1.08),
            (10517, "1/1/2000", 1.09),
            (10517, "1/2/2000", 1.10),
            (10517, "1/3/2000", 1.11),
            (10517, "1/4/2000", 1.12),
            (10517, "1/5/2000", 1.05),
            (10517, "1/6/2000", 1.06),
            (10517, "1/7/2000", 1.07),
            (10517, "1/8/2000", 1.08),
        ],
        columns=["PERMNO", "Date", "RET"],
    )

    df_period = df_period_str.copy()
    df_period["Date"] = pd.to_datetime(df_period["Date"])

    expect_dfs = [
        pd.DataFrame(
            data=[
                (10516, Timestamp("2000-01-01 00:00:00"), 1.01, 0),
                (10516, Timestamp("2000-01-02 00:00:00"), 1.02, 1),
                (10516, Timestamp("2000-01-03 00:00:00"), 1.03, 1),
                (10516, Timestamp("2000-01-04 00:00:00"), 1.04, 2),
                (10516, Timestamp("2000-01-05 00:00:00"), 1.05, 2),
                (10516, Timestamp("2000-01-06 00:00:00"), 1.06, 3),
                (10516, Timestamp("2000-01-07 00:00:00"), 1.07, 3),
                (10516, Timestamp("2000-01-08 00:00:00"), 1.08, 3),
                (10517, Timestamp("2000-01-01 00:00:00"), 1.09, 0),
                (10517, Timestamp("2000-01-02 00:00:00"), 1.1, 1),
                (10517, Timestamp("2000-01-03 00:00:00"), 1.11, 1),
                (10517, Timestamp("2000-01-04 00:00:00"), 1.12, 2),
                (10517, Timestamp("2000-01-05 00:00:00"), 1.05, 2),
                (10517, Timestamp("2000-01-06 00:00:00"), 1.06, 3),
                (10517, Timestamp("2000-01-07 00:00:00"), 1.07, 3),
                (10517, Timestamp("2000-01-08 00:00:00"), 1.08, 3),
            ],
            columns=["PERMNO", "Date", "RET", "__map_window__"],
        ),
        pd.DataFrame(
            data=[
                (10516, Timestamp("2000-01-01 00:00:00"), 1.01, 0),
                (10516, Timestamp("2000-01-02 00:00:00"), 1.02, 1),
                (10516, Timestamp("2000-01-03 00:00:00"), 1.03, 1),
                (10516, Timestamp("2000-01-04 00:00:00"), 1.04, 1),
                (10516, Timestamp("2000-01-05 00:00:00"), 1.05, 1),
                (10516, Timestamp("2000-01-06 00:00:00"), 1.06, 2),
                (10516, Timestamp("2000-01-07 00:00:00"), 1.07, 3),
                (10516, Timestamp("2000-01-08 00:00:00"), 1.08, 3),
                (10517, Timestamp("2000-01-01 00:00:00"), 1.09, 0),
                (10517, Timestamp("2000-01-02 00:00:00"), 1.1, 1),
                (10517, Timestamp("2000-01-03 00:00:00"), 1.11, 1),
                (10517, Timestamp("2000-01-04 00:00:00"), 1.12, 1),
                (10517, Timestamp("2000-01-05 00:00:00"), 1.05, 1),
                (10517, Timestamp("2000-01-06 00:00:00"), 1.06, 2),
                (10517, Timestamp("2000-01-07 00:00:00"), 1.07, 3),
                (10517, Timestamp("2000-01-08 00:00:00"), 1.08, 3),
            ],
            columns=["PERMNO", "Date", "RET", "__map_window__"],
        ),
        pd.DataFrame(
            data=[
                (10516, Timestamp("2000-01-01 00:00:00"), 1.01, 0),
                (10516, Timestamp("2000-01-02 00:00:00"), 1.02, 1),
                (10516, Timestamp("2000-01-03 00:00:00"), 1.03, 2),
                (10516, Timestamp("2000-01-04 00:00:00"), 1.04, 3),
                (10516, Timestamp("2000-01-05 00:00:00"), 1.05, 3),
                (10516, Timestamp("2000-01-06 00:00:00"), 1.06, 3),
                (10516, Timestamp("2000-01-07 00:00:00"), 1.07, 3),
                (10516, Timestamp("2000-01-08 00:00:00"), 1.08, 3),
                (10517, Timestamp("2000-01-01 00:00:00"), 1.09, 0),
                (10517, Timestamp("2000-01-02 00:00:00"), 1.1, 1),
                (10517, Timestamp("2000-01-03 00:00:00"), 1.11, 2),
                (10517, Timestamp("2000-01-04 00:00:00"), 1.12, 3),
                (10517, Timestamp("2000-01-05 00:00:00"), 1.05, 3),
                (10517, Timestamp("2000-01-06 00:00:00"), 1.06, 3),
                (10517, Timestamp("2000-01-07 00:00:00"), 1.07, 3),
                (10517, Timestamp("2000-01-08 00:00:00"), 1.08, 3),
            ],
            columns=["PERMNO", "Date", "RET", "__map_window__"],
        ),
        pd.DataFrame(
            data=[
                (10516, Timestamp("2000-01-01 00:00:00"), 1.01, 0),
                (10516, Timestamp("2000-01-02 00:00:00"), 1.02, 1),
                (10516, Timestamp("2000-01-03 00:00:00"), 1.03, 2),
                (10516, Timestamp("2000-01-04 00:00:00"), 1.04, 3),
                (10516, Timestamp("2000-01-05 00:00:00"), 1.05, 3),
                (10516, Timestamp("2000-01-06 00:00:00"), 1.06, 3),
                (10516, Timestamp("2000-01-07 00:00:00"), 1.07, 3),
                (10516, Timestamp("2000-01-08 00:00:00"), 1.08, 3),
                (10517, Timestamp("2000-01-01 00:00:00"), 1.09, 0),
                (10517, Timestamp("2000-01-02 00:00:00"), 1.1, 1),
                (10517, Timestamp("2000-01-03 00:00:00"), 1.11, 2),
                (10517, Timestamp("2000-01-04 00:00:00"), 1.12, 3),
                (10517, Timestamp("2000-01-05 00:00:00"), 1.05, 3),
                (10517, Timestamp("2000-01-06 00:00:00"), 1.06, 3),
                (10517, Timestamp("2000-01-07 00:00:00"), 1.07, 3),
                (10517, Timestamp("2000-01-08 00:00:00"), 1.08, 3),
            ],
            columns=["PERMNO", "Date", "RET", "__map_window__"],
        ),
        pd.DataFrame(
            data=[
                (10516, Timestamp("2000-01-01 00:00:00"), 1.01, 0),
                (10516, Timestamp("2000-01-02 00:00:00"), 1.02, 1),
                (10516, Timestamp("2000-01-03 00:00:00"), 1.03, 2),
                (10516, Timestamp("2000-01-04 00:00:00"), 1.04, 3),
                (10516, Timestamp("2000-01-05 00:00:00"), 1.05, 3),
                (10516, Timestamp("2000-01-06 00:00:00"), 1.06, 3),
                (10516, Timestamp("2000-01-07 00:00:00"), 1.07, 3),
                (10516, Timestamp("2000-01-08 00:00:00"), 1.08, 3),
                (10517, Timestamp("2000-01-01 00:00:00"), 1.09, 0),
                (10517, Timestamp("2000-01-02 00:00:00"), 1.1, 1),
                (10517, Timestamp("2000-01-03 00:00:00"), 1.11, 2),
                (10517, Timestamp("2000-01-04 00:00:00"), 1.12, 3),
                (10517, Timestamp("2000-01-05 00:00:00"), 1.05, 3),
                (10517, Timestamp("2000-01-06 00:00:00"), 1.06, 3),
                (10517, Timestamp("2000-01-07 00:00:00"), 1.07, 3),
                (10517, Timestamp("2000-01-08 00:00:00"), 1.08, 3),
            ],
            columns=["PERMNO", "Date", "RET", "__map_window__"],
        ),
    ]

    expect_df_first = pd.DataFrame(
        data=[
            (10516, Timestamp("2000-01-01 00:00:00"), 1.01, 0),
            (10516, Timestamp("2000-01-02 00:00:00"), 1.02, 1),
            (10516, Timestamp("2000-01-03 00:00:00"), 1.03, 1),
            (10516, Timestamp("2000-01-04 00:00:00"), 1.04, 1),
            (10516, Timestamp("2000-01-05 00:00:00"), 1.05, 1),
            (10516, Timestamp("2000-01-06 00:00:00"), 1.06, 1),
            (10516, Timestamp("2000-01-07 00:00:00"), 1.07, 1),
            (10516, Timestamp("2000-01-08 00:00:00"), 1.08, 1),
            (10517, Timestamp("2000-01-01 00:00:00"), 1.09, 0),
            (10517, Timestamp("2000-01-02 00:00:00"), 1.1, 1),
            (10517, Timestamp("2000-01-03 00:00:00"), 1.11, 1),
            (10517, Timestamp("2000-01-04 00:00:00"), 1.12, 1),
            (10517, Timestamp("2000-01-05 00:00:00"), 1.05, 1),
            (10517, Timestamp("2000-01-06 00:00:00"), 1.06, 1),
            (10517, Timestamp("2000-01-07 00:00:00"), 1.07, 1),
            (10517, Timestamp("2000-01-08 00:00:00"), 1.08, 1),
        ],
        columns=["PERMNO", "Date", "RET", "__map_window__"],
    )

    def run_for_each_time(func):
        """
        Decorator that can be applied to any function whose args are (self, time, expect_df) which runs the function
        for each time in self.times and picks the appropriate matching expect_df
        """

        def run(self):
            for t, time in enumerate(self.times):
                func(self, time, self.expect_dfs[t])

        return run

    def test_method_first(self):

        result = pd_utils.cum._map_windows(
            self.df_period,
            self.times[0],
            method="first",
            periodvar="Date",
            byvars=["PERMNO"],
        )

        assert_frame_equal(result, self.expect_df_first)

    @run_for_each_time
    def test_method_between(self, time, expect_df):

        result = pd_utils.cum._map_windows(
            self.df_period, time, method="between", periodvar="Date", byvars=["PERMNO"]
        )

        assert_frame_equal(result, expect_df)


class TestLeftMergeLatest(DataFrameTest):
    def test_left_merge_latest(self):
        expect_df = pd.DataFrame(
            data=[
                (
                    "001076",
                    Timestamp("1995-03-01 00:00:00"),
                    Timestamp("1995-02-01 00:00:00"),
                ),
                (
                    "001076",
                    Timestamp("1995-04-01 00:00:00"),
                    Timestamp("1995-03-02 00:00:00"),
                ),
                (
                    "001722",
                    Timestamp("2012-01-01 00:00:00"),
                    Timestamp("2011-11-01 00:00:00"),
                ),
                (
                    "001722",
                    Timestamp("2012-07-01 00:00:00"),
                    Timestamp("2011-11-01 00:00:00"),
                ),
                (
                    "001722",
                    numpy.timedelta64("NaT", "ns"),
                    numpy.timedelta64("NaT", "ns"),
                ),
                (
                    numpy.datetime64("NaT"),
                    numpy.datetime64("2012-01-01T00:00:00.000000000"),
                    numpy.datetime64("NaT"),
                ),
            ],
            columns=["GVKEY", "Date", "Date_y"],
        )

        lm = pd_utils.merge.left_merge_latest(
            self.df_gvkey_str, self.df_gvkey_str2, on="GVKEY"
        )
        lm_low_mem = pd_utils.merge.left_merge_latest(
            self.df_gvkey_str, self.df_gvkey_str2, on="GVKEY", low_memory=True
        )
        lm_sql = pd_utils.merge.left_merge_latest(
            self.df_gvkey_str, self.df_gvkey_str2, on="GVKEY", backend="sql"
        )

        assert_frame_equal(expect_df, lm, check_dtype=False)
        assert_frame_equal(expect_df.iloc[:-1], lm_low_mem, check_dtype=False)
        assert_frame_equal(expect_df, lm_sql, check_dtype=False)


class TestVarChangeByGroups(DataFrameTest):
    def test_multi_byvar_single_var(self):
        expect_df = pd.DataFrame(
            data=[
                (10516, "a", "1/1/2000", 1.01, nan),
                (10516, "a", "1/2/2000", 1.02, 0.010000000000000009),
                (10516, "a", "1/3/2000", 1.03, 0.010000000000000009),
                (10516, "a", "1/4/2000", 1.04, 0.010000000000000009),
                (10516, "b", "1/1/2000", 1.05, nan),
                (10516, "b", "1/2/2000", 1.06, 0.010000000000000009),
                (10516, "b", "1/3/2000", 1.07, 0.010000000000000009),
                (10516, "b", "1/4/2000", 1.08, 0.010000000000000009),
                (10517, "a", "1/1/2000", 1.09, nan),
                (10517, "a", "1/2/2000", 1.1, 0.010000000000000009),
                (10517, "a", "1/3/2000", 1.11, 0.010000000000000009),
                (10517, "a", "1/4/2000", 1.12, 0.010000000000000009),
            ],
            columns=["PERMNO", "byvar", "Date", "RET", "RET_change"],
        )

        vc = pd_utils.transform.var_change_by_groups(self.df, "RET", ["PERMNO", "byvar"])

        assert_frame_equal(expect_df, vc)

    def test_multi_byvar_multi_var(self):
        expect_df = pd.DataFrame(
            data=[
                (10516, "a", "1/1/2000", 1.01, 0, nan, nan),
                (10516, "a", "1/2/2000", 1.02, 1, 0.010000000000000009, 1.0),
                (10516, "a", "1/3/2000", 1.03, 1, 0.010000000000000009, 0.0),
                (10516, "a", "1/4/2000", 1.04, 0, 0.010000000000000009, -1.0),
                (10516, "b", "1/1/2000", 1.05, 1, nan, nan),
                (10516, "b", "1/2/2000", 1.06, 1, 0.010000000000000009, 0.0),
                (10516, "b", "1/3/2000", 1.07, 1, 0.010000000000000009, 0.0),
                (10516, "b", "1/4/2000", 1.08, 1, 0.010000000000000009, 0.0),
                (10517, "a", "1/1/2000", 1.09, 0, nan, nan),
                (10517, "a", "1/2/2000", 1.1, 0, 0.010000000000000009, 0.0),
                (10517, "a", "1/3/2000", 1.11, 0, 0.010000000000000009, 0.0),
                (10517, "a", "1/4/2000", 1.12, 1, 0.010000000000000009, 1.0),
            ],
            columns=[
                "PERMNO",
                "byvar",
                "Date",
                "RET",
                "weight",
                "RET_change",
                "weight_change",
            ],
        )

        vc = pd_utils.transform.var_change_by_groups(
            self.df_weight, ["RET", "weight"], ["PERMNO", "byvar"]
        )

        assert_frame_equal(expect_df, vc)


class TestFillExcludedRows(DataFrameTest):

    expect_df_nofill = pd.DataFrame(
        data=[
            ("001076", Timestamp("1995-03-01 00:00:00")),
            ("001076", Timestamp("1995-04-01 00:00:00")),
            ("001076", Timestamp("2012-01-01 00:00:00")),
            ("001076", Timestamp("2012-07-01 00:00:00")),
            ("001722", Timestamp("1995-03-01 00:00:00")),
            ("001722", Timestamp("1995-04-01 00:00:00")),
            ("001722", Timestamp("2012-01-01 00:00:00")),
            ("001722", Timestamp("2012-07-01 00:00:00")),
        ],
        columns=["GVKEY", "Date"],
    )

    def test_no_fillvars_str_byvars(self):
        result = pd_utils.filldata.fill_excluded_rows(self.df_gvkey_str, ["GVKEY", "Date"])
        assert_frame_equal(self.expect_df_nofill, result)

    def test_no_fillvars_series_byvars(self):
        result = pd_utils.filldata.fill_excluded_rows(
            self.df_gvkey_str, [self.df_gvkey_str["GVKEY"], "Date"]
        )
        assert_frame_equal(self.expect_df_nofill, result)

    def test_fillvars(self):
        var_df = self.df_gvkey_str.copy()
        var_df["var"] = 1

        expect_df = pd.DataFrame(
            data=[
                ("001076", Timestamp("1995-03-01 00:00:00"), 1.0),
                ("001076", Timestamp("1995-04-01 00:00:00"), 1.0),
                ("001076", Timestamp("2012-01-01 00:00:00"), 0.0),
                ("001076", Timestamp("2012-07-01 00:00:00"), 0.0),
                ("001722", Timestamp("1995-03-01 00:00:00"), 0.0),
                ("001722", Timestamp("1995-04-01 00:00:00"), 0.0),
                ("001722", Timestamp("2012-01-01 00:00:00"), 1.0),
                ("001722", Timestamp("2012-07-01 00:00:00"), 1.0),
            ],
            columns=["GVKEY", "Date", "var"],
        )

        result = pd_utils.filldata.fill_excluded_rows(var_df, ["GVKEY", "Date"], "var", value=0)
        assert_frame_equal(expect_df, result)


class TestFillnaByGroups(DataFrameTest):
    def test_fillna_by_group(self):
        expect_df = pd.DataFrame(
            data=[
                ("a", 4, "c", 51.5),
                ("a", 1, "d", 3.0),
                ("a", 10, "e", 100.0),
                ("b", 2, "f", 6.0),
                ("b", 5, "f", 8.0),
                ("b", 11, "g", 150.0),
            ],
            columns=["group", "y", "x1", "x2"],
        )

        result = pd_utils.fillna_by_groups(self.df_fill_data, "group")

        assert_frame_equal(expect_df, result)

    def test_fillna_by_group_keep_one(self):

        expect_df = pd.DataFrame(
            data=[("a", 4, "c", 51.5), ("b", 2, "f", 6.0),],
            columns=["group", "y", "x1", "x2"],
            index=[0, 3],
        )

        result = pd_utils.fillna_by_groups_and_keep_one_per_group(
            self.df_fill_data, "group"
        )

        assert_frame_equal(expect_df, result)

