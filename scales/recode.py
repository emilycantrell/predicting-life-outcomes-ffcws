# flake8: noqa

import re
import sys
from datetime import date
from inspect import signature

import numpy as np
import pandas as pd
import pyreadstat
import swifter
from scales.custom_scales import *
from scales.hetero_units import *
from scales.wave2 import *
from scales.wave3 import *
from scales.wave4 import *
from scales.wave5 import *
from scales.wave6 import *


def get_range_str(s):
    """Extract range part of the string

    Args:
        s (str): Input string

    Returns:
        str: Part of the string between square braces
    """
    try:
        return re.search(r"\[(.*)\]", s).group(1)
    except AttributeError:
        return None


def get_var_list(varstr):
    """Generate list of variables from a compressed representation

    Returns:
        list: List of variables
    """
    csplit = varstr.split(",")
    vars = []
    for token in csplit:
        # remove spaces around
        token = token.strip()
        r = get_range_str(token)
        if r is None:
            vars.append(token)
        else:
            prefix = token.split("[")[0]
            start, end = r.split("-")
            try:
                vars.extend(
                    ["%s%d" % (prefix, x) for x in range(int(start), int(end) + 1)]
                )
            except ValueError:
                vars.extend([prefix + chr(x) for x in range(ord(start), ord(end) + 1)])
    return vars


def reverse_func(start, end):
    """HOF that generates a function to reverse values in a range

    Args:
        start (int): Starting value
        end (int): Ending value
    """

    def f(x):
        """Returns the reverse encoded value

        Args:
            x (int): input

        Returns:
            int: reverse code
        """
        return start + end - x

    return f


def subtract_func(y=1):
    """HOF that returns a function to subtract a given amount

    Args:
        y (int, optional): amount to subtract. Defaults to 1.
    """

    def f(x):
        """Single argument subtract function

        Args:
            x (int): input

        Returns:
            int: Subtracted amount
        """
        return x - y

    return f


def recode_func(y=2):
    """HOF that returns a function which recodes a set value to zero

    Args:
        y (int, optional): Value that's set to zero.Defaults to 2.
    """

    def f(x):
        return 0 if x == y else x

    return f


def combine_funcs(funcs):
    """HOF that combines multiple functions. The functions are applied in the
    given order

    Args:
        funcs ([type]): [description]
    """

    def f(x):
        for subf in funcs:
            x = subf(x)
        return x

    return f


def range_func(range_start, range_end):
    """HOF that returns a function to check if a given value is in range

    Args:
        range_start (int): Starting value
        range_end (int): Ending value

    Returns:
        function: Function returns true if input is in range (inclusive)
    """
    if range_start == -1 and range_end == -1:

        def always_true(x):
            return True

        return always_true

    def inrange(x):
        return isinstance(x, list) or (x >= range_start and x <= range_end)

    return inrange


def apply_func(range_start, range_end, recode_NA, reverse, subtract, recode_2to0, op):
    """Generates a function that applies the scale according to instructions

    Args:
        range_start (int): Start value
        range_end (int): End value
        recode_NA (boolean): If true recode NA
        reverse (boolean): If true then reverse values
        subtract (boolean): If true then subtract 1
        recode_2to0 (boolean): If true then recode 2 to zero
        op (function): Function to apply after applying operators

    Returns:
        function: [description]
    """
    inrange = range_func(range_start, range_end)
    change_func = combine_funcs(
        [
            func
            for func, is_enabled in [
                (reverse_func(range_start, range_end), reverse),
                (subtract_func(1), subtract),
                (recode_func(), recode_2to0),
            ]
            if is_enabled
        ]
    )

    def f(x):
        changed_values = []
        for index, item in zip(x.index, x):
            if "_intermediate" in index:
                changed_values.append(item)
            else:
                changed_values.append(
                    np.nan if not inrange(item) else change_func(item)
                )
        scale_val = op(pd.Series(changed_values, index=x.index))
        return scale_val

    return f

def custom_scale(func_name, range_start, range_end, subdf, meta):
    """Applies custom scale calculation functions

    Args:
        func_name (str): Function name
        range_start (int): Starting value
        range_end (int): Ending value
        subdf (pd.DataFrame): Data for the variables used in scale calculation

    Returns:
        pd.Series: Computed scale
    """
    inrange = range_func(range_start, range_end)

    def custom_apply_func(func_name):
        def f(x):
            changed_values = x.apply(lambda x: np.nan if not inrange(x) else x)
            # NOTE : some custom functions require meta. If the number of arguments
            # required for the function is two then send meta as the second argument
            if len(signature(custom_func_map[func_name]).parameters) > 1:
                return custom_func_map[func_name](changed_values, meta)
            return custom_func_map[func_name](changed_values)

        return f

    return subdf.swifter.apply(custom_apply_func(func_name), axis=1)

op_map = {
    "mean": lambda x: x.mean(skipna=False),
    "mean_ignoreNA": lambda x: x.mean(skipna=True),
    "sum": lambda x: x.sum(skipna=False),
    "moduloandsum": lambda x: ((x + 1) % 3).sum(skipna=False),
    "moduloandsum_2": lambda x: (((x + 1) % 2)).sum(skipna=False),
    "moduloandmean": lambda x: ((x + 1) % 3).mean(skipna=False),
    "identity": lambda x: x,
    "identity_list": lambda x: list(x),
    "wtmean_4": lambda x: (x.sum(skipna=False) / 4.0 if x.min() >= 0 else np.nan),
    "wtmean_5": lambda x: (x.sum(skipna=False) / 5.0 if x.min() >= 0 else np.nan),
    "wtmean_8": lambda x: (x.sum(skipna=False) / 8.0 if x.min() >= 0 else np.nan),
    "wtmean_10": lambda x: (x.sum(skipna=False) / 10.0 if x.min() >= 0 else np.nan),
    "wtmean_11": lambda x: (x.sum(skipna=False) / 11.0 if x.min() >= 0 else np.nan),
    "wtmean_13": lambda x: (x.sum(skipna=False) / 13.0 if x.min() >= 0 else np.nan),
    "wtmean_15": lambda x: (x.sum(skipna=False) / 15.0 if x.min() >= 0 else np.nan),
    "mean_80pctNA": lambda x: (x.mean() if x.count() / x.size >= 0.8 else np.nan),
}

custom_func_map = {
    # custom hetero unit
    "unit_standardization": unit_standardization,
    "unit_standardization_per": unit_standardization_per,
    "month_year": month_year,
    # custom scales
    "CBCL_combo": CBCL_combo,
    "conflict_tactics": conflict_tactics,
    "list_mean": list_mean,
    "list_mean_80pct": list_mean_80pct,
    "food_insecurity": food_insecurity,
    "familyMentalHealth": familyMentalHealth,
    "reverseSubset_andsubOne_mean": reverseSubset_andsubOne_mean,
    "reverseSubset_andsubOne_sum": reverseSubset_andsubOne_sum,
    "HOME_homeInterior": Home_homeInterior,
    # wave 6
    "dep_liberal_6": dep_liberal_6,
    "dep_past_wave_liberal": dep_past_wave_liberal,
    "dep_past_wave_conservative": dep_past_wave_conservative,
    "mat_hardship_past_wave": mat_hardship_past_wave,
    "eviction_6": eviction_6,
    # wave 5
    "sdq": sdq,
    "sdq_externalizing": sdq_externalizing,
    "sdq_internalizing": sdq_internalizing,
    "dep_liberal_m5": dep_liberal_m5,
    "dep_liberal_f5": dep_liberal_f5,
    "dep_liberal_n5": dep_liberal_n5,
    "pubertal_p5": pubertal_p5,
    # wave 4
    "dep_liberal_m4": dep_liberal_m4,
    "dep_liberal_f4": dep_liberal_f4,
    "HOME_devStim4": HOME_devStim4,
    "HOME_accessToReading4": HOME_accessToReading4,
    "HOME_outingActivities4": HOME_outingActivities4,
    # wave 3
    "dep_liberal_m3": dep_liberal_m3,
    "dep_liberal_f3": dep_liberal_f3,
    "alcohol_m3": alcohol_m3,
    "alcohol_f3": alcohol_f3,
    "drug_m3": drug_m3,
    "drug_f3": drug_f3,
    "HOME_devStim3": HOME_devStim3,
    # wave 2
    "dep_liberal_m2": dep_liberal_m2,
    "dep_liberal_f2": dep_liberal_f2,
    "emotionality_m2": emotionality_m2,
    "emotionality_f2": emotionality_f2
    # wave 1
}


def recode(
    df,
    scale_name,
    range_str,
    original_values,
    op,
    recode_NA=False,
    reverse=False,
    subtract=False,
    recode_2to0=False,
    custom_func_name="",
    meta=None,
):
    """Main recode function

    Args:
        df (pd.DataFrame): raw data
        scale_name (str): scale name
        range_str (str): string representing range of variables that are used
        for scale construction
        original_values (str): Range of original values
        op (function): Function object
        recode_NA (bool, optional): Remove NAs from calculation
        reverse (bool, optional): Reverse code values
        subtract (bool, optional): Subtract1
        recode_2to0 (bool, optional): Reverse code 2 to 0
        custom_func_name (str, optional): Name of the custom function
    """
    vars = get_var_list(range_str)
    subdf = df[vars]
    subdf = subdf.apply(
        lambda x: x.apply(
            lambda x: float(x.split(" ")[0]) if isinstance(x, str) else x
        ),
        axis=1,
    )
    range_start, range_end = original_values.split(":")
    range_start = int(range_start)
    range_end = int(range_end)
    if custom_func_name != "":
        df[scale_name] = custom_scale(
            custom_func_name, range_start, range_end, subdf, meta
        )
    else:
        df[scale_name] = subdf.apply(
            apply_func(
                range_start, range_end, recode_NA, reverse, subtract, recode_2to0, op
            ),
            axis=1,
        )


def get_scale_data(inp_fpath, df, meta):
    # columns in the instructions table
    # raw_df = pd.read_csv(instructions_fpath)
    # parsable instructions
    inp_df = pd.read_csv(inp_fpath, comment="#")
    cols = inp_df[
        ~inp_df.metadata_new_name.str.contains("intermediate")
    ].metadata_new_name.tolist()
    intermediate_cols = inp_df[
        inp_df.metadata_new_name.str.contains("intermediate")
    ].metadata_new_name.tolist()
    inp_df.fillna(value="", inplace=True)
    inp_df["op"] = inp_df.op.apply(lambda x: op_map[x])
    # read data
    # constructed scales
    scale_data = pd.DataFrame()
    for index in range(len(inp_df)):
        row = inp_df.iloc[index]
        print(row)
        recode(
            df,
            row.metadata_new_name,
            row.varstr,
            row.original,
            recode_NA=row.recodeNA,
            reverse=row.reverse,
            subtract=row.subtract1,
            recode_2to0=row.recode2to0,
            op=row.op,
            custom_func_name=row.custom,
            meta=meta,
        )
        scale_data[row.metadata_new_name] = df[row.metadata_new_name]
    # generate metadata for the scale data
    tups = []
    for col in scale_data.columns:
        tups.append(
            (
                col,
                "scales",
                "scales",
                True if "dummy" in col else None,
                "Continuous",
                20,
            )
        )
    meta_columns = inp_df.columns[inp_df.columns.str.startswith("metadata")]
    meta_df = inp_df[meta_columns]
    meta_df.replace("", np.nan, inplace=True)
    meta_df.columns = meta_df.columns.str.replace("metadata_", "")
    return scale_data, meta_df, cols, intermediate_cols


def read_stata(fpath):
    df, meta = pyreadstat.read_dta(str(fpath), encoding="LATIN1")
    return df, meta


def main(inp_fpath, stata_fpath, prefix):
    """Entry point for calculating scales

    Args:
        inp_fpath (str): Formatted instructions file
        stata_fpath (str): Stata raw data file
    """
    df, meta = read_stata(stata_fpath)
    scale_data, _, cols = get_scale_data(inp_fpath, df, meta)
    out = scale_data[cols]
    out.rename(columns={"idnum": "idnum_FFCWS"}, inplace=True)
    # count number of nulls for each column
    num_na = out.isnull().sum().sort_values(ascending=False)
    today = date.today()
    out.to_csv(f"{prefix}_Pranay_%s.csv" % today.strftime("%Y-%m-%d"), index=None)
    num_na.to_csv("numna_%s.csv" % today.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    main(*sys.argv[1:])
