#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:09:00 2021

@author: zt
"""
import numpy as np
import pandas as pd

# Idea for future implementation re depression scales:
# 1. Create a general function a la food_insecurity with a "generic_index"
# 2. Making sure vars are entered in the parsable file in the same order (they should be),
# create an intermediate scale and use op identity_list
# 3. Pull the identity_list into the general function and match variables to the generic_index
# (Would need to create a separate function to handle since past wave, but all liberal past years are the same)


def eviction_6(x):
    """Calculates eviction scale (used in FFC)
    Args:
        x (pd.Series): Values for the variables involved in scale calculation
    Returns:
        int: Binary eviction value
    """
    if x.p6j40 == 1 or x.p6j51 == 1:
        return 1
    elif x.p6j40 == 2 and x.p6j51 == 2:
        return 0
    else:
        return np.nan


def mat_hardship_past_wave(x):
    """Calculates material hardship for past wave
    Instructions
     1 if p6j37==1 or p6j48==1,
       0 if p6j37==2 and p6j48==2,
       NA otherwise
    Args:
        x (pd.Series): Values for the variables involved
    Returns:
        float: Computed scale value
    """

    vals = []

    for a, b in zip(range(37, 48), range(48, 59)):
        var_a = "p6j%d" % a
        var_b = "p6j%d" % b
        if x[var_a] == 1 or x[var_b] == 1:
            val = 1
        elif x[var_a] == 2 and x[var_b] == 2:
            val = 0
        else:
            val = np.nan
        vals.append(val)

    return pd.Series(vals).mean(skipna=False)


def isin(x, var, vals):
    return (x[var] == vals).any()


def dep_liberal_6(x):
    total = np.nan

    dysphoric = (x.p6h7 == 1) and (x.p6h8 == 1 or x.p6h8 == 2 or x.p6h8 == 3) and (x.p6h9 == 1 or x.p6h9 == 2)
    anhedonia = (x.p6h11 == 1) and (x.p6h12 == 1 or x.p6h12 == 2 or x.p6h12 == 3) and (x.p6h13 == 1 or x.p6h13 == 2)

    not_dysphoric = (x.p6h7 == 2) or (x.p6h8 == 4) or (x.p6h9 == 3)
    not_anhedonia = (x.p6h11 == 2) or (x.p6h12 == 4) or (x.p6h13 == 3)

    if not_dysphoric or not_anhedonia:
        total = 0

    if dysphoric or anhedonia:
        total = 0
        bool_list = [x.p6h7 == 1, x.p6h10 == 1, x.p6h11 == 1, x.p6h14 == 1, x.p6h16 >= 10,
                     (x.p6h18 == 1 or x.p6h18 == 2), x.p6h19 == 1, x.p6h20 == 1, x.p6h21 == 1]
        for y in bool_list:
            if y:
                total += 1

    if x.p6h7 == -14 or x.p6h11 == -14 or x.p6h7 == 14 or x.p6h11 == 14:
        total = np.nan

    caseness = {
        0: 0.0001,
        1: 0.0568,
        2: 0.2352,
        3: 0.5542,
        4: 0.8125,
        5: 0.8895,
        6: 0.8895,
        7: 0.9083,
        8: 0.9083
    }
    return caseness.get(total, np.nan)

def dep_past_wave_liberal(x):
    scale = np.nan

    dysphoric = (x.p6h22 == 1) and (x.p6h23 == 1 or x.p6h23 == 2 or x.p6h23 == 3) and (x.p6h24 == 1 or x.p6h24 == 2)
    anhedonia = (x.p6h26 == 1) and (x.p6h27 == 1 or x.p6h27 == 2 or x.p6h27 == 3) and (x.p6h28 == 1 or x.p6h28 == 2)

    not_dysphoric = (x.p6h22 == 2) or (x.p6h23 == 4) or (x.p6h24 == 3)
    not_anhedonia = (x.p6h26 == 2) or (x.p6h27 == 4) or (x.p6h28 == 3)

    if not_dysphoric or not_anhedonia:
        scale = 0

    if dysphoric or anhedonia:
        scale = 0
        bool_list = [x.p6h22 == 1, x.p6h25 == 1, x.p6h26 == 1, x.p6h29 == 1, x.p6h31 >= 10,
                     (x.p6h33 == 1 or x.p6h33 == 2), x.p6h34 == 1, x.p6h35 == 1, x.p6h36 == 1]
        for y in bool_list:
            if y:
                scale += 1

    if x.p6h22 == -14 or x.p6h26 == -14 or x.p6h22 == 14 or x.p6h26 == 14:
        scale = np.nan

    caseness = {
        0: 0.0001,
        1: 0.0568,
        2: 0.2352,
        3: 0.5542,
        4: 0.8125,
        5: 0.8895,
        6: 0.8895,
        7: 0.9083,
        8: 0.9083
    }
    flipped_caseness = {value: key for key, value in caseness.items()}
    flipped_pastYear = flipped_caseness.get(x.p6_scale_CIDISFDepression_liberalPastYear,
                                            np.nan)  # retrive original total from past year
    if flipped_pastYear is None:
        flipped_pastYear = np.nan

    if np.isnan(flipped_pastYear) and scale >= 1: 
        total = caseness.get(scale, np.nan)
    elif flipped_pastYear >= 1 and np.isnan(scale):
        total = caseness.get(flipped_pastYear, np.nan)
    else:  # otherwise, can safely make things NA if one thing is NA
        total = caseness.get(scale + flipped_pastYear, np.nan)
    # print(flipped_pastYear, scale, total)
    return total


def dep_past_wave_conservative(x):
    stem = np.nan
    if (x.p6h7 == 1 and isin(x, "p6h8", [1, 2]) and isin(x, "p6h9", [1, 2])) or (
            x.p6h11 == 1 and isin(x, "p6h12", [1, 2]) and isin(x, "p6h13", [1, 2])
    ):
        stem = 1
    if (
            x.p6h22 == 2
            or x.p6h23 == 4
            or x.p6h23 == 3
            or x.p6h24 == 3
            or x.p6h26 == 2
            or x.p6h27 == 4
            or x.p6h27 == 3
            or x.p6h28 == 3
    ):
        stem = 0
    if (x.p6h22 == 1 and (isin(x, "p6h23", [1, 2]) and isin(x, "p6h24", [1, 2]))) or (
            x.p6h26 == 1 and (isin(x, "p6h27", [1, 2]) and isin(x, "p6h28", [1, 2]))
    ):
        stem = 1
    if isin(x, ["p6h7", "p6h11"], 14):
        stem = np.nan

    if np.isnan(stem):
        return np.nan
    weight_change = np.nan
    if isin(x, "p6h30", [1, 2, 3]) and x.p6h31 >= 10:
        weight_change = 1
    if isin(x, "p6h31", -3) or (0 < x.p6h31 < 10):
        weight_change = 0
    if x.p6h30 == -2:
        weight_change = np.nan

    scale = 0
    if stem == 1:
        scale = 0
        scale += (
                (x[["p6h29", "p6h34", "p6h35", "p6h36"]] == 1).sum()
                + (weight_change == 1)
                + isin(x, "p6h33", [1, 2])
                + isin(x, ["p6h25", "p6h26"], 1).any()
        )

    caseness = {
        0: 0.0001,
        1: 0.0568,
        2: 0.2352,
        3: 0.5542,
        4: 0.8125,
        5: 0.8895,
        6: 0.8895,
        7: 0.9083
    }

    flipped_caseness = {value: key for key, value in caseness.items()}
    flipped_pastYear = flipped_caseness.get(x.p6_scale_CIDISFDepression_conservativePastYear)
    total = scale + flipped_pastYear

    return caseness.get(total, np.nan)