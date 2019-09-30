import pandas as pd
import functools
import os
import sys
import warnings
from typing import Callable, Dict, List, Union
import numpy as np


def add_constant_index_level(df: pd.DataFrame, value: None, level_name=str):
    return pd.concat([df], keys=[value], names=[level_name])


def flatten_cols(df: pd.DataFrame, delim: str = ';'):
    new_cols = [delim.join((col_lev for col_lev in tup if col_lev)) for tup in df.columns.values]
    ndf = df.copy()
    ndf.columns = new_cols
    return ndf


def unflatten_cols(df: pd.DataFrame, delim: str =';'):
    new_cols = pd.MultiIndex.from_tuples([tuple(col.split(delim)) for col in df.columns])
    ndf = df.copy()
    ndf.columns = new_cols
    return ndf


def index_level_types(df):
    return [f"{df.index.names[i]}: {df.index.get_level_values(n).dtype}"
            for i, n in enumerate(df.index.names)]


def unique_counts(df):
    for i in df.columns:
        count = df[i].nunique()
        print(i, ": ", count)


def check(varname: str, value, expected_types: list):
        """
        one-liner for checking types
        usage:
        check('x', x, [int, float])
        :param varname: name of the variable
        :param value: value of the varname
        :param expected_types: types we expect the item to be
        returns: type error if data is not of the expected type
        """
        is_expected_type = False
        for t in expected_types:
                if isinstance(value, t):
                        is_expected_type = True
                        break
        if not is_expected_type:
                raise TypeError(
                    "{varname} should be one of {expected_types}".format(
                        varname=varname, expected_types=expected_types))
