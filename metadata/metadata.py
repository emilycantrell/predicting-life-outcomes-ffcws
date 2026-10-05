#!/usr/bin/env python3

"""
Handle all metadata operations
"""

from exceptions import raise_metadata_edit_error

from utils import get_logger, read_csv

logger = get_logger(__name__)


class Metadata:
    def __init__(self):
        self.flag = False
        self.id_field = "new_name"

    def load(self, fpath):
        if not self.flag:
            self.df = read_csv(fpath, sep=",")
            if self.df.index.name != self.id_field:
                self.df.set_index(self.id_field, inplace=True)

    def edit(self, id_val, key, new_val):
        logger.info(
            f"Editing metadata for varname={id_val} key={key}. Current value {self.df.at[id_val, key]} to {new_val}"
        )
        self.df.at[id_val, key] = new_val

    def get_month_vars(self):
        """get month variables"""
        month_codes = (
            "1-Jan|2-Feb|3-Mar|4-Apr|5-May|6-Jun|7-Jul|8-Aug|9-Sep|10-Oct|11-Nov|12-Dec"
        )
        mm = self.df.filter(regex="label*", axis=1)
        has_month_codes = mm.stack().str.contains(month_codes).any(level=0)
        month_vars = self.df.loc[has_month_codes[has_month_codes].index].index
        return month_vars

    def append(self, new_df, axis=0):
        """Append metadata for new variables from a dataframe"""
        logger.info("Appending to metadata")
        if axis not in [0, 1]:
            raise ValueError(f"Metadata append not supported for axis = {axis}")
        if axis == 0:  # add new rows
            if self.id_field not in new_df:
                msg = f"Cannot append to metadata. ID field {self.id_field} not present in new dataframe"
                logger.error(msg)
                raise_metadata_edit_error(msg)
            self.df = self.df.append(new_df.set_index(self.id_field))
        else:
            raise NotImplementedError("column based append is not yet supported")

    def get_snapshot(self):
        return self.df.copy(deep=True)

    def get_shape(self):
        return self.df.shape


_meta = Metadata()
load = _meta.load
edit = _meta.edit
append = _meta.append
get_snapshot = _meta.get_snapshot
get_month_vars = _meta.get_month_vars
get_shape = _meta.get_shape
