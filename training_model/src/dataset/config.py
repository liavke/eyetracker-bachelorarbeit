import pandas

class Data():
    def __init__(self, table, columns) -> None:
        self.table :pandas.DataFrame = table
        self.col: list[str] = table.columns


class ColumnNames():
    DILATION_RIGHT = 'RPD'
    DILATION_LEFT = 'LPD'
    DILATION = 'dilation'

class Features:
    MIN = 'min'
    MAX = 'max'
    MEAN = 'mean'
    STD = 'std'
    VAR = 'variance'
