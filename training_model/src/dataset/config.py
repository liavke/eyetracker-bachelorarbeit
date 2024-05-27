import pandas

class Data():
    def __init__(self, table, columns) -> None:
        self.table :pandas.DataFrame = table
        self.col: list[str] = table.columns