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
    RELATIVE_TIME = 'rel_time'
    RELATIVE_TIME_ALIGNED = 'rel_time_aligned'
    CORRELATION_LAG = 'corr_lag'
    MEAN_CORRELATION_COEFFICIENT = 'mean_corrcoef'
    TRIAL_ID = 'trial_id'
    MIN = 'min'
    MAX = 'max'
    MEAN = 'mean'
    STD = 'std'
    VAR = 'variance'
    RMS = 'rms'
    POWER = 'power'
    PEAK = 'peak'
    P2P = 'p2p'
    CRESTFACTOR = 'cfactor'
    MAX_FOURIER = "max_f"
    SUM_FOURIER = "sum_f"
    MEAN_FOURIER = "mean_f"
    VAR_FOURIER= "var_f"
    PEAK_FOURIER = "peak_f"
