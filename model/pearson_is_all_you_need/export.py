from model import export_notebook, RegexModifierFilter, InsertionFilter

export_notebook(__file__, files_to_concat=[
    'util.py',
    'visualization/metric.py',
    'pipeline/__init__.py',
    'pipeline/backtest.py',
    'pipeline/fundamental.py',
    'pipeline/parse_raw_df.py',
    'datatools.py',
    'model/neural_network.py',
], filters=[
    # General removal
    RegexModifierFilter(r'^from (pipeline|visualization|datatools|util|model).*', ''),
    RegexModifierFilter(r'^import (pipeline|visualization|datatools|util|model).*', ''),
    RegexModifierFilter(r'.*ERASE_MAGIC.*', ''),

    # QIDS related
    RegexModifierFilter(r'^from qids_lib import QIDS', 'from qids_package.qids import make_env'),
    RegexModifierFilter(r'qids: QIDS', 'qids'),
    RegexModifierFilter(r'QIDS\(path_prefix=qids_path_prefix\)', 'make_env()'),
    RegexModifierFilter(r'qids = QIDS.*', r'from qids_package import qids'),

    # Backtest
    RegexModifierFilter(r'isinstance\(model, SupportsPredict\)', 'not callable(model)'),
    InsertionFilter(['from typing import Any', 'ModelLike = Any', 'SupportsPredict = Any'], apply_when=0),
    RegexModifierFilter(r"f'{path_prefix}/{PARSED_PATH}'.*$", r"'/kaggle/working/parsed'"),
    RegexModifierFilter(r"f'{path_prefix}/{RAW_PATH}'.*$",
                        r"'/kaggle/input/hku-qids-2023-quantitative-investment-competition'"),

    # Dataset
    RegexModifierFilter(r"'../../data/nc'", r"'/kaggle/working/data/nc'"),
    RegexModifierFilter(r'n_days=-1\)', r'n_days=-1, path_prefix="/kaggle/working")'),
], filter_debug=False, clip=True)
