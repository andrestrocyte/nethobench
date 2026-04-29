import pandas as pd
from pathlib import Path

def load_df(path: Path):
    if str(path).endswith(".csv"):
        return pd.read_csv(path, index_col=None, header=0, sep=",")
    elif str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise Exception(f"Only parquet and csv files allowed")
    

def load_gt_and_preds(gt_dir: Path, inf_dir: Path):



    gt_df = load_df(gt_dir)
    inf_df = load_df(inf_dir)
    diff1 = set(list(gt_df.columns)).difference(set(list(inf_df.columns)))
    diff2 = set(list(inf_df.columns)).difference(set(list(gt_df.columns)))

    assert len(diff1) == 0 and len(diff2) == 0, f"Unequal cols found: in gt but not preds: {diff1}, in preds but not gt: {diff2}"

    return gt_df, inf_df
