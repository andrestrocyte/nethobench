import os
import pandas as pd

if __name__ == "__main__":
    for root, dirs, files in os.walk("data/predictions"):
        for file in sorted(list(files)):
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                data = pd.read_csv(path, index_col=None, header=0)
                data.query("itemPosition<1140").to_csv(path, index=None)
