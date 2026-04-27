import os

gt_file_dict = {
    "ba2": "data-clean-300-1140-ba2.csv",
    "ba4": "data-clean-300-1140-ba4.csv",
    "ba8": "data-clean-300-1140-ba8.csv",
    "ba16": "data-clean-300-1140-ba16.csv"
}

if __name__ == "__main__":
    for root, dirs, files in os.walk("data/predictions"):
        for i, file in enumerate(sorted([f for f in files if f.endswith(".csv")])):
            gt_file = gt_file_dict[file.split("-")[3]]
            os.system(f"nethobench neuro-scores --preds {os.path.join(root,file)} --gt data/gt/{gt_file}")
            os.system(f"nethobench fidelity-scores --preds {os.path.join(root,file)} --gt data/gt/{gt_file}")
            