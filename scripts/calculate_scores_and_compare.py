import os

import subprocess
import os

def assert_no_changes_in_folder(repo_path: str, folder_path: str):
    """
    Asserts that there are no uncommitted changes in a specific folder.
    """
    # Ensure the repo path exists
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    # The git command components:
    # 'HEAD': Checks both staged and unstaged changes against the last commit.
    # '--quiet': Disables output and implies --exit-code.
    # '--': Tells git that what follows are file/folder paths, not branch names.
    cmd = ['git', 'diff', 'HEAD', '--quiet', '--', folder_path]
    
    try:
        # Run the command in the context of your repository
        result = subprocess.run(
            cmd,
            cwd=repo_path,      # Execute the command inside the git repo
            check=False         # We will handle the exit code manually
        )
        
        # Assert that the return code is 0 (No changes)
        assert result.returncode == 0, f"Changes detected in folder: {folder_path}"
        
    except FileNotFoundError:
        raise RuntimeError("Git is not installed or not accessible in the system path.")



neural_tuples = [
    ("data/neural/gt/data-clean-300-1140-ba4.csv", "data/neural/predictions/netho-seq-sl30-ba4-data100-seed101-epoch-10-predictions.csv"),
    ("data/neural/gt/data-clean-300-1140-ba4.csv", "data/neural/predictions/netho-seq-sl90-ba4-data100-seed103-epoch-10-predictions.csv"),
    ("data/neural/gt/data-clean-300-1140-ba4.csv", "data/neural/predictions/netho-seq-sl300-ba4-data100-seed102-epoch-10-predictions.csv"),
    ("data/neural/gt/data-clean-300-1140-ba16.csv", "data/neural/predictions/netho-seq-sl90-ba16-data100-seed102-epoch-30-predictions.csv"),
    ("data/neural/gt/data-clean-300-1140-ba16.csv", "data/neural/predictions/netho-seq-sl300-ba16-data100-seed101-epoch-30-predictions.csv")
]

etho_preds = [
    "data/behavioural/predictions/sequifier-behav-seq-real-2-last-50-predictions.csv",
    "data/behavioural/predictions/sequifier-behav-seq-real-2-last-100-predictions.csv",
    "data/behavioural/predictions/sequifier-behav-seq-real-2-last-200-predictions.csv"
]

if __name__ == "__main__":

    for gt, preds in neural_tuples:
        cmd = ["nethobench", "neuro-scores", "--gt", f"{gt}",  "--preds",  f"{preds}"]
        r = subprocess.run(cmd)
        assert r.returncode == 0, f"neuro-scores: {' '.join(cmd)}"

        cmd = ["nethobench", "neuro-analysis", "--gt", f"{gt}",  "--preds",  f"{preds}"]
        r = subprocess.run(cmd)
        assert r.returncode == 0, f"neuro-analysis: {' '.join(cmd)}"

        cmd = ["nethobench",  "fidelity-scores", "--gt",  f"{gt}",  "--preds", f"{preds}"]
        r = subprocess.run(cmd)
        assert r.returncode == 0, f"fidelity-scores: {' '.join(cmd)}"



    for preds in etho_preds:
        cmd = ["nethobench", "etho-scores", "--gt-dir", "data/behavioural/gt.parquet", "--inf-dir", f"{preds}"]
        r = subprocess.run(cmd)
        assert r.returncode == 0, f"etho-scores: {' '.join(cmd)}"

    cmd = ["nethobench", "cross-scores", "--gt", "data/cross/gt/cross-gt-behavior-neuro.csv", "--preds", "data/cross/predictions/sequifier-cross-noisy-behavior-neuro-last-100.csv"]
    r = subprocess.run(cmd)
    assert r.returncode == 0, f"cross-scores: {' '.join(cmd)}"


    my_repo = "."
    my_folder = "outputs" # Path relative to the repo root
    
    assert_no_changes_in_folder(my_repo, my_folder)
