import os
import subprocess

# Directory containing the sample_full CSV files
SAMPLE_DIR = os.path.join("Sentiment_Score", "Directional_Score", "Directional_Sample")

# Get absolute path for the directory
workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sample_dir_abs = os.path.join(workspace_dir, SAMPLE_DIR)

# Path to splits.py
splits_py_path = os.path.join(os.path.dirname(__file__), "splits.py")

# Loop through all *_sample_full.csv files
for fname in os.listdir(sample_dir_abs):
    if fname.endswith("_sample_full.csv"):
        input_path = os.path.join(sample_dir_abs, fname)
        print(f"Processing {fname}...")
        subprocess.run([
            "python", splits_py_path, input_path
        ], check=True)
print("Batch splitting complete.")
