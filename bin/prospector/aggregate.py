"""Run after all SLURM jobs complete to aggregate results."""
import numpy as np
import pandas as pd
from save_results import aggregate_summaries

from hubersed.paths import PATHS
RESULTS_DIR = str(PATHS["RESULTS"])

output_dir = RESULTS_DIR
rows = aggregate_summaries(output_dir)
df   = pd.DataFrame(rows)
df.to_csv(f"{output_dir}/all_summaries.csv", index=False)
print(f"Aggregated {len(df)} galaxies")
print(df[["id", "chi2_red_cont", "chi2_red_full", 
          "ism_NaD", "ism_MgI"]].describe())