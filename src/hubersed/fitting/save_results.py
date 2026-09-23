import os
import pickle


def load_galaxy_results(gal_id, output_dir):
    """Load full results for one galaxy."""
    pkl_path = os.path.join(output_dir, str(gal_id), "results.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)
