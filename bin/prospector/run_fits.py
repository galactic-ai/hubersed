"""
run_fits.py — fits one galaxy, called by SLURM array job.
Usage: python run_fits.py --task_id 0 --outlier_list outliers.npy --output_dir /path/to/output
"""
import argparse
import traceback

from fit_single import fit_galaxy
from save_results import save_galaxy_results, is_done
import parameter_file as P

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id",      type=int, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--run_full",     action="store_true", default=True)
    parser.add_argument("--cont_nseeds",  type=int, default=3)
    parser.add_argument("--full_nseeds",  type=int, default=3)
    parser.add_argument("--cont_nprod",   type=int, default=1000)
    parser.add_argument("--full_nprod",   type=int, default=3000)
    args = parser.parse_args()

    # Load list of outlier indices
    outlier_indices = P.OUTLIERS_IDX
    outlier_idx = int(outlier_indices[args.task_id])

    # Get galaxy ID for checking if done
    _, _, _, _, gal_id = P.get_outlier_info(args.task_id, streaming=False)

    if is_done(outlier_idx, gal_id, args.output_dir):
        print(f"Task {args.task_id} (outlier {outlier_idx}, id={gal_id}): "
              f"already done, skipping")
        return

    print(f"Task {args.task_id}: fitting outlier {outlier_idx} (id={gal_id})")

    try:
        results = fit_galaxy(
            outlier_idx       = args.task_id,
            parameter_file    = None,
            run_continuum     = True,
            run_full          = args.run_full,
            cont_nseeds       = args.cont_nseeds,
            cont_maxfev       = 30_000,
            cont_nburn        = 300,
            cont_nprod        = args.cont_nprod,
            full_nseeds       = args.full_nseeds,
            full_maxfev       = 30_000,
            full_nburn        = 1000,
            full_nprod        = args.full_nprod,
        )
        save_galaxy_results(results, args.output_dir)
        print(f"Task {args.task_id}: done. "
              f"chi2_red_cont={results.get('chi2_red_cont', -99):.3f}")

    except Exception as e:
        print(f"Task {args.task_id}: FAILED with error: {e}")
        traceback.print_exc()
        # Save minimal failure record
        with open(f"{args.output_dir}/failed_{outlier_idx}.txt", "w") as f:
            f.write(f"outlier_idx={outlier_idx}, id={gal_id}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()