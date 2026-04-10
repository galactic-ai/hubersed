import os
import pickle
import json

def save_galaxy_results(results, output_dir):
    """Save results for one galaxy — pickle for arrays, JSON for scalars."""
    gal_id  = results["id"]
    out_dir = os.path.join(output_dir, str(gal_id))
    os.makedirs(out_dir, exist_ok=True)

    # Save full results as pickle
    with open(os.path.join(out_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Save scalar summary as JSON for easy aggregation later
    summary = {
        "outlier_idx":     results["outlier_idx"],
        "id":              int(results["id"]),
        "redshift":        float(results["redshift"]),
        "continuum_status": results.get("continuum_status", "not_run"),
        "full_status":     results.get("full_status", "not_run"),
    }
    for fit in ["cont", "full"]:
        if results.get(f"{fit}_status") == "success" \
                or results.get(f"continuum_status") == "success":
            key = f"chi2_red_{fit}" if fit == "full" else "chi2_red_cont"
            if key in results:
                summary[key] = results[key]
            # Key scalar params
            params = results.get(f"params_{fit}", {})
            for p in ["logmass", "logzsol", "dust2", 
                      "dust_ratio", "dust_index", "sigma_smooth",
                      "gas_logz", "gas_logu", "eline_sigma"]:
                if p in params and "q50" in params[p]:
                    summary[f"{fit}_{p}_q50"] = float(params[p]["q50"])
                    summary[f"{fit}_{p}_q16"] = float(params[p]["q16"])
                    summary[f"{fit}_{p}_q84"] = float(params[p]["q84"])
            # ISM residuals
            if "ism_residuals" in results:
                for line, val in results["ism_residuals"].items():
                    summary[f"ism_{line}"] = val

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved → {out_dir}")

def load_galaxy_results(gal_id, output_dir):
    """Load full results for one galaxy."""
    pkl_path = os.path.join(output_dir, str(gal_id), "results.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def aggregate_summaries(output_dir):
    """Collect all summary JSONs into one numpy-friendly dict."""
    import json, glob
    rows = []
    for path in glob.glob(os.path.join(output_dir, "*/summary.json")):
        with open(path) as f:
            rows.append(json.load(f))
    return rows

def is_done(outlier_idx, gal_id, output_dir):
    """Check if this galaxy has already been fit."""
    return os.path.exists(
        os.path.join(output_dir, str(gal_id), "results.pkl")
    )