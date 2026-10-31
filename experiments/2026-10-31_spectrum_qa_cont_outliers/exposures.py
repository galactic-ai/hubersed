"""Save the DR1 SV3 bright exposures with their sky brightness and effective time.

One query of desi_dr1.exposure on the Astro Data Lab TAP service. analysis.py joins it to each
target through TILEID_LIST of the FastSpecFit VAC.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-31_spectrum_qa_cont_outliers/exposures.py``.
OUTPUT: results/2026-10-31_spectrum_qa_cont_outliers/sv3_bright_exposures.csv.
"""

import warnings

from astroquery.utils.tap.core import TapPlus

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-31_spectrum_qa_cont_outliers"
QUERY = (
    "SELECT expid, tileid, night, survey, program, sky_mag_g_spec, efftime_spec, seeing_gfa, "
    "transparency_gfa "
    "FROM desi_dr1.exposure WHERE survey = 'sv3' AND program = 'bright'"
)


def main():
    """Run the query and write the CSV."""
    warnings.filterwarnings("ignore", module="astroquery")
    tap = TapPlus(url="https://datalab.noirlab.edu/tap", verbose=False)
    tab = tap.launch_job(QUERY).get_results()
    OUT.mkdir(parents=True, exist_ok=True)
    tab.write(OUT / "sv3_bright_exposures.csv", format="csv", overwrite=True)
    print(f"{len(tab)} exposures on {len(set(tab['tileid']))} tiles -> sv3_bright_exposures.csv")


if __name__ == "__main__":
    main()
