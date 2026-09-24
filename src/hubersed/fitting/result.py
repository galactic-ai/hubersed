"""Check a MAP fit's parameters by name before they are saved or reused."""

from dataclasses import dataclass

import numpy as np


def _labels(theta):
    """Return prospector's theta_labels for a theta dict, in dict order."""
    out = []
    for name, value in theta.items():
        n = np.size(value)
        out += [name] if n == 1 else [f"{name}_{i + 1}" for i in range(n)]
    return tuple(out)


@dataclass(frozen=True)
class MapFitResult:
    """Best fit parameters of one galaxy, keyed by parameter name.

    Parameters
    ----------
    target_id : int
        DESI TARGETID.
    z : float
        Redshift used in the fit.
    theta : dict of str to np.ndarray
        Free parameter values by name, in the order of the theta vector. Vector
        parameters such as ``logsfr_ratios`` hold all their entries.
    labels : tuple of str
        ``model.theta_labels()``, one label per entry of the theta vector.

    Raises
    ------
    ValueError
        If ``labels`` does not match ``theta``. Prospector names the entries of a vector
        parameter ``name_1``, ``name_2`` and so on (``SpecModel.theta_labels``).
    """

    target_id: int
    z: float
    theta: dict
    labels: tuple

    def __post_init__(self):
        """Check that labels match theta."""
        expected = _labels(self.theta)
        if expected != tuple(self.labels):
            raise ValueError(
                f"TARGETID {self.target_id}: theta gives labels {expected}, "
                f"record has {tuple(self.labels)}"
            )

    def vector(self):
        """Return theta as one array in ``labels`` order."""
        return np.concatenate([np.atleast_1d(v) for v in self.theta.values()]).astype(float)

    @classmethod
    def from_record(cls, rec):
        """Build from a fit_one record and check it against the saved theta vector.

        Parameters
        ----------
        rec : dict
            Record written by ``map_fits.fit_one``.

        Returns
        -------
        MapFitResult

        Raises
        ------
        ValueError
            If the labels do not match ``theta_dict``, or ``theta_dict`` does not
            give back ``rec["theta"]``.
        """
        res = cls(
            int(rec["target_id"]), float(rec["z"]), dict(rec["theta_dict"]), tuple(rec["labels"])
        )
        if not np.array_equal(res.vector(), np.asarray(rec["theta"], float)):
            raise ValueError(f"TARGETID {res.target_id}: theta_dict does not match theta")
        return res
