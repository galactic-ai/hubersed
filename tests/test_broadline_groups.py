"""Line groups and parameters of the broadline profile options."""

import numpy as np
import pytest

pytestmark = pytest.mark.fsps  # broadline and prospect read SPS_HOME files on import

# Cue line names at rest 3500-9000 A (cuejax/data/cue_emlines_info.dat)
BALMER = ["Ba-8 3798", "Ba-7 3835", "Ba-6 3889", "Ba-5 3970", "Ba-delta 4101.76A",
          "Ba-gamma 4341", "Ba-beta 4861", "Ba-alpha 6563"]  # fmt: skip
HEI = ["He I 3888.63A", "He I 4471.49A", "He I 5875.64A", "He I 6678.15A", "He I 7065.22A"]
HIGH = ["[Ne III] 3869", "[Ne III] 3968", "[O III] 4363", "He II 4685.64A", "[Ar IV] 4711",
        "[Ne IV] 4720", "[Ar IV] 4740", "[O III] 4959", "[O III] 5007", "[Ar IV] 7332"]  # fmt: skip
LOW = ["[S III] 3722", "[O II] 3726", "[O II] 3729", "[S II] 4070", "[S II] 4078", "[C  I] 4621",
       "[Ar III] 5192", "[N I] 5200", "[Cl III] 5518", "[Cl III] 5538", "[O I] 5577",
       "[N II] 5755", "[O I] 6300", "[S III] 6312", "[O I] 6363", "[N II] 6548", "[N II] 6584",
       "[S II] 6716", "[S II] 6731", "[Ar III] 7135", "[O II] 7323", "[O II] 7332",
       "[Ar III] 7751", "[Cl II] 8579", "[C I] 8727"]  # fmt: skip
NAMES = BALMER + HEI + HIGH + LOW


def test_groups_with_he_balmer():
    """Balmer and He I lines get the Balmer profile; [O III], [Ne III], [Ar IV], [Ne IV], He II are high."""
    from hubersed.sps import broadline

    balmer, high = broadline.line_groups(NAMES, he_balmer_profile=True)
    n_b, n_h = len(BALMER) + len(HEI), len(HIGH)
    assert balmer.tolist() == [True] * n_b + [False] * (n_h + len(LOW))
    assert high.tolist() == [False] * n_b + [True] * n_h + [False] * len(LOW)


def test_groups_default_keeps_he_out_of_balmer():
    """Without the option the He I lines stay out of the Balmer group, as before."""
    from hubersed.sps import broadline

    balmer, high = broadline.line_groups(NAMES)
    assert balmer.sum() == len(BALMER)
    # He I lines are in neither group, so they keep the low-ionization width
    he = np.isin(NAMES, HEI)
    assert not balmer[he].any() and not high[he].any()


def test_add_broad_params_options():
    """The options add their parameters only when set, and the split needs the forbidden width."""
    from prospect.models import priors

    from hubersed.sps import broadline

    base = {
        "eline_sigma": dict(
            N=1, isfree=True, init=100.0, prior=priors.TopHat(mini=10.0, maxi=250.0)
        )
    }
    p = broadline.add_broad_params(dict(base), forbidden_broad="shared", sigma_split=45.0)
    assert "eline_sigma_forb_hi" not in p and "eline_he_balmer_profile" not in p
    p = broadline.add_broad_params(
        dict(base), forbidden_broad="shared", sigma_split=45.0, split_forbidden=True,
        he_balmer_profile=True,
    )  # fmt: skip
    assert p["eline_sigma_forb_hi"]["prior"].params["maxi"] == 45.0
    assert p["eline_he_balmer_profile"] == dict(N=1, isfree=False, init=1.0)
    with pytest.raises(ValueError):
        broadline.add_broad_params(dict(base), separate_forbidden_width=False, split_forbidden=True)
