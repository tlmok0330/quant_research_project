"""Statistical inference: regression, break tests, permutation and DiD."""

from src.inference.breaks import (
    chow_test,
    sup_wald_test,
)
from src.inference.did import staggered_did, randomization_inference
from src.inference.multiple_testing import TestRegistry, adjust_pvalues
from src.inference.permutation import (
    holiday_matched_test,
    permutation_test,
    stratified_permutation_test,
)
from src.inference.regression import (
    RegressionResult,
    newey_west_lags,
    ols_cluster,
    ols_hac,
)

__all__ = [
    "RegressionResult",
    "TestRegistry",
    "adjust_pvalues",
    "chow_test",
    "holiday_matched_test",
    "newey_west_lags",
    "ols_cluster",
    "ols_hac",
    "permutation_test",
    "randomization_inference",
    "staggered_did",
    "stratified_permutation_test",
    "sup_wald_test",
]
