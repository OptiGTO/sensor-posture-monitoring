MiniRocket multivariate kernel from sktime 0.40.1,
`sktime/transformations/panel/rocket/_minirocket_multi_numba.py`.
Author: Angus Dempster (angus924); license: LICENSE.sktime (BSD-3-Clause).
Source: https://github.com/sktime/sktime/blob/v0.40.1/sktime/transformations/panel/rocket/_minirocket_multi_numba.py

Only optional-import plumbing is replaced by direct, lazy-at-caller Numba imports.
The algorithm and seeded fit/transform are unchanged. Vendoring avoids downgrading
the existing scikit-learn runtime to satisfy the full sktime distribution.
Upstream source SHA256: 02d74c545205484bdaec06a40d6fe6193175177bbdfc12df7899e68443148439
