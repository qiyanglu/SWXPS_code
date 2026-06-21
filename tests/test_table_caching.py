from pathlib import Path

import numpy as np

from swxps.imfp import clear_imfp_cache, load_imfp
from swxps.optical_constants import (
    clear_optical_constants_cache,
    load_optical_constants,
)


def _write_optical_table(path: Path, first_delta: str = "0.01") -> None:
    path.write_text(
        "Material Density=1.0\n"
        "Energy(eV), Delta, Beta\n"
        f"100 {first_delta} 0.001\n"
        "200 0.02 0.002\n",
        encoding="utf-8",
    )


def _write_imfp_table(path: Path, first_imfp: str = "5.0") -> None:
    path.write_text(
        "COMPOUND\n"
        "Material\n"
        "ENERGY IMFP\n"
        f"100 {first_imfp}\n"
        "200 8.0\n",
        encoding="utf-8",
    )


def test_optical_constants_reuses_cached_table(tmp_path):
    path = tmp_path / "material.dat"
    _write_optical_table(path)
    clear_optical_constants_cache()

    first = load_optical_constants(path)
    second = load_optical_constants(path)

    assert second is first


def test_optical_constants_cache_invalidates_after_rewrite(tmp_path):
    path = tmp_path / "material.dat"
    _write_optical_table(path)
    clear_optical_constants_cache()
    first = load_optical_constants(path)

    _write_optical_table(path, first_delta="0.003")
    second = load_optical_constants(path)

    assert second is not first
    assert second.delta[0] == 0.003


def test_optical_constants_cached_and_fresh_results_match(tmp_path):
    path = tmp_path / "material.dat"
    _write_optical_table(path)
    clear_optical_constants_cache()
    cached = load_optical_constants(path)

    clear_optical_constants_cache()
    fresh = load_optical_constants(path)

    assert fresh is not cached
    assert fresh.material == cached.material
    assert fresh.density == cached.density
    assert np.array_equal(fresh.energy_ev, cached.energy_ev)
    assert np.array_equal(fresh.delta, cached.delta)
    assert np.array_equal(fresh.beta, cached.beta)


def test_imfp_reuses_cached_table(tmp_path):
    path = tmp_path / "material.ANG"
    _write_imfp_table(path)
    clear_imfp_cache()

    first = load_imfp(path)
    second = load_imfp(path)

    assert second is first


def test_imfp_cache_invalidates_after_rewrite(tmp_path):
    path = tmp_path / "material.ANG"
    _write_imfp_table(path)
    clear_imfp_cache()
    first = load_imfp(path)

    _write_imfp_table(path, first_imfp="6.25")
    second = load_imfp(path)

    assert second is not first
    assert second.imfp[0] == 6.25


def test_imfp_cached_and_fresh_results_match(tmp_path):
    path = tmp_path / "material.ANG"
    _write_imfp_table(path)
    clear_imfp_cache()
    cached = load_imfp(path)

    clear_imfp_cache()
    fresh = load_imfp(path)

    assert fresh is not cached
    assert fresh.material == cached.material
    assert np.array_equal(fresh.kinetic_energy_ev, cached.kinetic_energy_ev)
    assert np.array_equal(fresh.imfp, cached.imfp)
