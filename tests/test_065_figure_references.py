# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Tests for tutorial figure and GIF reference validation (`tools/check_figure_refs.py`)."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_figure_refs import (  # noqa: E402
    REPO_ROOT as CHECK_ROOT,
    get_git_tracked_files,
    main as check_figure_refs_main,
    run_mutation_checks,
    validate_all_tutorial_figures,
    validate_script_figure_refs,
)

MAX_GIF_BYTES = 5 * 1024 * 1024

EXPECTED_TUTORIAL_GIFS = [
    "tutorials/07_light_sources/07-06_led_reciprocity/figures/02_reciprocal_wave_movie.gif",
    "tutorials/04_radiation_and_antennas/04-03_dipole_near_mirror/figures/05_dipole_mirror_interference.gif",
    "tutorials/07_light_sources/07-03_microled_horn/figures/06_pulse_propagation_dgrin.gif",
    "tutorials/06_periodic_structures/06-02_sws_cold_test/figures/09_bloch_mode_animation.gif",
    "tutorials/07_light_sources/07-02_sled_tilted_facet/figures/pulse_propagation_7deg.gif",
    "tutorials/05_guided_waves/05-01_y_junction/figures/09_pulse_propagation.gif",
    "tutorials/05_guided_waves/05-05_crossing/figures/09_pulse_propagation.gif",
    "tutorials/05_guided_waves/05-06_mmi_1x4/figures/09_pulse_propagation.gif",
    "tutorials/05_guided_waves/05-08_directional_coupler/figures/09_pulse_propagation.gif",
]


def test_all_tutorial_figure_references_exist_and_tracked():
    """Verify every figure/GIF referenced in tutorial scripts exists, is < 5 MB, and is tracked by git."""
    tracked = get_git_tracked_files(CHECK_ROOT)
    errors = validate_all_tutorial_figures(CHECK_ROOT, tracked)
    assert not errors, "Missing or untracked tutorial figure references:\n" + "\n".join(
        errors
    )
    assert check_figure_refs_main() == 0

    for rel_gif in EXPECTED_TUTORIAL_GIFS:
        gif_path = CHECK_ROOT / rel_gif
        assert gif_path.is_file(), f"Expected tutorial GIF does not exist: {rel_gif}"
        size_bytes = gif_path.stat().st_size
        assert 0 < size_bytes < MAX_GIF_BYTES, (
            f"Tutorial GIF {rel_gif} size {size_bytes} bytes out of (0, {MAX_GIF_BYTES}) range"
        )
        with gif_path.open("rb") as fh:
            header = fh.read(6)
        assert header in (b"GIF87a", b"GIF89a"), (
            f"Invalid GIF header in {rel_gif}: {header!r}"
        )
        assert rel_gif in tracked, f"Tutorial GIF is not tracked by git: {rel_gif}"


def test_figure_reference_validator_mutations(tmp_path):
    """Verify negative-control mutations fail when figure/GIF references are missing or untracked."""
    tracked = get_git_tracked_files(CHECK_ROOT)
    run_mutation_checks(CHECK_ROOT, tracked)

    tut_dir = tmp_path / "tutorials" / "01_chap" / "01-01_demo"
    fig_dir = tut_dir / "figures"
    fig_dir.mkdir(parents=True)

    tracked_png = fig_dir / "01_plot.png"
    tracked_gif = fig_dir / "02_anim.gif"
    untracked_gif = fig_dir / "03_untracked.gif"
    tracked_png.write_bytes(b"\x89PNG\r\n\x1a\n")
    tracked_gif.write_bytes(b"GIF89a")
    untracked_gif.write_bytes(b"GIF89a")

    rel_png = tracked_png.relative_to(tmp_path).as_posix()
    rel_gif = tracked_gif.relative_to(tmp_path).as_posix()
    rel_untracked = untracked_gif.relative_to(tmp_path).as_posix()
    mock_tracked = {rel_png, rel_gif}

    valid_script = "\n".join(
        [
            "# ![Static plot](figures/01_plot.png)",
            "# See `figures/02_anim.gif`.",
            'display(Image(filename="figures/02_anim.gif"))',
            '_ = FIG_DIR / "01_plot.png"',
        ]
    )
    assert (
        validate_script_figure_refs(valid_script, tut_dir, tmp_path, mock_tracked) == []
    )

    # Missing Markdown image link
    err_md = validate_script_figure_refs(
        valid_script + "\n# ![Missing](figures/missing_anim.gif)\n",
        tut_dir,
        tmp_path,
        mock_tracked,
    )
    assert any("missing figure file" in e and "missing_anim.gif" in e for e in err_md)

    # Missing Image(filename=...) call
    err_img = validate_script_figure_refs(
        valid_script + '\ndisplay(Image(filename="figures/missing_call.gif"))\n',
        tut_dir,
        tmp_path,
        mock_tracked,
    )
    assert any("missing figure file" in e and "missing_call.gif" in e for e in err_img)

    # Missing FIG_DIR / "..." expression
    err_fdir = validate_script_figure_refs(
        valid_script + '\nout = FIG_DIR / "missing_figdir.png"\n',
        tut_dir,
        tmp_path,
        mock_tracked,
    )
    assert any(
        "missing figure file" in e and "missing_figdir.png" in e for e in err_fdir
    )

    # Existing on-disk file that is not tracked by git
    err_untracked = validate_script_figure_refs(
        valid_script + "\n# ![Untracked](figures/03_untracked.gif)\n",
        tut_dir,
        tmp_path,
        mock_tracked,
    )
    assert any("untracked figure file" in e and rel_untracked in e for e in err_untracked)
