"""Profile figure: disconnection map with the lesion outline, and the concept profile as bars."""

from __future__ import annotations

import numpy as np
import nibabel as nib


def profile_figure(path, disconnection_img, lesion_img, profile: dict) -> None:
    """Write a PNG: left a slice through the lesion (disconnection map, lesion contour in red),
    right the positive-disconnection score per concept (filled bars: significant)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting

    data = np.asanyarray(disconnection_img.dataobj)
    lesion = np.asanyarray(lesion_img.dataobj) > 0
    idx = np.argwhere(lesion)
    centre_vox = np.round(idx.mean(axis=0)) if len(idx) else np.array(lesion.shape) / 2
    z_mm = float(nib.affines.apply_affine(lesion_img.affine, centre_vox)[2])
    vmax = float(np.max(np.abs(data)))
    vmax = vmax if vmax > 0 else 1.0

    concepts = profile["concepts"]
    scores = [c["disconnection_positive"]["score"] for c in concepts]
    sig = [c["disconnection_positive"]["significant"] for c in concepts]
    plotted = [0.0 if s is None else s for s in scores]
    n = len(concepts)

    fig = plt.figure(figsize=(13, max(4.5, 0.28 * n + 1.5)), dpi=100)
    ax_map = fig.add_axes([0.02, 0.12, 0.42, 0.72])
    display = plotting.plot_stat_map(
        disconnection_img,
        display_mode="z",
        cut_coords=[z_mm],
        axes=ax_map,
        vmax=vmax,
        threshold=vmax * 0.02,
        colorbar=True,
        cmap="coolwarm",
        title="Disconnection map, lesion outline in black",
    )
    display.add_contours(lesion_img, levels=[0.5], colors="k", linewidths=1.0)

    ax_bar = fig.add_axes([0.55, 0.12, 0.42, 0.72])
    y = np.arange(n)[::-1]
    colors = ["#1f77b4" if s else "#c7c7c7" for s in sig]
    ax_bar.barh(y, plotted, color=colors, edgecolor="#555555", linewidth=0.5)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([c["concept"] for c in concepts], fontsize=8)
    ax_bar.set_xlabel("positive disconnection score (CDR)")
    ax_bar.set_title("Concept profile (blue: significant after correction)")
    ax_bar.axvline(1.0, color="#999999", linewidth=0.6, linestyle=":")
    fig.text(0.5, 0.02, "Research use only - not a medical device", ha="center", fontsize=9)
    fig.savefig(str(path), format="png", metadata={"Software": "cognet-stroke"})
    plt.close(fig)
