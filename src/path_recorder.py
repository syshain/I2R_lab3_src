"""path_recorder — record the executed cup-tip path, then plot it in 3D.

Segments are collected while pick_and_place.py runs (gated_path records the
tip after every executed step; console vias are attached to the segment they
belong to). At the end of the run save_path_plots writes one PNG (3D plot,
vias drawn as larger open dots) plus a CSV of the raw points into
TRAJECTORY_DIR, for the report.
"""

import os
import time

import numpy as np

from lab_config import TRAJECTORY_DIR
from reporting import progress


class PathRecorder:
    """Executed cup-tip positions, segment by segment (mm, base frame)."""

    def __init__(self):
        self.segments = []          # [{'label', 'points_mm', 'vias_mm'}]
        self._current = None

    def start_segment(self, label):
        self._current = {'label': label, 'points_mm': [], 'vias_mm': []}
        self.segments.append(self._current)

    def add_tip(self, p_tip_m):
        """Record one executed cup-tip position (m, base frame)."""
        if self._current is not None:
            self._current['points_mm'].append(
                np.asarray(p_tip_m, dtype=float) * 1000.0)

    def add_via(self, p_tip_m):
        """Record a console-entered via point (m, base frame)."""
        if self._current is not None:
            self._current['vias_mm'].append(
                np.asarray(p_tip_m, dtype=float) * 1000.0)

    def has_data(self):
        return any(len(s['points_mm']) >= 1 for s in self.segments)


def save_path_plots(recorder, out_dir=TRAJECTORY_DIR):
    """3D plot (+ CSV) of every recorded segment into out_dir.

    Returns the PNG path, or None when there is nothing to plot. Vias the
    student entered in the console are drawn as larger open dots.
    """
    if not recorder.has_data():
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')           # no display needed; run is headless-safe
        import matplotlib.pyplot as plt
    except Exception as exc:
        progress("! could not plot executed paths (matplotlib: %s)" % exc)
        return None

    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    png_path = os.path.join(out_dir, 'executed_paths_%s.png' % stamp)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    palette = plt.get_cmap('tab10').colors
    for k, seg in enumerate(recorder.segments):
        if not seg['points_mm']:
            continue
        pts = np.asarray(seg['points_mm'])
        c = palette[k % len(palette)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-o', markersize=2,
                color=c, label=seg['label'])
        if seg['vias_mm']:
            vias = np.asarray(seg['vias_mm'])
            ax.scatter(vias[:, 0], vias[:, 1], vias[:, 2], s=110,
                       marker='o', facecolors='none', edgecolors=c,
                       linewidths=2.5, label='%s — vias' % seg['label'])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title('Executed cup-tip paths (base frame)')
    # Equal-ish aspect: table geometry should not look stretched.
    all_pts = np.vstack([np.asarray(s['points_mm'])
                         for s in recorder.segments if s['points_mm']])
    spans = np.maximum(all_pts.max(axis=0) - all_pts.min(axis=0), 1.0)
    ax.set_box_aspect(spans)
    ax.legend(loc='best', fontsize=8)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    csv_path = png_path[:-4] + '.csv'
    with open(csv_path, 'w', newline='') as f:
        f.write('segment,index,x_mm,y_mm,z_mm,is_via\n')
        for seg in recorder.segments:
            for j, p in enumerate(seg['points_mm']):
                f.write('%s,%d,%.2f,%.2f,%.2f,0\n'
                        % (seg['label'], j, p[0], p[1], p[2]))
            for j, p in enumerate(seg['vias_mm']):
                f.write('%s,%d,%.2f,%.2f,%.2f,1\n'
                        % (seg['label'], j, p[0], p[1], p[2]))
    progress("executed-path plot: %s  (+ %s)" % (png_path, csv_path))
    return png_path
