"""4DAnyone Space (the Gradio GUI) is not available in this fork.

Upstream's GUI drives the GVHMR and SMPL-X pose path and shows SMPL-X body geometry.
This fork replaces that path with SAM 3D Body (``--sam3d_npz``), which both of those
licences made impossible to use commercially, so the GUI cannot run as written.
``fdanyone/space/`` is kept unchanged so that merging upstream stays clean; porting it
to the SAM 3D Body pose is future work.

Use ``inference.py`` instead (or the comfyui-cumuli node pack, which supplies the pose).
"""

from __future__ import annotations

import sys


def main() -> None:
    print(__doc__, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
