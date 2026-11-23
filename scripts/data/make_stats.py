r"""Fit the normalization stats of a system.

Takes the same overrides as ``mundo-train``, so the stats are fitted on the train
split and with the settings training will use, and writes them to ``system.stats_path``.

Example:
    python scripts/data/make_stats.py +experiment=era5 system.args.root=<data dir> \
        system.stats_path=<file>
"""

import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

from mundo.config import CONF_DIR
from mundo.systems import SYSTEMS


def make_stats(overrides: list[str]) -> Path:
    with initialize_config_dir(config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=overrides)
    path = Path(cfg.system.stats_path)
    if path.exists():
        raise FileExistsError(f"{path} exists: remove it to refit")
    system = SYSTEMS[str(cfg.system.name)](**dict(cfg.system.args))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(system.fit_stats(), indent=2) + "\n")
    return path


if __name__ == "__main__":
    print("wrote", make_stats(sys.argv[1:]))
