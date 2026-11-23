"""Run the official Habitat generator using a software EGL device."""

import runpy
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: generate_habitat_cpu.py KIT_PATH [generator options]")
    simulator = Path(sys.argv.pop(1)).resolve() / "simulator"
    sys.path.insert(0, str(simulator / "engine"))

    from habitat.config import read_write
    import probe_social

    config = probe_social.config

    def software_config(*args, **kwargs):
        cfg = config(*args, **kwargs)
        with read_write(cfg):
            cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        return cfg

    probe_social.config = software_config
    runpy.run_path(str(simulator / "generate.py"), run_name="__main__")


if __name__ == "__main__":
    main()
