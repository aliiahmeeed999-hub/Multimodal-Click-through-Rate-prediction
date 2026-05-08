# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Patched: use sys.executable + repo cwd for Windows / conda; default GPU 0.
# =========================================================================

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import fuxictr_version
from fuxictr import autotuner

_REPO_ROOT = os.path.dirname(os.path.realpath(__file__))


def grid_search_same_python(config_dir, gpu_list, expid_tag=None, script="run_expid.py"):
    """Same logic as fuxictr.autotuner.grid_search, but uses sys.executable and repo cwd."""
    experiment_id_list = autotuner.load_experiment_ids(config_dir)
    if expid_tag is not None:
        experiment_id_list = [e for e in experiment_id_list if str(expid_tag) in e]
        assert len(experiment_id_list) > 0, "tag=%s does not match any expid." % (expid_tag,)
    gpu_list = list(gpu_list)
    idle_queue = list(range(len(gpu_list)))
    processes = {}
    script_path = os.path.join(_REPO_ROOT, script)
    while len(experiment_id_list) > 0:
        if len(idle_queue) > 0:
            idle_idx = idle_queue.pop(0)
            gpu_id = gpu_list[idle_idx]
            expid = experiment_id_list.pop(0)
            cmd = [
                sys.executable,
                "-u",
                script_path,
                "--config",
                config_dir,
                "--expid",
                expid,
                "--gpu",
                str(gpu_id),
            ]
            p = subprocess.Popen(cmd, cwd=_REPO_ROOT)
            processes[idle_idx] = p
        else:
            time.sleep(3)
            for idle_idx, p in list(processes.items()):
                if p.poll() is not None:
                    idle_queue.append(idle_idx)
    for p in processes.values():
        p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../config/tuner_config.yaml",
        help="The config file for para tuning.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Use the tag to determine which expid to run (e.g. 001 for the first expid).",
    )
    parser.add_argument(
        "--gpu",
        nargs="+",
        type=int,
        default=[0],
        help="CUDA device id(s) for parallel tuner workers. Use -1 for CPU only.",
    )
    args = vars(parser.parse_args())
    gpu_list = args["gpu"]
    expid_tag = args["tag"]

    config_dir = autotuner.enumerate_params(args["config"])
    grid_search_same_python(config_dir, gpu_list, expid_tag)
