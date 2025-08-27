import argparse
import os
from pathlib import Path
import mlflow

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-s", help="Source MLFlow experiment ID.")
parser.add_argument("-d", help="Destination MLFLow experiment ID.")
args = parser.parse_args()
old_exp = args.s
new_exp = args.d

exp_p = Path(os.environ["MLRUNS_DIR"])
old_p = exp_p / old_exp
new_p = exp_p / new_exp
for run in old_p.iterdir():
    if run.is_dir():
        new_dir = new_p / run.name
        run = run.replace(new_dir)
        meta_f = run / "meta.yaml"
        meta = yaml.safe_load(meta_f.read_text())
        meta["experiment_id"] = new_exp
        meta["artifact_uri"] = str(run / "artifacts")
        meta_str = yaml.safe_dump(meta)
        meta_f.write_text(meta_str)

mlflow.delete_experiment(old_exp)
