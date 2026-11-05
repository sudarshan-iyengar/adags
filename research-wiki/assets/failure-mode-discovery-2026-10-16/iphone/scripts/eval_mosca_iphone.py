"""fmd mono2: MoSca's own iPhone evaluation on an existing run, metric step only.
mosca_evaluate.test_main(tto_flag=True, skip_test_gen=True): reads <logdir>/tto_test (written by the run's own TTO
step) and computes masked co-visible mPSNR/mSSIM/mLPIPS with eval_utils (writes tto_dycheck_metrics.xlsx).
The in-run call failed only because jax had no GPU backend. Usage (MoSca repo root): eval_mosca_iphone.py WS LOGDIR"""
import jax  # initialise the jax CUDA backend BEFORE torch loads its bundled CUDA 12.1 libs (else cuSOLVER fails to load)
print("jax devices", jax.devices())
import sys, os, torch
sys.path.insert(0, os.getcwd())
from omegaconf import OmegaConf
from mosca_evaluate import test_main
ws, logdir = sys.argv[1:3]
cfg = OmegaConf.load("./profile/iphone/iphone_fit.yaml")
test_main(cfg, saved_dir=logdir, data_root=ws, device=torch.device("cuda"), tto_flag=True, skip_test_gen=True)
import pandas as pd
df = pd.read_excel(os.path.join(logdir, "tto_dycheck_metrics.xlsx"))
print("COLUMNS", list(df.columns)); print("MEANS", df.mean(numeric_only=True).to_dict())
