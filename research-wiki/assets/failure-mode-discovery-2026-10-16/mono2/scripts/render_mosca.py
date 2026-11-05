"""fmd mono2 lane: render a trained MoSca model at the held-out cameras WITHOUT test-time pose optimisation.
Calls MoSca's own mosca_evaluate.test_main(tto_flag=False): render_test() aligns the known held-out cameras by the
Sim(3) between the known and the (slightly refined) training trajectory, then renders s_model + d_model(t) at full
resolution into <logdir>/test/{1,2,3}_NNNNN.png. Its DyCheck metric step needs jax (not installed) and may fail after
the renders are written; that is caught.
Usage (from the MoSca repo root): render_mosca.py WS LOGDIR CFG"""
import sys, os, logging, torch
sys.path.insert(0, os.getcwd())
from omegaconf import OmegaConf
from mosca_evaluate import test_main
ws, logdir, cfgp = sys.argv[1:4]
cfg = OmegaConf.load(cfgp)
try:
    test_main(cfg, saved_dir=logdir, data_root=ws, device=torch.device("cuda"), tto_flag=False)
except Exception as e:
    logging.warning(f"test_main ended with {type(e).__name__}: {e}")
n = len(os.listdir(os.path.join(logdir, "test")))
print("rendered test frames:", n)
sys.exit(0 if n >= 900 else 6)
