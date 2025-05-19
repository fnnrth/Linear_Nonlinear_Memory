# launcher.py
import os
import multiprocessing


n_threads = 4
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
os.environ["TORCH_NUM_THREADS"] = str(n_threads)

import subprocess
from multiprocessing import Pool

Ls = [100]
Ms=[100]
taus = [0.1]
fixed_init_options = [False]
runs = range(5)
epochs=100

def launch(args):
    L, tau, run, fixed_init = args
    cmd = [
        "python", "run_one_config.py",
        "--dataset", "smnist",
        "--L", str(L),
        "--tau", str(tau),
        "--run", str(run),
        "--epochs", str(epochs),
    ]
    if fixed_init:
        cmd.append("--fixed_init")
    return subprocess.call(cmd)

if __name__ == "__main__":
    jobs = [(L, tau, run, fixed_init) for L in Ls for tau in taus for run in runs for fixed_init in fixed_init_options]
    with Pool(processes=5) as pool:  # Adjust number of workers (CPUs)
        pool.map(launch, jobs)
