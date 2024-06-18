from collections import defaultdict
import numpy as np
import time
import torch


def timer(func):
    def wrapper(*args, **kwargs):
        st = time.perf_counter()
        res = func(*args, **kwargs)
        en = time.perf_counter()
        return res, en - st
    return wrapper

@timer
def matmul(t1, t2):
    return t1.matmul(t2)

if __name__ == "__main__":
    t1 = torch.rand([10000, 500])
    t2 = torch.rand([500, 20000])
    n_samples = 100

    devices = ["cpu", "cuda"]
    times = defaultdict(list)
    for device in devices:
        for _ in range(n_samples):
            result, time_taken = matmul(t1.to(device=device), t2.to(device=device))
            times[device].append(time_taken)
        print(f"{device} percentiles: {np.percentile(times[device], q=[0.50, 0.90, 0.95, 0.99])}")
    
    speedups = [cpu / cuda for cpu, cuda in zip(times["cpu"], times["cuda"])]
    print(f"speedup percentiles: {np.percentile(speedups, q=[0.50, 0.90, 0.95, 0.99])}")

"""
Output for RTX 4050 6GiB (80 W):
--------------------------------
cpu percentiles: [0.30203317 0.30234825 0.30238763 0.30241914]
cuda percentiles: [7.73454349e-05 7.74297829e-05 7.74403264e-05 7.74487612e-05]
speedup percentiles: [47.07481633 74.47030797 77.89474442 80.63429359]
"""