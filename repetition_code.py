import numpy as np
import cudaq_qec as qec
from itertools import product

H = np.array([[1, 1, 0],
              [0, 1, 1]], dtype=np.uint8)
L = np.array([[1, 1, 1]], dtype=np.uint8)
p = [0.1, 0.1, 0.1]
syndromes = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]], dtype=np.uint8)

decoder = qec.get_decoder(
    "tensor_network_decoder",
    H,
    logical_obs=L,
    noise_model=p,
    dtype="float64",
    device="cuda",
)

result = decoder.decode_batch(syndromes)
posterior = result.result
print(posterior)

for s_idx, s in enumerate(syndromes):    # iterate through the syndromes
    Z = {0: 0.0,
         1: 0.0}                         # partition function Z_lambda(s), for when lambda = 0 and = 1
    L_flat = L.flatten()
    for e in product([0,1], repeat=3):   # generates all 8 possible tuple (0, 0, 0), (1, 0, 0)
        e = np.array(e)
        if not np.array_equal((H @ e) % 2, s):
            continue
        weight = 1.0
        for ei, pi in zip(e, p):         # probability pi for physical qubit ei
            if ei:
                weight *= np.prod(pi)
            else:
                weight *= np.prod(1-pi)
        lam = int((L_flat @ e) % 2)      # which logical sector
        Z[lam] += weight
    print(s, Z[1] / (Z[0] + Z[1]))