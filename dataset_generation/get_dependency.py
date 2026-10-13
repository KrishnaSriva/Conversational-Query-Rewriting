import numpy as np
import random
from collections import Counter
np.random.seed(42)
def _softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

def sample_next(
    prev_dep,
    temperature=1.0,
    dirichlet_concentration=200.00,
    rng=None
):

    matrix = {
  "Direct":   [("Direct",0.40),("Multi",0.25),("LongRange",0.35)],
  "LongRange":[("Direct",0.45),("LongRange",0.20),("Multi",0.35)],
  "Multi":    [("Direct",0.40) , ("Multi",0.35),("LongRange",0.25)],
  "NoDep":    [("Direct",1.0) , ("Multi",0.00),("LongRange",0.0)],
}
    if rng is None:
        rng = np.random

    choices, p = zip(*matrix[prev_dep])
    p = np.array(p, dtype=float)

    logits = np.log(np.clip(p, 1e-12, None))
    logits = logits / float(temperature)
    p_temp = _softmax(logits)

    if dirichlet_concentration is not None and dirichlet_concentration > 0:
        alpha = p_temp * float(dirichlet_concentration)
        alpha = np.clip(alpha, 1e-6, None)
        p_noisy = rng.dirichlet(alpha)
    else:
        p_noisy = p_temp

    return rng.choice(choices, p=p_noisy)



