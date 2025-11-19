# sinusoid_features.py
import math
import numpy as np
PLANET_PERIODS = {0: 10, 1: 20, 2: 200}
def build_features(planet: int,step: int,d: int = 5,max_steps: int = 1000):

    T = PLANET_PERIODS.get(planet, 20)
    phase = (step % T) / T
    angle = 2.0 * math.pi * phase
    if d == 3:
        feats = [
            1.0,
            math.sin(angle),
            math.cos(angle)]

    if d==5:
        feats = [
            1.0,
            math.sin(angle),
            math.cos(angle),
            math.sin(2.0 * angle),
            math.cos(2.0 * angle)]
    else:
        raise ValueError(f"Dimension des features non supportée: d={d}")
    return np.array(feats, dtype=float)
