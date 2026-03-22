"""v13: Ensemble of v3 and v1 (CNN + MLP hybrid).

v3 (CNN) = 3.21e-04
v1 (MLP) = 7.30e-04

Combining complementary architectures.
"""

import numpy as np

from verify import evaluate


def main():
    # Load predictions
    v3 = np.load("predictions_v3.npz")["gamma_scores"]
    v1 = np.load("predictions_v1.npz")["gamma_scores"]

    # Ensemble: 70% CNN, 30% MLP (CNN is better)
    ensemble = 0.7 * v3 + 0.3 * v1

    np.savez("predictions_v13.npz", gamma_scores=ensemble)

    metric = evaluate(ensemble, "v13: Ensemble v3 (CNN) 70% + v1 (MLP) 30%")
    return metric


if __name__ == "__main__":
    main()
