"""Create ensemble from multiple saved predictions.

Usage: python create_ensemble.py v3 v10 v11 -w 0.33 0.33 0.34
"""

import argparse
import numpy as np
from verify import evaluate


def main():
    parser = argparse.ArgumentParser(description="Create ensemble of predictions")
    parser.add_argument("models", nargs="+", help="Model versions to ensemble (e.g. v3 v10 v11)")
    parser.add_argument("-w", "--weights", nargs="+", type=float, default=None,
                       help="Weights for each model (must sum to 1)")
    parser.add_argument("-o", "--output", default="predictions_ensemble.npz",
                       help="Output filename")
    args = parser.parse_args()

    if len(args.models) != len(args.weights):
        print(f"Error: {len(args.models)} models but {len(args.weights)} weights")
        return

    if abs(sum(args.weights) - 1.0) > 1e-6:
        print(f"Warning: weights sum to {sum(args.weights)}, not 1.0")
        args.weights = [w / sum(args.weights) for w in args.weights]

    print("Loading predictions...")
    ensemble = None
    for model, weight in zip(args.models, args.weights):
        pred_file = f"predictions_{model}.npz"
        print(f"  {model}: {weight:.3f}")
        pred = np.load(pred_file)
        scores = pred["gamma_scores"]
        if ensemble is None:
            ensemble = weight * scores
        else:
            ensemble += weight * scores

    np.savez(args.output, gamma_scores=ensemble)
    print(f"Saved ensemble to {args.output}")

    # Evaluate
    metric = evaluate(ensemble, f"Ensemble: {' + '.join(args.models)}")


if __name__ == "__main__":
    main()
