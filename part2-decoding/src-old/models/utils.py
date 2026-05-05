import torch
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(predictions, targets):
    """
    Args:
        predictions: (N, 8) binary predictions
        targets: (N, 8) ground truth bits
    
    Returns:
        dict with per-bit accuracy and exact match rate
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Per-bit accuracy
    per_bit_acc = []
    for i in range(8):
        acc = accuracy_score(targets[:, i], predictions[:, i])
        per_bit_acc.append(acc)
    
    # Exact match (all 8 bits correct)
    exact_match = (predictions == targets).all(axis=1).mean()
    
    metrics = {
        'per_bit_accuracy': per_bit_acc,
        'mean_bit_accuracy': np.mean(per_bit_acc),
        'exact_match_rate': exact_match
    }
    
    return metrics

def print_metrics(metrics, prefix=''):
    """Pretty print metrics"""
    print(f"\n{prefix}Metrics:")
    print(f"  Mean bit accuracy: {metrics['mean_bit_accuracy']:.4f}")
    print(f"  Exact match rate:  {metrics['exact_match_rate']:.4f}")
    print(f"  Per-bit accuracy:")
    for i, acc in enumerate(metrics['per_bit_accuracy']):
        print(f"    Bit {i}: {acc:.4f}")

# Test
if __name__ == '__main__':
    # Simulate random predictions
    preds = torch.randint(0, 2, (100, 8)).float()
    targets = torch.randint(0, 2, (100, 8)).float()
    
    metrics = compute_metrics(preds, targets)
    print_metrics(metrics, prefix='Test ')
    
    print(f"\n✓ Expected ~50% per-bit accuracy (random guessing)")
    print(f"✓ Expected ~0.4% exact match rate (0.5^8 = {0.5**8:.4f})")
