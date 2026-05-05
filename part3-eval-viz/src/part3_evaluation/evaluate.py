import os
import sys
import json
import yaml
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import WatermarkDataset
from src.models import get_model
from src.utils import compute_metrics, print_metrics

def evaluate_test_set():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint_path = config['output']['checkpoint']
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {config['experiment']['name']}")
    print(f"Checkpoint: epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Val exact match at save time: {checkpoint['metrics']['exact_match_rate']:.4f}")
    print(f"{'='*60}\n")
    
    # Load model
    model = get_model(
        architecture=config['model']['architecture'],
        num_outputs=8,
        pretrained=False  # Don't need pretrained weights for evaluation
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test set
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = WatermarkDataset(
        metadata_path=config['data']['metadata_path'],
        image_dir=config['data']['images_path'],
        transform=transform
    )
    
    with open(config['data']['splits_path'], 'r') as f:
        splits = json.load(f)
    
    test_dataset = Subset(full_dataset, splits['test'])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Evaluate
    all_preds = []
    all_targets = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            targets = batch['bits'].to(device)
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds)
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = compute_metrics(all_preds, all_targets)
    
    print(f"\n{'='*60}")
    print("TEST SET RESULTS (unseen during training)")
    print(f"{'='*60}")
    print_metrics(metrics, prefix='Test ')
    print(f"{'='*60}\n")
    
    # Compare to validation
    val_exact = checkpoint['metrics']['exact_match_rate']
    test_exact = metrics['exact_match_rate']
    diff = abs(val_exact - test_exact)
    
    print(f"Comparison:")
    print(f"  Val exact match (saved):  {val_exact:.4f}")
    print(f"  Test exact match:         {test_exact:.4f}")
    print(f"  Difference:               {diff:.4f} ({diff/val_exact*100:.1f}%)")
    
    if test_exact >= val_exact * 0.95:
        print("\n EXCELLENT: Test within 5% of validation")
        print(" Model generalizes well, minimal overfitting")
    elif test_exact >= val_exact * 0.85:
        print("\n  MILD OVERFITTING: Test 5-15% worse than validation")
        print(" Acceptable for this task, consider early stopping next time")
    else:
        print("\n SEVERE OVERFITTING: Test >15% worse than validation")
        print(" Model memorized training data, needs regularization")
    
    # Save results to JSON
    results_dir = os.path.dirname(config['output']['results'])
    os.makedirs(os.path.join(results_dir, 'test_results'), exist_ok=True)
    
    results_path = os.path.join(results_dir, 'test_results', f"{config['experiment']['name']}.json")
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': config['experiment']['name'],
            'architecture': config['model']['architecture'],
            'checkpoint_epoch': checkpoint.get('epoch'),
            'val_exact_match': val_exact,
            'test_metrics': {
                'mean_bit_accuracy': metrics['mean_bit_accuracy'],
                'exact_match_rate': metrics['exact_match_rate'],
                'per_bit_accuracy': metrics['per_bit_accuracy']
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}\n")

if __name__ == '__main__':
    evaluate_test_set()