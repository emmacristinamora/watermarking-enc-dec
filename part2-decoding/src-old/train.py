import os
import sys
import json
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import WatermarkDataset
from src.models import get_model
from src.utils import compute_metrics, print_metrics

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc='Training'):
        images = batch['image'].to(device)
        targets = batch['bits'].to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    
    for batch in tqdm(loader, desc='Evaluating'):
        images = batch['image'].to(device)
        targets = batch['bits'].to(device)
        
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        all_preds.append(preds)
        all_targets.append(targets)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(loader)
    
    return metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"{'='*60}\n")
    
    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = WatermarkDataset(
        metadata_path=config['data']['metadata_path'],
        image_dir=config['data']['images_path'],
        transform=transform
    )
    
    # Load splits
    with open(config['data']['splits_path'], 'r') as f:
        splits = json.load(f)
    
    train_dataset = Subset(full_dataset, splits['train'])
    val_dataset = Subset(full_dataset, splits['val'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Get model using factory function
    model = get_model(
        architecture=config['model']['architecture'],
        num_outputs=8,
        pretrained=config['model']['pretrained']
    ).to(device)
    
    print(f"Model: {config['model']['architecture']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Create output directory
    os.makedirs(os.path.dirname(config['output']['checkpoint']), exist_ok=True)
    
    # Training loop
    best_exact_match = 0
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"{'='*60}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"\nTrain loss: {train_loss:.4f}")
        print_metrics(val_metrics, prefix='Val ')
        
        # Save best model
        if val_metrics['exact_match_rate'] > best_exact_match:
            best_exact_match = val_metrics['exact_match_rate']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': val_metrics
            }, config['output']['checkpoint'])
            
            print(f"\n✓ Saved best model (exact match: {best_exact_match:.4f})")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best exact match rate: {best_exact_match:.4f}")
    print(f"Checkpoint saved: {config['output']['checkpoint']}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()