import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Load all results
with open('results/test_results/baseline_resnet50.json', 'r') as f:
    baseline = json.load(f)

with open('results/test_results/ablation1_efficientnet_b0.json', 'r') as f:
    efficientnet = json.load(f)

with open('results/test_results/ablation2_resolution_512.json', 'r') as f:
    res512 = json.load(f)

# Extract data
experiments = ['ResNet-50\n1024×1024', 'EfficientNet-B0\n1024×1024', 'ResNet-50\n512×512']
mean_accs = [
    baseline['test_metrics']['mean_bit_accuracy'],
    efficientnet['test_metrics']['mean_bit_accuracy'],
    res512['test_metrics']['mean_bit_accuracy']
]
exact_matches = [
    baseline['test_metrics']['exact_match_rate'],
    efficientnet['test_metrics']['exact_match_rate'],
    res512['test_metrics']['exact_match_rate']
]

bit_labels = ['Bit 0\n(warm/cool)', 'Bit 1\n(sharp/soft)', 'Bit 2\n(grainy/clean)', 
              'Bit 3\n(bright/dark)', 'Bit 4\n(contrast)', 'Bit 5\n(saturation)',
              'Bit 6\n(detail)', 'Bit 7\n(vintage/modern)']

baseline_bits = baseline['test_metrics']['per_bit_accuracy']
efficientnet_bits = efficientnet['test_metrics']['per_bit_accuracy']
res512_bits = res512['test_metrics']['per_bit_accuracy']

# Figure 1: Overall Performance Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#2E86AB', '#A23B72', '#06A77D']
x = np.arange(len(experiments))

# Mean accuracy
bars1 = ax1.bar(x, [acc * 100 for acc in mean_accs], color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Mean Bit Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Mean Bit Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(experiments, fontsize=10)
ax1.set_ylim(90, 95)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=94, color='gray', linestyle='--', alpha=0.5, label='Target (94%)')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars1, mean_accs)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val*100:.2f}%', ha='center', va='bottom', fontweight='bold')

# Exact match
bars2 = ax2.bar(x, [em * 100 for em in exact_matches], color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Exact Match Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Exact Match Rate Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(experiments, fontsize=10)
ax2.set_ylim(50, 70)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='Target (60%)')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars2, exact_matches)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val*100:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/overall_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/overall_comparison.png")
plt.close()

# Figure 2: Per-Bit Comparison (Baseline vs EfficientNet)
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(bit_labels))
width = 0.35

bars1 = ax.bar(x - width/2, [b * 100 for b in baseline_bits], width, 
               label='ResNet-50 (Baseline)', color='#2E86AB', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, [b * 100 for b in efficientnet_bits], width,
               label='EfficientNet-B0', color='#A23B72', alpha=0.8, edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Architecture Comparison: Per-Bit Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(bit_labels, fontsize=9)
ax.legend(fontsize=11)
ax.set_ylim(80, 105)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')

plt.tight_layout()
plt.savefig('figures/architecture_per_bit.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/architecture_per_bit.png")
plt.close()

# Figure 3: Per-Bit Comparison (1024 vs 512 Resolution)
fig, ax = plt.subplots(figsize=(14, 6))

bars1 = ax.bar(x - width/2, [b * 100 for b in baseline_bits], width,
               label='1024×1024', color='#2E86AB', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, [b * 100 for b in res512_bits], width,
               label='512×512', color='#06A77D', alpha=0.8, edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Resolution Comparison: Per-Bit Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(bit_labels, fontsize=9)
ax.legend(fontsize=11)
ax.set_ylim(80, 105)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)

# Highlight improvements
for i in range(len(bit_labels)):
    diff = (res512_bits[i] - baseline_bits[i]) * 100
    if abs(diff) > 1:  # Only show significant differences
        y_pos = max(baseline_bits[i], res512_bits[i]) * 100 + 1
        color = 'green' if diff > 0 else 'red'
        ax.text(i, y_pos, f'{diff:+.1f}%', ha='center', fontsize=8, 
                color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/resolution_per_bit.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/resolution_per_bit.png")
plt.close()

print("\n All figures created successfully!")
print("Location: figures/")
