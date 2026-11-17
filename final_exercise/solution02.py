###
## Solution to exercise at the end of Week 02
###

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load the CSV file
# ------------------------------------------------------------
# Use np.genfromtxt since it handles headers and text columns
data = np.genfromtxt('three_normals.csv', delimiter=',', names=True, dtype=None, encoding=None)

# Check structure
print('Data columns:', data.dtype.names)
print('First 5 rows:\n', data[:5])

# Extract columns
x = data['x']
groups = data['group']
# Remove quotes (for later use)
groups = np.char.strip(groups, '"')
print('Groups:\n', groups)

# ------------------------------------------------------------
# 2. Descriptive statistics per group
# ------------------------------------------------------------
for g in np.unique(groups):
    xg = x[groups == g]
    print(f'Group {g}: mean={np.mean(xg):.3f}, std={np.std(xg, ddof=1):.3f}, n={len(xg)}')

# ------------------------------------------------------------
# 3. Histograms
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
bins = np.linspace(min(x), max(x), 40)
# Check out use of "zip" to create lists of tuples
for g, color in zip(np.unique(groups), ['blue', 'orange', 'green']):
    plt.hist(x[groups == g], bins=bins, alpha=0.5, label=f'group {g}', color=color)
plt.title('Overlaid Histograms by Group')
plt.xlabel('x'); plt.ylabel('Frequency')
plt.legend(); plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 4. Box plots
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
# Python form to synthetically create lists
data_groups = [x[groups == g] for g in np.unique(groups)]
plt.boxplot(data_groups, labels=[f'group {g}' for g in np.unique(groups)])
plt.title('Box Plots by Group')
plt.xlabel('Group'); plt.ylabel('x')
plt.show()

# ------------------------------------------------------------
# 5. Scatter plot (make a synthetic y = x + noise)
# ------------------------------------------------------------
# Add errors to y = x
rng = np.random.default_rng(4123)
y = x + rng.normal(0, 0.5, size=len(x))
# Scatter plot
plt.figure(figsize=(8, 5))
colors = {'A': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green'}
for g in np.unique(groups):
    mask = groups == g
    plt.scatter(x[mask], y[mask], s=10, alpha=0.5, color=colors[g], label=f'group {g}')
plt.title('Scatter Plot: y = x + noise')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 6. Heatmap (mean values in coarse bins)
# ------------------------------------------------------------
# Bin x values (example: 15 bins)
bins = np.linspace(np.min(x), np.max(x), 16)
digitised = np.digitize(x, bins)
# Build matrix: groups × bins (mean x values)
groups_unique = np.unique(groups)
heatmap = np.zeros((len(groups_unique), len(bins)-1))
for i, g in enumerate(groups_unique):
    for j in range(1, len(bins)):
        mask = (groups == g) & (digitised == j)
        if np.any(mask):
            heatmap[i, j-1] = np.mean(x[mask])
plt.figure(figsize=(8, 3))
plt.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="viridis")
plt.colorbar(label="mean(x)")
plt.yticks(range(len(groups_unique)), groups_unique)
plt.xticks(range(len(bins)-1), labels=[f"{bins[j]:.1f}" for j in range(len(bins)-1)], rotation=90)
plt.title("Heatmap: mean(x) by group and x-bin")
plt.xlabel("x-bin"); plt.ylabel("group")
plt.show()