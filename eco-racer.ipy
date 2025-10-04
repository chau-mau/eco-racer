import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# -------------------------
# Load dataset
# -------------------------
# Assume CSV format: columns = SNP1, SNP2,..., SNPn, Population
data = pd.read_csv('snp_data.csv')

# Extract SNP data and population labels
snp_columns = data.columns[:-1]  # all columns except last (Population)
X = data[snp_columns].values
pop_labels = data['Population'].values

# -------------------------
# Compute variance per population
# -------------------------
pop_variances = {}
unique_pops = np.unique(pop_labels)
for pop in unique_pops:
    pop_data = X[pop_labels == pop]
    pop_variances[pop] = np.mean(np.var(pop_data, axis=0))

# Get top 3 populations by variance
top_3_pops = sorted(pop_variances, key=pop_variances.get, reverse=True)[:3]

# -------------------------
# Perform PCA
# -------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create DataFrame for plotting
pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Population': pop_labels
})

# -------------------------
# Function to draw confidence ellipse
# -------------------------
def plot_confidence_ellipse(ax, x, y, n_std=2.0, **kwargs):
    """
    Draws an ellipse representing the covariance of x and y.
    """
    if x.size <= 1:
        return
    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    
    ell = Ellipse(xy=(mean_x, mean_y),
                  width=lambda_[0]*n_std*2, height=lambda_[1]*n_std*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])),
                  **kwargs)
    ax.add_patch(ell)

# -------------------------
# Plot PCA
# -------------------------
plt.figure(figsize=(12, 8), dpi=300)
ax = plt.gca()

# Colors for populations
palette = sns.color_palette("hsv", len(unique_pops))
pop_color_dict = {pop: palette[i] for i, pop in enumerate(unique_pops)}

# Scatter points
for pop in unique_pops:
    subset = pca_df[pca_df['Population'] == pop]
    ax.scatter(subset['PC1'], subset['PC2'], label=pop, color=pop_color_dict[pop], alpha=0.7, edgecolor='k', s=50)

# Draw confidence ellipses for top 3 populations
for pop in top_3_pops:
    subset = pca_df[pca_df['Population'] == pop]
    plot_confidence_ellipse(ax, subset['PC1'].values, subset['PC2'].values,
                            edgecolor=pop_color_dict[pop], facecolor='none', linewidth=2)

# Add annotation box for top 3 populations
textstr = 'Top 3 populations by variance:\n'
for i, pop in enumerate(top_3_pops):
    textstr += f"{i+1}. {pop}: {pop_variances[pop]:.6f}\n"

props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
ax.text(1.02, 0.5, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='center', bbox=props)

# Labels and legend
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
ax.set_title('PCA of SNP Data')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('PCA_plot_high_quality.png', dpi=600, bbox_inches='tight')
plt.show()
print("PCA plot saved as 'PCA_plot_high_quality.png'")
