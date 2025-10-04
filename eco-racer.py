#!/usr/bin/env python3
"""
Integrated SNP Informativeness & PCA Analysis
- Calculates Wright's FST, Informativeness (In), and δ values
- Identifies informative SNPs (AIMs)
- Performs PCA on selected SNPs
- Plots PCA with confidence ellipses for top populations
"""

import os
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.0
})

# -----------------------
# VCF Parsing
# -----------------------
def parse_vcf(vcf_file, selected_positions=None):
    """Parse VCF and return genotype matrix"""
    genotypes_dict = {}
    samples = []

    if vcf_file.endswith('.gz'):
        file_handle = gzip.open(vcf_file, 'rt')
    else:
        file_handle = open(vcf_file, 'r')

    with file_handle:
        for line in file_handle:
            if line.startswith('##'):
                continue
            elif line.startswith('#CHROM'):
                headers = line.strip().split('\t')
                samples = headers[9:]
            else:
                fields = line.strip().split('\t')
                chrom, pos = fields[0], int(fields[1])
                variant_id = f"{chrom}:{pos}"

                if selected_positions and variant_id not in selected_positions:
                    continue

                gt_data = []
                for i in range(9, len(fields)):
                    gt = fields[i].split(':')[0]
                    if gt in ['0/0', '0|0']:
                        gt_data.append(0)
                    elif gt in ['0/1', '1/0', '0|1', '1|0']:
                        gt_data.append(1)
                    elif gt in ['1/1', '1|1']:
                        gt_data.append(2)
                    else:
                        gt_data.append(np.nan)
                genotypes_dict[variant_id] = gt_data

    if selected_positions:
        ordered_variants = [pos for pos in selected_positions if pos in genotypes_dict]
    else:
        ordered_variants = list(genotypes_dict.keys())

    genotype_matrix = np.array([genotypes_dict[var] for var in ordered_variants]).T
    return genotype_matrix, samples, ordered_variants

# -----------------------
# Load sample metadata
# -----------------------
def load_sample_metadata(sample_file, samples):
    pop_df = pd.read_csv(sample_file, sep='\t', header=0)
    sample_col = [c for c in pop_df.columns if 'sample' in c.lower()][0] if any('sample' in c.lower() for c in pop_df.columns) else pop_df.columns[0]
    pop_col = [c for c in pop_df.columns if 'pop' in c.lower() or 'race' in c.lower()][0] if any('pop' in c.lower() or 'race' in c.lower() for c in pop_df.columns) else pop_df.columns[1]
    pop_map = dict(zip(pop_df[sample_col], pop_df[pop_col]))
    populations = [pop_map.get(s, 'Unknown') for s in samples]
    return populations, pop_map

# -----------------------
# Calculate allele frequencies
# -----------------------
def calculate_allele_freqs(genotype_matrix, populations):
    pop_indices = defaultdict(list)
    for i, pop in enumerate(populations):
        if pop != 'Unknown':
            pop_indices[pop].append(i)

    pop_af = {}
    pop_counts = {}

    for pop, indices in pop_indices.items():
        pop_genotypes = genotype_matrix[indices, :]
        af_list, count_list = [], []

        for col in range(pop_genotypes.shape[1]):
            gts = pop_genotypes[:, col]
            valid = gts[~np.isnan(gts)]
            ref_count = np.sum(valid == 0)
            alt_count = np.sum(valid == 2)
            het_count = np.sum(valid == 1)
            total = ref_count + alt_count + het_count
            if total > 0:
                ref_freq = (ref_count + 0.5*het_count)/total
                alt_freq = (alt_count + 0.5*het_count)/total
            else:
                ref_freq = alt_freq = 0
            af_list.append([ref_freq, alt_freq])
            count_list.append([ref_count+het_count, alt_count+het_count])
        pop_af[pop] = np.array(af_list)
        pop_counts[pop] = np.array(count_list)
    return pop_af, pop_counts

# -----------------------
# FST, In, delta calculations
# -----------------------
def calculate_fst(pop_af, pop_counts):
    pop_list = list(pop_af.keys())
    n_variants = len(pop_af[pop_list[0]])
    fst_values = np.zeros(n_variants)
    pair_count = 0
    for i in range(len(pop_list)):
        for j in range(i+1, len(pop_list)):
            p1 = pop_af[pop_list[i]][:,1]
            p2 = pop_af[pop_list[j]][:,1]
            var_mean = (p1 + p2)/2
            var_between = ((p1 - var_mean)**2 + (p2 - var_mean)**2)/2
            var_total = var_mean*(1-var_mean)
            fst_pair = np.divide(var_between, var_total, out=np.zeros_like(var_between), where=var_total!=0)
            fst_values += np.clip(fst_pair,0,1)
            pair_count += 1
    fst_values /= max(pair_count,1)
    return fst_values

def calculate_informativeness(pop_af):
    pop_list = list(pop_af.keys())
    n_variants = len(pop_af[pop_list[0]])
    in_values = np.zeros(n_variants)
    for var_idx in range(n_variants):
        freqs = [pop_af[pop][var_idx] for pop in pop_list]
        mean_freq = np.mean(freqs, axis=0)
        in_values[var_idx] = 1 - np.sum(mean_freq**2)
    return in_values

def calculate_delta(pop_af):
    pop_list = list(pop_af.keys())
    n_variants = len(pop_af[pop_list[0]])
    delta_values = np.zeros(n_variants)
    for var_idx in range(n_variants):
        max_delta = 0
        for i in range(len(pop_list)):
            for j in range(i+1,len(pop_list)):
                deltas = np.abs(pop_af[pop_list[i]][var_idx]-pop_af[pop_list[j]][var_idx])
                max_delta = max(max_delta, np.max(deltas))
        delta_values[var_idx] = max_delta
    return delta_values

# -----------------------
# Filter SNPs
# -----------------------
def filter_snps(results_df, fst_thresh=0.1, in_thresh=0.1, delta_thresh=0.1):
    common_snps = results_df[(results_df['fst']>fst_thresh) &
                             (results_df['informativeness']>in_thresh) &
                             (results_df['delta']>delta_thresh)].copy()
    return common_snps

# -----------------------
# PCA & Plotting
# -----------------------
def perform_pca(genotype_matrix, populations, snp_positions):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(genotype_matrix)
    pca = PCA(n_components=10)
    pcs = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    return pcs, explained_var

def plot_pca(pcs, populations, top_pops=None):
    df = pd.DataFrame({
        'PC1': pcs[:,0],
        'PC2': pcs[:,1],
        'Population': populations
    })
    unique_pops = [p for p in set(populations) if p!='Unknown']
    n_pops = len(unique_pops)
    colors = sns.color_palette("tab10", n_pops) if n_pops<=10 else sns.color_palette("husl", n_pops)
    pop_colors = dict(zip(unique_pops, colors))
    fig, ax = plt.subplots(figsize=(12,10))
    for pop in unique_pops:
        subset = df[df['Population']==pop]
        ax.scatter(subset['PC1'], subset['PC2'], label=pop, c=[pop_colors[pop]], alpha=0.7, edgecolors='black', s=60)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA of Informative SNPs')
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Draw confidence ellipses for top populations
    if top_pops:
        for pop in top_pops:
            subset = df[df['Population']==pop][['PC1','PC2']].values
            if len(subset) < 2:
                continue
            cov = np.cov(subset, rowvar=False)
            mean = np.mean(subset, axis=0)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
            width, height = 2 * np.sqrt(eigvals * stats.chi2.ppf(0.95, df=2))
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                              edgecolor=pop_colors.get(pop,'grey'), fc='none', lw=2, ls='--')
            ax.add_patch(ellipse)

    plt.tight_layout()
    plt.savefig("PCA_informative_SNPs.png", dpi=600)
    plt.show()

# -----------------------
# Main workflow
# -----------------------
if __name__ == "__main__":
    vcf_file = "example.vcf.gz"         # Replace with your VCF path
    sample_file = "sample_metadata.tsv" # Replace with your metadata file

    print("Parsing VCF...")
    genotype_matrix, samples, snp_positions = parse_vcf(vcf_file)
    print(f"Loaded {genotype_matrix.shape[1]} SNPs for {genotype_matrix.shape[0]} samples.")

    populations, pop_map = load_sample_metadata(sample_file, samples)
    print(f"Detected populations: {set(populations)}")

    # Calculate allele frequencies
    pop_af, pop_counts = calculate_allele_freqs(genotype_matrix, populations)

    # Calculate FST, Informativeness, Delta
    fst_values = calculate_fst(pop_af, pop_counts)
    in_values = calculate_informativeness(pop_af)
    delta_values = calculate_delta(pop_af)

    # Compile results
    results_df = pd.DataFrame({
        'SNP': snp_positions,
        'fst': fst_values,
        'informativeness': in_values,
        'delta': delta_values
    })

    # Filter informative SNPs (AIMs)
    informative_snps = filter_snps(results_df, fst_thresh=0.1, in_thresh=0.1, delta_thresh=0.1)
    print(f"Selected {informative_snps.shape[0]} informative SNPs (AIMs).")

    # Subset genotype matrix to informative SNPs
    selected_positions = informative_snps['SNP'].tolist()
    genotype_matrix_aims, _, _ = parse_vcf(vcf_file, selected_positions=selected_positions)

    # PCA
    pcs, explained_var = perform_pca(genotype_matrix_aims, populations, selected_positions)
    print(f"Explained variance by PC1 & PC2: {explained_var[0]:.2f}, {explained_var[1]:.2f}")

    # Plot PCA with ellipses for top 3 populations
    pop_counts_dict = pd.Series(populations).value_counts()
    top_pops = pop_counts_dict.nlargest(3).index.tolist()
    plot_pca(pcs, populations, top_pops=top_pops)

    # Save top AIMs table
    informative_snps.sort_values(by=['fst','informativeness','delta'], ascending=False, inplace=True)
    informative_snps.to_csv("Top_AIMs.csv", index=False)
    print("Analysis complete. PCA plot saved as 'PCA_informative_SNPs.png'.")
    print("Top AIMs saved as 'Top_AIMs.csv'.")
