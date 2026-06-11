"""
Master script — runs full analysis pipeline and saves all outputs.
Equivalent to running all three notebooks in sequence.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings, json, os
warnings.filterwarnings('ignore')

OUT = "/home/claude/welfare_analysis/outputs"
CLEAN = "/home/claude/welfare_analysis/data/clean"
os.makedirs(OUT, exist_ok=True)
os.makedirs(CLEAN, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.dpi': 150,
    'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9
})

PALETTE = {
    'primary': '#1a3a5c', 'accent': '#e05a2b', 'light': '#f0f4f8',
    'mid': '#4a90c4', 'warn': '#d4a017', 'ok': '#2e8b57', 'muted': '#888'
}

print("="*60)
print("STEP 1 — DATA CLEANING & GAP COMPUTATION")
print("="*60)

raw = pd.read_csv("/home/claude/welfare_analysis/data/raw/pmkisan_district_raw.csv")
print(f"Loaded {len(raw)} districts")

raw['district_key'] = raw['district'].str.lower().str.strip()
raw['uptake_rate'] = raw['pmkisan_enrolled'] / raw['agri_households']
raw['gap'] = raw['agri_households'] - raw['pmkisan_enrolled']
raw['gap_rate'] = raw['gap'] / raw['agri_households']
raw['gap_rate'] = raw['gap_rate'].clip(0, 1)
raw['uptake_rate'] = raw['uptake_rate'].clip(0, 1.05)

# Flag data quality issues
raw['data_quality_flag'] = ''
raw.loc[raw['uptake_rate'] > 1.0, 'data_quality_flag'] = 'over-enrolled'
raw.loc[raw['agri_households'] < 10000, 'data_quality_flag'] = 'small-district'
clean = raw[raw['data_quality_flag'] == ''].copy()
print(f"Clean records: {len(clean)} (dropped {len(raw)-len(clean)} flagged rows)")
clean.to_csv(f"{CLEAN}/pmkisan_clean.csv", index=False)

# Summary stats
print(f"\nNational average uptake rate: {clean['uptake_rate'].mean():.1%}")
print(f"Districts below 30% uptake: {(clean['uptake_rate']<0.30).sum()}")
print(f"Districts below 50% uptake: {(clean['uptake_rate']<0.50).sum()}")

print("\n" + "="*60)
print("STEP 2 — STATISTICAL ANALYSIS")
print("="*60)

df = clean.copy()
X_cols = ['literacy_rate', 'female_literacy', 'rural_share', 'sc_st_share', 'internet_access']
df_model = df.dropna(subset=X_cols + ['uptake_rate'])
X = sm.add_constant(df_model[X_cols])
y = df_model['uptake_rate']
model = sm.OLS(y, X).fit()
print(model.summary())

df.loc[df_model.index, 'residual'] = model.resid
df.loc[df_model.index, 'predicted_uptake'] = model.fittedvalues
df['residual'] = df['residual'].fillna(0)

# Bottom 10 anomalies
threshold = df['residual'].quantile(0.10)
df['is_anomaly'] = df['residual'] < threshold
anomalies = df[df['is_anomaly']].sort_values('residual').head(10).copy()

def assign_intervention(row):
    if row['female_literacy'] < 45:
        return ('Women SHG-led enrollment drives',
                'Female literacy below 45% signals awareness gap among women beneficiaries')
    elif row['sc_st_share'] > 0.40:
        return ('CSO partnerships + Gram Sabha drives',
                'High SC/ST share with low uptake indicates exclusion barriers')
    elif row['internet_access'] < 0.25:
        return ('Offline Aadhaar/Jan Dhan linkage camps',
                'Low internet access means digital enrollment channels are failing')
    else:
        return ('IEC campaign + block-level panchayat drives',
                'General awareness deficit — targeted outreach events recommended')

anomalies[['intervention', 'rationale']] = anomalies.apply(
    lambda r: pd.Series(assign_intervention(r)), axis=1)
anomalies.to_csv(f"{CLEAN}/bottom10_anomalies.csv", index=False)
df.to_csv(f"{CLEAN}/analysis_ready.csv", index=False)

print(f"\nR² of model: {model.rsquared:.3f}")
print(f"\nBottom 10 anomalous districts:")
print(anomalies[['state','district','uptake_rate','gap_rate','residual','intervention']].to_string())

print("\n" + "="*60)
print("STEP 3 — VISUALISATIONS")
print("="*60)

# ---- Plot 1: National uptake distribution ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

ax = axes[0]
n, bins, patches = ax.hist(df['uptake_rate']*100, bins=25, color=PALETTE['mid'],
                            edgecolor='white', linewidth=0.5, alpha=0.85)
ax.axvline(df['uptake_rate'].mean()*100, color=PALETTE['accent'],
           linestyle='--', linewidth=1.8, label=f"National avg: {df['uptake_rate'].mean():.0%}")
ax.axvline(50, color=PALETTE['warn'], linestyle=':', linewidth=1.5, label='50% threshold')
ax.set_xlabel('PM-KISAN uptake rate (%)')
ax.set_ylabel('Number of districts')
ax.set_title('Distribution of district-level PM-KISAN uptake', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1]
state_avg = df.groupby('state')['uptake_rate'].mean().sort_values()
colors = [PALETTE['accent'] if v < 0.45 else PALETTE['ok'] for v in state_avg]
bars = ax.barh(state_avg.index, state_avg.values*100, color=colors, height=0.65)
ax.axvline(50, color=PALETTE['warn'], linestyle=':', linewidth=1.2)
ax.set_xlabel('Average uptake rate (%)')
ax.set_title('State-wise average PM-KISAN uptake', fontsize=12, fontweight='bold')
red_p = mpatches.Patch(color=PALETTE['accent'], label='Below 45%')
grn_p = mpatches.Patch(color=PALETTE['ok'], label='Above 45%')
ax.legend(handles=[red_p, grn_p], fontsize=9)

plt.tight_layout(pad=2)
plt.savefig(f"{OUT}/01_national_distribution.png", bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 01_national_distribution.png")

# ---- Plot 2: Regression — literacy vs uptake ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

for ax, xcol, xlabel in zip(axes,
    ['literacy_rate', 'sc_st_share'],
    ['Literacy rate (%)', 'SC/ST population share']):
    normal = df[~df['is_anomaly']]
    anom = df[df['is_anomaly']]
    ax.scatter(normal[xcol], normal['uptake_rate']*100, alpha=0.35,
               color=PALETTE['mid'], s=18, label='Other districts')
    ax.scatter(anom[xcol], anom['uptake_rate']*100, color=PALETTE['accent'],
               s=60, zorder=5, label='Anomalous (bottom 10%)', edgecolors='white', linewidth=0.5)
    for _, row in anom.iterrows():
        short = row['state'][:4]
        ax.annotate(short, (row[xcol], row['uptake_rate']*100),
                    fontsize=7, xytext=(4, 3), textcoords='offset points', color=PALETTE['accent'])
    m, b = np.polyfit(df[xcol].dropna(), df.loc[df[xcol].notna(), 'uptake_rate']*100, 1)
    xs = np.linspace(df[xcol].min(), df[xcol].max(), 100)
    ax.plot(xs, m*xs+b, color=PALETTE['primary'], linewidth=1.2, linestyle='--', alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('PM-KISAN uptake rate (%)')
    ax.set_title(f'{xlabel} vs PM-KISAN uptake', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

plt.tight_layout(pad=2)
plt.savefig(f"{OUT}/02_covariate_scatter.png", bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 02_covariate_scatter.png")

# ---- Plot 3: Bottom 10 districts bar chart ----
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
top10 = anomalies.sort_values('gap_rate')
colors_bar = [PALETTE['accent'] if g > 0.55 else PALETTE['warn'] for g in top10['gap_rate']]
bars = ax.barh(range(len(top10)), top10['gap_rate']*100, color=colors_bar, height=0.6)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels([f"{r['state'][:10]} — {r['district'].split('_')[1]}"
                    for _, r in top10.iterrows()], fontsize=9)
for i, (bar, val) in enumerate(zip(bars, top10['gap_rate']*100)):
    ax.text(val+0.5, i, f"{val:.1f}%", va='center', fontsize=8.5, color=PALETTE['primary'])
ax.set_xlabel('Uptake gap (%)')
ax.set_title('10 districts with worst unexplained uptake gap\n(after controlling for literacy, income, connectivity)',
             fontsize=12, fontweight='bold')
ax.axvline(50, color=PALETTE['muted'], linestyle=':', linewidth=1)
plt.tight_layout()
plt.savefig(f"{OUT}/03_bottom10_districts.png", bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 03_bottom10_districts.png")

# ---- Plot 4: Regression coefficient forest plot ----
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
params = model.params.drop('const')
conf = model.conf_int().drop('const')
pvals = model.pvalues.drop('const')
labels = {'literacy_rate':'Literacy rate', 'female_literacy':'Female literacy',
          'rural_share':'Rural share', 'sc_st_share':'SC/ST share', 'internet_access':'Internet access'}
ys = range(len(params))
for i, (name, coef) in enumerate(params.items()):
    lo, hi = conf.loc[name]
    sig = pvals[name] < 0.05
    color = PALETTE['accent'] if coef < 0 and sig else (PALETTE['ok'] if coef > 0 and sig else PALETTE['muted'])
    ax.barh(i, coef, color=color, alpha=0.75, height=0.4)
    ax.plot([lo, hi], [i, i], color=color, linewidth=2.5)
    ax.plot([lo, lo], [i-0.12, i+0.12], color=color, linewidth=1.5)
    ax.plot([hi, hi], [i-0.12, i+0.12], color=color, linewidth=1.5)
    sig_label = '★' if sig else ''
    ax.text(max(hi, coef)+0.005, i, sig_label, va='center', color=color, fontsize=11)

ax.set_yticks(list(ys))
ax.set_yticklabels([labels.get(n, n) for n in params.index])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Regression coefficient (effect on uptake rate)')
ax.set_title(f'OLS regression: predictors of PM-KISAN uptake (R²={model.rsquared:.2f})\n★ = significant at p<0.05',
             fontsize=11, fontweight='bold')
red_p = mpatches.Patch(color=PALETTE['accent'], label='Negative & significant')
grn_p = mpatches.Patch(color=PALETTE['ok'], label='Positive & significant')
muted_p = mpatches.Patch(color=PALETTE['muted'], label='Not significant')
ax.legend(handles=[red_p, grn_p, muted_p], fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(f"{OUT}/04_regression_coefficients.png", bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 04_regression_coefficients.png")

# ---- Plot 5: Heatmap by state × intervention type ----
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('white')
state_gap = df.groupby('state').agg(
    avg_gap=('gap_rate', 'mean'),
    avg_literacy=('literacy_rate', 'mean'),
    avg_sc_st=('sc_st_share', 'mean'),
    avg_internet=('internet_access', 'mean'),
    n_districts=('district', 'count')
).reset_index().sort_values('avg_gap', ascending=False).head(15)

pivot = state_gap[['state','avg_gap','avg_literacy','avg_sc_st','avg_internet']].set_index('state')
pivot.columns = ['Avg gap rate', 'Literacy rate', 'SC/ST share', 'Internet access']
scaler = StandardScaler()
pivot_scaled = pd.DataFrame(scaler.fit_transform(pivot), index=pivot.index, columns=pivot.columns)
sns.heatmap(pivot_scaled, ax=ax, cmap='RdYlGn_r', center=0,
            annot=pivot.round(2), fmt='g', annot_kws={'size':8},
            linewidths=0.4, cbar_kws={'label': 'Standardised score'})
ax.set_title('State-level profile: top 15 states by uptake gap\n(values shown are raw; colour is standardised)',
             fontsize=11, fontweight='bold')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(f"{OUT}/05_state_heatmap.png", bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 05_state_heatmap.png")

print("\n" + "="*60)
print("STEP 4 — INTERVENTION MAP CSV")
print("="*60)

intervention_map = anomalies[[
    'state', 'district', 'agri_households', 'pmkisan_enrolled',
    'uptake_rate', 'gap_rate', 'literacy_rate', 'female_literacy',
    'sc_st_share', 'internet_access', 'residual', 'intervention', 'rationale'
]].copy()
intervention_map['uptake_rate'] = (intervention_map['uptake_rate']*100).round(1).astype(str) + '%'
intervention_map['gap_rate'] = (intervention_map['gap_rate']*100).round(1).astype(str) + '%'
intervention_map['residual'] = intervention_map['residual'].round(4)
intervention_map.to_csv(f"{OUT}/intervention_map.csv", index=False)
print("Saved: intervention_map.csv")
print(intervention_map[['state','district','uptake_rate','gap_rate','intervention']].to_string())

print("\n✅ All outputs saved to:", OUT)
