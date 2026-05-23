import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
})

SAVE_DIR = '/home/cc/codebook_eval/plots/aw_plots'

models = ['Gemma2:9b', 'Qwen2.5:7b', 'Mistral:7b', 'LLaMA3.1:8b']

results = {
    'Gemma2:9b':   {'No CB': 75.4, 'CB v1': 78.8, 'CoT': 49.3, 'ICL': 77.4, 'CB v2': 76.1},
    'Qwen2.5:7b':  {'No CB': 74.4, 'CB v1': 78.1, 'CoT': 43.1, 'ICL': 77.4, 'CB v2': 78.9},
    'Mistral:7b':  {'No CB': 78.6, 'CB v1': 76.4, 'CoT': 68.7, 'ICL': 79.8, 'CB v2': 81.7},
    'LLaMA3.1:8b': {'No CB': 74.3, 'CB v1': 79.3, 'CoT': 44.1, 'ICL': 81.4, 'CB v2': 81.0},
}

baselines = {
    'GPT-3.5':  76.3,
    'GPT-4':    87.0,
    'ZSP Tree': 88.0,
}

colors = {'Gemma2:9b': '#2563EB', 'Qwen2.5:7b': '#059669', 'Mistral:7b': '#D97706', 'LLaMA3.1:8b': '#DC2626'}


# FIGURE 1 — All models x methods (excl CoT)

fig, ax = plt.subplots(figsize=(10, 5.5))
methods = ['No CB', 'CB v1', 'ICL', 'CB v2']
x = np.arange(len(methods))
w = 0.19

for j, model in enumerate(models):
    vals = [results[model][m] for m in methods]
    offset = (j - 1.5) * w
    bars = ax.bar(x + offset, vals, w * 0.88, label=model,
                  color=colors[model], alpha=0.88, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.axhline(baselines['GPT-4'], color='#6B7280', ls='--', lw=1.2, alpha=0.7, label='GPT-4 (87.0)')
ax.axhline(baselines['ZSP Tree'], color='#111827', ls='-', lw=1.5, alpha=0.6, label='ZSP Tree (88.0)')

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel('Binary Macro F1', fontsize=12)
ax.set_title('A/W Dataset: 4 Models × 4 Methods', fontsize=14, fontweight='bold', pad=12)
ax.set_ylim(70, 92)
ax.legend(fontsize=9, loc='lower right', framealpha=0.95, edgecolor='#D1D5DB', ncol=2)
ax.grid(axis='y', alpha=0.25, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/fig1_aw_main_results.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{SAVE_DIR}/fig1_aw_main_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("fig1 done")


# FIGURE 2 — CB v2 all models vs baselines

fig, ax = plt.subplots(figsize=(8, 5.5))

systems = {m + ' (CB v2)': results[m]['CB v2'] for m in models}
systems.update(baselines)
names = list(systems.keys())
sys_colors = [colors[m] for m in models] + ['#9CA3AF', '#6B7280', '#1F2937']

x = np.arange(len(names))
bars = ax.bar(x, list(systems.values()), 0.6, color=sys_colors, alpha=0.88,
              edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, systems.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9, rotation=25, ha='right')
ax.set_ylabel('Binary Macro F1', fontsize=12)
ax.set_title('A/W Dataset: CB v2 vs Paper Baselines', fontsize=14, fontweight='bold', pad=12)
ax.set_ylim(70, 95)
ax.grid(axis='y', alpha=0.25, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/fig2_aw_cbv2_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{SAVE_DIR}/fig2_aw_cbv2_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("fig2 done")


# FIGURE 3 — All methods including CoT

fig, ax = plt.subplots(figsize=(10, 5.5))
method_order = ['No CB', 'ICL', 'CB v1', 'CB v2', 'CoT']
x = np.arange(len(method_order))
w = 0.19

for j, model in enumerate(models):
    vals = [results[model][m] for m in method_order]
    offset = (j - 1.5) * w
    bars = ax.bar(x + offset, vals, w * 0.88, label=model,
                  color=colors[model], alpha=0.88, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=90)

ax.axhline(87.0, color='#6B7280', ls='--', lw=1.2, alpha=0.7, label='GPT-4 (87.0)')
ax.axhline(88.0, color='#111827', ls='-', lw=1.5, alpha=0.6, label='ZSP Tree (88.0)')

ax.axvline(3.5, color='#E5E7EB', ls='-', lw=1.5)
ax.text(4, 88.5, 'CoT\n(failure mode)', ha='center', va='bottom', fontsize=9,
        color='#9CA3AF', fontstyle='italic')

ax.set_xticks(x)
ax.set_xticklabels(method_order, fontsize=11)
ax.set_ylabel('Binary Macro F1', fontsize=12)
ax.set_title('A/W Dataset: Binary F1 by Prompting Method', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=9, loc='upper left', framealpha=0.95, edgecolor='#D1D5DB', ncol=3)
ax.set_ylim(35, 96)
ax.grid(axis='y', alpha=0.25, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/fig3_aw_by_method.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{SAVE_DIR}/fig3_aw_by_method.png', dpi=300, bbox_inches='tight')
plt.close()
print("fig3 done")


# FIGURE 4 — Heatmap

fig, ax = plt.subplots(figsize=(7.5, 4))
all_methods = ['No CB', 'CB v1', 'CoT', 'ICL', 'CB v2']
data = np.array([[results[m][method] for method in all_methods] for m in models])

im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=40, vmax=85)

ax.set_xticks(np.arange(len(all_methods)))
ax.set_xticklabels(all_methods, fontsize=11)
ax.set_yticks(np.arange(len(models)))
ax.set_yticklabels(models, fontsize=11)

for i in range(len(models)):
    for j in range(len(all_methods)):
        val = data[i, j]
        color = 'white' if val < 50 else 'black'
        bold = 'bold' if val == data[:, j].max() else 'normal'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                fontsize=12, color=color, fontweight=bold)

ax.set_title('A/W Dataset: Binary Macro F1', fontsize=14, fontweight='bold', pad=12)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Macro F1', fontsize=10)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/fig4_aw_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{SAVE_DIR}/fig4_aw_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("fig4 done")

print("\nAll AW figures saved.")