import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
})

models = ['Gemma2:9b', 'Qwen2.5:7b', 'Mistral:7b', 'LLaMA3.1:8b']

results = {
    'Gemma2:9b':    {'No CB': [90.9,65.7,52.0], 'CB v1': [92.7,80.4,63.6], 'CoT': [49.0,30.9,19.6], 'ICL': [94.7,71.5,57.3], 'CB v2': [96.0,86.9,71.8]},
    'Qwen2.5:7b':   {'No CB': [88.2,62.9,50.4], 'CB v1': [89.5,76.7,57.9], 'CoT': [33.8,13.7,6.1],  'ICL': [90.3,64.9,51.4], 'CB v2': [90.0,79.9,61.0]},
    'Mistral:7b':   {'No CB': [91.3,54.4,40.9], 'CB v1': [89.2,79.7,55.5], 'CoT': [66.4,50.3,29.6], 'ICL': [86.3,62.5,44.4], 'CB v2': [90.0,79.4,61.7]},
    'LLaMA3.1:8b':  {'No CB': [86.8,55.9,39.5], 'CB v1': [90.8,75.8,52.6], 'CoT': [43.0,22.4,11.8], 'ICL': [88.9,59.6,42.9], 'CB v2': [89.5,76.8,58.7]},
}

baselines = {
    'GPT-3.5':  [90.1, 66.2, 40.9],
    'GPT-4':    [93.4, 76.7, 61.5],
    'ZSP Tree': [96.4, 89.6, 82.4],
}

colors = {'Gemma2:9b': '#2563EB', 'Qwen2.5:7b': '#059669', 'Mistral:7b': '#D97706', 'LLaMA3.1:8b': '#DC2626'}
levels = ['Binary', 'Quadcode', 'Rootcode']


def add_bar_labels(ax, bars, vals, fontsize=7.5):
    positions = []
    for bar, val in zip(bars, vals):
        cx = bar.get_x() + bar.get_width() / 2
        cy = bar.get_height()
        positions.append((cx, cy, val))

    for cx, cy, val in positions:
        ax.text(cx, cy + 0.8, f'{val:.1f}', ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold', rotation=90)


# FIGURE 1

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
methods = ['No CB', 'CB v1', 'ICL', 'CB v2']
w = 0.19
ylims = [(82, 100), (48, 95), (32, 80)]

for i, level in enumerate(levels):
    ax = axes[i]
    x = np.arange(len(methods))

    for j, model in enumerate(models):
        vals = [results[model][m][i] for m in methods]
        offset = (j - 1.5) * w
        bars = ax.bar(x + offset, vals, w * 0.88, label=model if i == 0 else None,
                      color=colors[model], alpha=0.88, edgecolor='white', linewidth=0.5)
        add_bar_labels(ax, bars, vals)

    ax.axhline(baselines['GPT-4'][i], color='#6B7280', ls='--', lw=1.2, alpha=0.7,
               label='GPT-4' if i == 0 else None)
    ax.axhline(baselines['ZSP Tree'][i], color='#111827', ls='-', lw=1.5, alpha=0.6,
               label='ZSP Tree' if i == 0 else None)

    ax.set_title(level, fontsize=15, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Macro F1' if i == 0 else '')
    ax.set_ylim(ylims[i][0], ylims[i][1])
    ax.grid(axis='y', alpha=0.25, lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].legend(fontsize=9, loc='lower left', framealpha=0.95, edgecolor='#D1D5DB')
plt.tight_layout(w_pad=2)
plt.savefig('/home/cc/codebook_eval/plots/plover_plots/fig1_main_results.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/cc/codebook_eval/plots/plover_plots/fig1_main_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("fig1 done")


# FIGURE 2

fig, ax = plt.subplots(figsize=(9, 5.5))

systems = {m + ' (CB v2)': results[m]['CB v2'] for m in models}
systems.update(baselines)
names = list(systems.keys())
sys_colors = [colors[m] for m in models] + ['#9CA3AF', '#6B7280', '#1F2937']

x = np.arange(3)
w = 0.12

for s, (name, vals) in enumerate(systems.items()):
    offset = (s - (len(names)-1)/2) * w
    bars = ax.bar(x + offset, vals, w * 0.88, label=name, color=sys_colors[s],
                  alpha=0.88, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(levels, fontsize=13)
ax.set_ylabel('Macro F1', fontsize=12)
ax.set_ylim(35, 105)
ax.set_title('Codebook v2 Performance vs Paper Baselines', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=8.5, loc='lower left', framealpha=0.95, edgecolor='#D1D5DB', ncol=2)
ax.grid(axis='y', alpha=0.25, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/cc/codebook_eval/plots/plover_plots/fig2_cbv2_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/cc/codebook_eval/plots/plover_plots/fig2_cbv2_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("fig2 done")


# FIGURE 3

fig, ax = plt.subplots(figsize=(10, 5.5))
method_order = ['No CB', 'ICL', 'CB v1', 'CB v2', 'CoT']
x = np.arange(len(method_order))
w = 0.19

for j, model in enumerate(models):
    vals = [results[model][m][2] for m in method_order]
    offset = (j - 1.5) * w
    bars = ax.bar(x + offset, vals, w * 0.88, label=model,
                  color=colors[model], alpha=0.88, edgecolor='white', linewidth=0.5)
    add_bar_labels(ax, bars, vals)

ax.axhline(61.5, color='#6B7280', ls='--', lw=1.2, alpha=0.7, label='GPT-4 (61.5)')
ax.axhline(82.4, color='#111827', ls='-', lw=1.5, alpha=0.6, label='ZSP Tree (82.4)')

ax.axvline(3.5, color='#E5E7EB', ls='-', lw=1.5)
ax.text(4, 83, 'CoT\n(failure mode)', ha='center', va='bottom', fontsize=9,
        color='#9CA3AF', fontstyle='italic')

ax.set_xticks(x)
ax.set_xticklabels(method_order, fontsize=11)
ax.set_ylabel('Rootcode Macro F1', fontsize=12)
ax.set_title('Rootcode F1 by Prompting Method', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=9, loc='upper left', framealpha=0.95, edgecolor='#D1D5DB', ncol=3)
ax.set_ylim(0, 95)
ax.grid(axis='y', alpha=0.25, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/cc/codebook_eval/plots/plover_plots/fig3_root_by_method.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/cc/codebook_eval/plots/plover_plots/fig3_root_by_method.png', dpi=300, bbox_inches='tight')
plt.close()
print("fig3 done")


# FIGURE 4

fig, ax = plt.subplots(figsize=(7.5, 4))
all_methods = ['No CB', 'CB v1', 'CoT', 'ICL', 'CB v2']
data = np.array([[results[m][method][2] for method in all_methods] for m in models])

im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=85)

ax.set_xticks(np.arange(len(all_methods)))
ax.set_xticklabels(all_methods, fontsize=11)
ax.set_yticks(np.arange(len(models)))
ax.set_yticklabels(models, fontsize=11)

for i in range(len(models)):
    for j in range(len(all_methods)):
        val = data[i, j]
        color = 'white' if val < 25 else 'black'
        bold = 'bold' if val == data[:, j].max() else 'normal'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                fontsize=12, color=color, fontweight=bold)

ax.set_title('Rootcode Macro F1', fontsize=14, fontweight='bold', pad=12)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Macro F1', fontsize=10)

plt.tight_layout()
plt.savefig('/home/cc/codebook_eval/plots/plover_plots/fig4_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/cc/codebook_eval/plots/plover_plots/fig4_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("fig4 done")

print("\nAll figures saved.")