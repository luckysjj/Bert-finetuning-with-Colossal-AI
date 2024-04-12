import matplotlib.pyplot as plt
import numpy as np

# Data given by the user, expanded to include batch sizes
data = {
    '8': {
        'torch_ddp': {'accuracy': 0.8481159420289855, 'f1': 0.8889830508474577, 'loss': 0.4626009166240692},
        'torch_ddp_fp16': {'accuracy': 0.8423188405797102, 'f1': 0.8863826232247285, 'loss': 0.4727810025215149},
        'gemini': {'accuracy': 0.8527536231884058, 'f1': 0.8924640135478409, 'loss': 0.0},
        'low_level_zero': {'accuracy': 0.8527536231884058, 'f1': 0.8924640135478409, 'loss': 0.44024258852005005}
    },
    '16': {
        'torch_ddp': {'accuracy': 0.8359420289855073, 'f1': 0.8817384036773924, 'loss': 0.4601142704486847},
        'torch_ddp_fp16': {'accuracy': 0.8318840579710145, 'f1': 0.8788638262322475, 'loss': 0.4753853380680084},
        'gemini': {'accuracy': 0.8347826086956521, 'f1': 0.881398252184769, 'loss': 0.0},
        'low_level_zero': {'accuracy': 0.8347826086956521, 'f1': 0.881398252184769, 'loss': 0.46357882022857666}
    },
    '32': {
        'torch_ddp': {'accuracy': 0.8336231884057971, 'f1': 0.8806652806652807, 'loss': 0.4085160493850708},
        'torch_ddp_fp16': {'accuracy': 0.8394202898550724, 'f1': 0.883662326753465, 'loss': 0.40064218640327454},
        'gemini': {'accuracy': 0.8371014492753623, 'f1': 0.882770129328327, 'loss': 0.0},
        'low_level_zero': {'accuracy': 0.8371014492753623, 'f1': 0.882770129328327, 'loss': 0.39939484000205994}
    },
    '64': {
        'torch_ddp': {'accuracy': 0.8266666666666667, 'f1': 0.8748430305567183, 'loss': 0.41784197092056274},
        'torch_ddp_fp16': {'accuracy': 0.8272463768115942, 'f1': 0.8754180602006689, 'loss': 0.41678813099861145},
        'gemini': {'accuracy': 0.8231884057971014, 'f1': 0.8721174004192872, 'loss': 0.0},
        'low_level_zero': {'accuracy': 0.8231884057971014, 'f1': 0.8721174004192872, 'loss': 0.41884133219718933}
    }
}

# Plotting
batch_sizes = list(data.keys())
metrics = ['accuracy', 'f1', 'loss']
colors = ['b', 'r', 'g']
width = 0.1  # the width of the bars

fig, axs = plt.subplots(len(batch_sizes), figsize=(10, 15))

for i, batch_size in enumerate(batch_sizes):
    plugins = list(data[batch_size].keys())
    x = range(len(plugins))
    
    for j, metric in enumerate(metrics):
        values = [data[batch_size][plugin][metric] for plugin in plugins]
        axs[i].bar([p + width*j for p in x], values, width, label=metric.capitalize(), color=colors[j])
    
    # Labeling and visual adjustments for subplots
    axs[i].set_xlabel('Plugins')
    axs[i].set_ylabel('Metrics')
    axs[i].set_title(f'Comparison of Plugins for Batch Size: {batch_size}')
    axs[i].set_xticks([p + width for p in x])
    axs[i].set_xticklabels(plugins)
    axs[i].legend()
    axs[i].grid(True)

# General plot adjustments
plt.tight_layout()
plt.show()
fig.savefig('./results/finetune.png', format='png', dpi=300)  # Save as PNG with high resolution

plt.close(fig)

# Creating a new plot for comparing accuracies
accuracies_per_batch = {batch_size: [plugin_data['accuracy'] for plugin_data in data[batch_size].values()] for batch_size in batch_sizes}
plugins = list(data['8'].keys())

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 0.1
index = np.arange(len(plugins))
colors = ['b', 'r', 'g', 'a']

for i, (batch_size, accuracies) in enumerate(accuracies_per_batch.items()):
    ax.bar(index + i * bar_width, accuracies, bar_width, label=f'Batch Size {batch_size}')

# Comparing the accuracy of different plugins across various batch sizes.
ax.set_xlabel('Plugins')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison Across Batch Sizes')
ax.set_xticks(index + bar_width / 2 * (len(batch_sizes)-1))
ax.set_xticklabels(plugins)
ax.legend()
ax.grid(True)

# Display the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
fig.savefig('./results/benchmark.png', format='png', dpi=300)  # Save as PNG with high resolution

plt.close(fig)