import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.pyplot as plt
import os

# Change to the base directory
base_dir = "/home/stefano/Desktop/stefano-assist/ParameterExploration"

# Initialize DataFrames for UDO and VCES
udo_collated_df = pd.DataFrame()
vces_collated_df = pd.DataFrame()

# Load UDO data
os.chdir(os.path.join(base_dir, "UDO"))
udo_filenames = [f'UDOwith({n}).csv' for n in [1, 2, 3, 4, 5]]
for n, filename in zip([1, 2, 3, 4, 5], udo_filenames):
    data = pd.read_csv(filename)
    udo_collated_df[n] = data['NumCells']

# Load VCES data
os.chdir(os.path.join(base_dir, "VCES"))
vces_filenames = [
    'VCESwith([0.4, 0.6]).csv',
    'VCESwith([0.3, 0.7]).csv',
    'VCESwith([0.2, 0.8]).csv',
    'VCESwith([0.1, 0.9]).csv'
]
for filename in vces_filenames:
    data = pd.read_csv(filename)
    header = filename.split('[')[1].split(']')[0]
    vces_collated_df[header] = data['NumCells']

# Calculate the maximum y-limit from both UDO and VCES data
y_max = max(udo_collated_df.max().max(),
            vces_collated_df.max().max()) * 1.1  # Add 10% buffer

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Set global font sizes
label_font_size = 14
title_font_size = 16
tick_font_size = 12

# Plot VCES data first
vces_collated_df.T.plot(kind='bar', stacked=False, ax=axs[0],
                        colormap='viridis', width=0.8)
axs[0].set_title('VCES', fontsize=title_font_size)
axs[0].set_xlabel('$(\\alpha, \\beta)$', fontsize=label_font_size)
axs[0].set_ylabel('Number of Elements', fontsize=label_font_size)
axs[0].set_xticks(range(len(vces_collated_df.columns)))
axs[0].set_xticklabels(vces_collated_df.columns, fontsize=tick_font_size)
axs[0].tick_params(axis='x', rotation=45)
axs[0].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
axs[0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
axs[0].set_ylim(0, y_max)  # Set y-axis limit using the calculated max
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[0].legend(title='Mesh #', loc='upper left', bbox_to_anchor=(0, 1))

# Adjust power label font size
axs[0].yaxis.get_offset_text().set_fontsize(label_font_size)

# Plot UDO data second
udo_collated_df.T.plot(kind='bar', stacked=False, ax=axs[1],
                       colormap='viridis', width=0.8)
axs[1].set_title('UDO', fontsize=title_font_size)
axs[1].set_xlabel('$n$', fontsize=label_font_size)
axs[1].set_ylabel('Number of Elements', fontsize=label_font_size)
axs[1].set_xticks(range(len(udo_collated_df.columns)))
axs[1].set_xticklabels(udo_collated_df.columns, fontsize=tick_font_size)
axs[1].tick_params(axis='x', rotation=0)
axs[1].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
axs[1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
axs[1].set_ylim(0, y_max)  # Set y-axis limit using the calculated max
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[1].legend(title='Mesh #', loc='upper left', bbox_to_anchor=(0, 1))

# Adjust power label font size
axs[1].yaxis.get_offset_text().set_fontsize(label_font_size)

plt.tight_layout(pad=2.0)  # Increase padding between subplots
plt.show()
