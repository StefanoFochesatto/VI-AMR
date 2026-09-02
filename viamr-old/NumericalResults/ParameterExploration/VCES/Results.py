import matplotlib.ticker as ticker
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import os

current_directory = os.getcwd()
file_name = 'VCESSphere.py'
file_path = os.path.join(current_directory, file_name)


for n in range(4):
    # Calculate the threshold values
    lower_thresh = float('{:.2g}'.format(.4 - n * .1))
    upper_thresh = float('{:.2g}'.format(.6 + n * .1))
    # Construct the command with the current values of `--thresh`
    command = [
        "python3", file_path, "--thresh", str(lower_thresh), str(upper_thresh)
    ]
    # Execute the command
    subprocess.run(command)

# Initialize an empty DataFrame
# List of filenames to read
filenames = [
    'VCESwith([0.4, 0.6]).csv',
    'VCESwith([0.3, 0.7]).csv',
    'VCESwith([0.2, 0.8]).csv',
    'VCESwith([0.1, 0.9]).csv'
]

collated_df = pd.DataFrame()

# Loop over the filenames
for filename in filenames:
    # Read the CSV file into a DataFrame
    data = pd.read_csv(filename)

    # Extract the numbers from the filename to use as column header
    header = filename.split('[')[1].split(']')[0]  # Extract "0.1, 0.9" part
    collated_df[header] = data['NumCells']


# Assuming collated_df is already defined and populated with data

# Plotting
fig, ax = plt.subplots()

# Generate the overlayed bar graph using a specific colormap (e.g., 'viridis')
bar_width = 0.8  # Adjust the bar width to be wider
collated_df.T.plot(kind='bar', stacked=False, ax=ax,
                   colormap='viridis', width=bar_width)

# Customize the plot
ax.set_title('Number of Elements vs Threshold Parameters')
ax.set_xlabel('$(\\alpha, \\beta)$')
ax.set_ylabel('Number of Elements')

# Set x-ticks to be the range of columns (headers) in collated_df
ax.set_xticks(range(len(collated_df.columns)))
ax.set_xticklabels(collated_df.columns)

# Rotate the x-axis labels
plt.xticks(rotation=45)

# Use scientific notation for the y-axis
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

# Enable the grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Customize the legend
plt.legend(title='Mesh #', loc='upper left', bbox_to_anchor=(0, 1))

plt.tight_layout()

# Show the plot
plt.show()
