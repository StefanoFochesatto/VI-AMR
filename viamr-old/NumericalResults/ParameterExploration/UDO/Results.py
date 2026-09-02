import matplotlib.ticker as ticker
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import os

current_directory = os.getcwd()
file_name = 'UDOSphere.py'
file_path = os.path.join(current_directory, file_name)

# Loop over values from 1 to 5 and execute the script with -n
for n in range(1, 6):
    # Construct the command with the current value of n
    command = ["python3", file_path, "--n", str(n)]
    subprocess.run(command)


# Initialize an empty DataFrame
collated_df = pd.DataFrame()

# Loop over the range of n values
for n in [1, 2, 3, 4, 5]:
    # Construct the filename string
    filename = f'UDOwith({n}).csv'

    # Read the CSV file into a DataFrame
    data = pd.read_csv(filename)

    # Add the 'NumCells' column to the main DataFrame as a new column with heading n
    collated_df[n] = data['NumCells']


# Assuming collated_df is already defined and populated with data

# Plotting
fig, ax = plt.subplots()

# Generate the overlayed bar graph using a specific colormap (e.g., 'viridis')
bar_width = 0.8  # Adjust the bar width to be wider
collated_df.T.plot(kind='bar', stacked=False, ax=ax,
                   colormap='viridis', width=bar_width)

# Customize the plot
ax.set_title('Number of Elements vs Neighborhood Depth')
ax.set_xlabel('$n$')
ax.set_ylabel('Number of Elements')

# Set x-ticks to be the range of n values
ax.set_xticks(range(len(collated_df.columns)))
ax.set_xticklabels(collated_df.columns)

# Rotate the x-axis labels to 90 degrees
plt.xticks(rotation=0)

# Use scientific notation for the y-axiss
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

# Enable the grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Customize the legend
plt.legend(title='Mesh #', loc='upper left', bbox_to_anchor=(0, 1))

plt.tight_layout()

# Show the plot
plt.show()
