import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess  # for running convergence.py later
import os
import argparse


def getPlot(Convdf, amrtype, x, y, plottype=None):
    """Generates and displays a convergence plot for different mesh refinement schemes.
    Args:
        df (pd.DataFrame): The DataFrame containing the convergence data. It is expected
            to have a MultiIndex with levels ['amrType', 'Scheme', 'index'] where 'Scheme' identifies
            different refinement strategies (e.g., AMR, Hybrid, Unif).
        amrtype (str): A string specifying the type of adaptive mesh refinement being analyzed.
            This is used in the plot title to distinguish between different AMR methods.
        x (str): The name of the column in the DataFrame to be used as the x-axis of the plot.
        y (str): The name of the column in the DataFrame to be used as the y-axis of the plot.
        plottype (str, optional): The type of plot to generate. It can be "loglog", "semilog",
            or None (default). In "loglog", both axes are logarithmic. In "semilog", only the
            x-axis is logarithmic. If None, a simple linear plot is drawn.

    Raises:
        KeyError: If the specified x or y columns are not present in the DataFrame.

    Example:
        getPlot(Convdf, "UDO", "Elements", "L2", plottype="loglog")
    """

    # Filter the DataFrame by the specified amrtype
    df = Convdf.loc[amrtype]

    # pull schemes list from df
    schemes = df.index.get_level_values('Scheme').unique().tolist()

    # plotting types
    if plottype == "loglog":
        convrates = []
        for scheme in schemes:
            Conv = np.polyfit(np.log(df.loc[scheme, x].to_numpy()), np.log(
                df.loc[scheme, y].to_numpy()), 1)
            convrates.append(Conv[0])

        plt.figure(figsize=(10, 6))
        for i, scheme in enumerate(schemes):
            plt.loglog(df.loc[scheme, x].to_numpy(
            ), df.loc[scheme, y].to_numpy(), label=f'{scheme} Convergence Rate: {convrates[i]:.2f}', marker='o')
    elif plottype == "semilog":
        plt.figure(figsize=(10, 6))
        for i, scheme in enumerate(schemes):
            plt.semilogx(df.loc[scheme, x].to_numpy(),
                         df.loc[scheme, y].to_numpy(), marker='o')

    else:
        plt.figure(figsize=(10, 6))
        for i, scheme in enumerate(schemes):
            plt.plot(df.loc[scheme, x].to_numpy(),
                     df.loc[scheme, y].to_numpy(), marker='o')

    plt.xlabel(x, fontsize=16)
    plt.ylabel(y, fontsize=16)
    plt.title(
        f'Convergence Plot {amrtype} - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)

    # Ensure the Plots directory exists
    results_dir = "Plots"
    os.makedirs(results_dir, exist_ok=True)

    # Construct the filename and save the plot
    file_title = f"{amrtype}_{plottype}_{x}_vs_{y}.png".replace(" ", "_")
    plt.savefig(os.path.join(results_dir, file_title))


def create_multiindex_dataframe(directory):
    """Creates a MultiIndex DataFrame from CSV files in a specified directory.

    The CSV files should be named using the convention 'Scheme_amrType.csv'.

    Args:
        directory (str): The path to the directory containing the CSV files.

    Returns:
        pd.DataFrame: A concatenated DataFrame with a MultiIndex on ['amrType', 'Scheme'].
    """
    all_dfs = []

    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract Scheme and amrType from the filename
            scheme, amr_type = filename.rsplit('.', 1)[0].split('_')

            # Capitalize the scheme name for consistency
            scheme = scheme.capitalize()

            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(directory, filename))

            # Add Scheme and amrType columns
            df = df.assign(Scheme=scheme, amrType=amr_type)

            # Append to the list of DataFrames
            all_dfs.append(df)

    # Concatenate all DataFrames
    Convdf = pd.concat(all_dfs, ignore_index=True)

    # Set the new MultiIndex with amrType and Scheme
    Convdf.set_index(['amrType', 'Scheme', Convdf.index], inplace=True)

    # Sort the MultiIndex
    Convdf.sort_index(inplace=True)

    return Convdf


if __name__ == "__main__":

    # flag for running convergence script
    parser = argparse.ArgumentParser(
        description="Script to run Convergence with various parameters")
    parser.add_argument('--runconvergence', action='store_true',
                        help='Run convergence script for all amr_methods and refinements.')
    args = parser.parse_args()

    if args.runconvergence:
        # adjust flags for running convergence script, small modification for metric based adaptation
        # amr_methods = ["udo", "vces", "metric"] <- will need to adjust if statements in Convergence.py
        amr_methods = ["udo", "vces"]
        refinements = ["Uniform", "Hybrid", "Adaptive"]

        script_path = "Convergence.py"
        for amr_method in amr_methods:
            for refinement in refinements:
                # Construct the command with arguments
                cmd = [
                    "python3", script_path,
                    "-a", amr_method,
                    "-r", refinement
                ]

                # Print the command to be executed
                print(f"Running command: {' '.join(cmd)}")

                # Call the script using subprocess
                result = subprocess.run(cmd, capture_output=True, text=True)

                # Check the result and handle any errors
                if result.returncode == 0:
                    print(f"Success: {result.stdout}")
                else:
                    print(f"Error: {result.stderr}")

    current_dir = os.getcwd()
    results_dir = os.path.join(current_dir, "Results")
    Convdf = create_multiindex_dataframe(results_dir)
    print("MultiIndex DataFrame created successfully.")
    print(Convdf.head())

    # Example Plot
    getPlot(Convdf, "udo", "Elements", "L2", plottype="loglog")
    getPlot(Convdf, "vces", "Elements", "L2", plottype="loglog")
    getPlot(Convdf, "udo", "Elements", "Hausdorff", plottype="loglog")
    getPlot(Convdf, "vces", "Elements", "Hausdorff", plottype="loglog")
    getPlot(Convdf, "udo", "Elements", "Jaccard", plottype="loglog")
    getPlot(Convdf, "vces", "Elements", "Jaccard", plottype="loglog")

    # loglog Number of elements vs l2 with convergence rate
    # Number of elements vs Hausdorff distance
    # loglog Number of elements vs Hausdorff distance
    # Number of elements vs IoU
