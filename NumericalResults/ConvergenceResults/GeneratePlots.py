import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess  
import os
import argparse


def getPlot(Convdf, methodslist, x, y, title, plottype=None, methodcolors = None,legend = True):

    # plotting types
    if plottype == "loglog":
        convrates = []
        for method in methodslist:
            Conv = np.polyfit(np.log(Convdf[method][x].to_numpy()), np.log(
                Convdf[method][y].to_numpy()), 1)
            convrates.append(Conv[0])

        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methodslist):
            plt.loglog(Convdf[method][x].to_numpy(), Convdf[method][y].to_numpy(
            ), color = methodcolors[method], label=f'{method} Convergence Rate: {convrates[i]:.2f}', marker='o')

    elif plottype == "semilogx":
        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methodslist):
            plt.semilogx(Convdf[method][x].to_numpy(),
                         Convdf[method][y].to_numpy(),color = methodcolors[method], marker='o')
    elif plottype == "semilogy":
        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methodslist):
            plt.semilogy(Convdf[method][x].to_numpy(),
                         Convdf[method][y].to_numpy(),color = methodcolors[method], marker='o')

    else:
        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methodslist):
            plt.plot(Convdf[method][x].to_numpy(),
                     Convdf[method][y].to_numpy(), color = methodcolors[method],marker='o')

    plt.xlabel(x, fontsize=16)
    plt.ylabel(y, fontsize=16)
    plt.title(title)
    if legend:
        plt.legend()
    plt.grid(True)

    # Ensure the Plots directory exists
    results_dir = "PlotsTest"
    os.makedirs(results_dir, exist_ok=True)

    # Construct the filename and save the plot
    concattitle = ""
    for method in methodslist:
        concattitle += method + "_"

    file_title = f"{concattitle}{x}_vs_{y}.png".replace(" ", "_").replace("/","_per_")
    
    return plt, file_title





def create_multiindex_dataframe(result_dir, methodlist):
    # Initialize an empty list to hold each method's DataFrame
    dfs = []
    for method in methodlist:
        filename = f"{method}.csv"
        file_path = os.path.join(result_dir, filename)
        df = pd.read_csv(file_path)

        # Create a MultiIndex for columns using the method name and original headers
        multi_cols = [(method, col) for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(multi_cols)

        dfs.append(df)

    # Concatenate all DataFrames horizontally to create the multi-index structure
    combined_df = pd.concat(dfs, axis=1)
    return combined_df

    CG1, DG0 = amr_instance.spaces(mesh)



if __name__ == "__main__":
    os.chdir("/home/stefano/Desktop/VI-AMR/NumericalResults/ConvergenceResults")
    methodlist = ['vces', 'udo', 'metricIso', 'vcesUnif', 'udoUnif', 'metricIsoHess', 'vcesBR', 'udoBR', 'uniform']


    # flag for running convergence script
    parser = argparse.ArgumentParser(
        description="Script to run Convergence with various parameters")
    parser.add_argument('--lshaped', action='store_true',
                        help='run script for lshaped domain problem')
    parser.add_argument('--runconvergence', action='store_true',
                        help='Run convergence script for all amr_methods and refinements.')
    args = parser.parse_args()

    methodlist = ['vces', 'udo', 'metricIso', 'vcesUnif','udoUnif', 'metricIsoHess', 'vcesBR', 'udoBR', 'uniform']
    print(args.runconvergence)
    
    if args.lshaped:
        script_path = "ConvergenceLShaped.py"
    else:        
        script_path = "Convergence.py"

        
    if args.runconvergence:
        for method in methodlist:
            # Construct the command with arguments
            cmd = ["python3", script_path,"-m", method]

            # Print the command to be executed
            print(f"Running command: {' '.join(cmd)}")

            # Call the script using subprocess
            result = subprocess.run(cmd, stdout=True)

    # Generate Convergence dataframe
    current_dir = os.getcwd()
    
    if args.lshaped:
        results_dir = os.path.join(current_dir, "ResultsLShaped")
    else:
        results_dir = os.path.join(current_dir, "Results")
    methodlist = [os.path.splitext(filename)[0] for filename in os.listdir(results_dir)]
    Convdf = create_multiindex_dataframe(results_dir, methodlist)

    # Compute things on the dataframe
    for method in methodlist:
        Convdf[method, 'MeshTime'] = Convdf[method]['PreMeshCompTime'] + Convdf[method]['RefineTime']
        Convdf[method, 'TotalTime'] = Convdf[method]['PreMeshCompTime'] + Convdf[method]['RefineTime'] + Convdf[method]['SolveTime']
        Convdf[method, 'PreMeshCompTime/Elements'] = Convdf[method]['PreMeshCompTime']/Convdf[method]['Elements']
        Convdf[method, 'MeshTime/Elements'] = Convdf[method]['MeshTime']/Convdf[method]['Elements']
        Convdf[method, 'SolveTime/Elements'] = Convdf[method]['SolveTime']/Convdf[method]['Elements']
        Convdf[method, 'TotalTime/Elements'] = Convdf[method]['TotalTime']/Convdf[method]['Elements']
    Convdf = Convdf.sort_index(axis=1)
    
    # Hardcoded high-saturation colors (hex codes)
    methodcolors = {
        # Group 1: metric* methods (greens)
        'metricIso': '#317256',     
        'metricIsoHess': '#52bf90', 
        
        # Group 2: vces* methods (blues)
        'vces': '#0021f3',          
        'vcesBR': '#0006b1',        
        'vcesUnif': '#05014a',      
        
        # Group 3: udo* methods (reds)
        'udo': '#440909',           
        'udoBR': '#a72b2b',         
        'udoUnif': '#d09c9c',       
        
        # Standalone methods (unique color)
        'uniform': '#9467bd',       
    }
    if args.lshaped:
        plot_dir = 'PlotsLShaped'
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = 'Plots'
        os.makedirs(plot_dir, exist_ok=True)
    
    # Individual Methods plotted compared to uniform
    plt, file_title = getPlot(Convdf, ['udo', 'udoBR', 'uniform'] , 'Elements', 'H1', 'UDO', plottype='loglog', methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))

    plt, file_title = getPlot(Convdf, ['vces', 'vcesBR', 'uniform'] , 'Elements', 'H1', 'VCES', plottype='loglog', methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))
    plt, file_title = getPlot(Convdf, ['metricIso', 'metricIsoHess', 'uniform'] , 'Elements', 'H1', 'Metric', plottype='loglog', methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))
    plt, file_title = getPlot(Convdf, ['udo', 'udoBR', 'uniform'] , 'Elements', 'Jaccard', 'UDO', plottype='loglog', methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))
    plt, file_title = getPlot(Convdf, ['vces', 'vcesBR', 'uniform'] , 'Elements', 'Jaccard', 'VCES', plottype='loglog', methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))
    plt, file_title = getPlot(Convdf, ['metricIso', 'metricIsoHess', 'uniform'] , 'Elements', 'Jaccard', 'Metric', plottype='loglog', methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))
    plt, file_title = getPlot(Convdf.iloc[1:], ['udo', 'vces', 'metricIso'] , 'Elements', 'Jaccard', 'Comparison of Best Methods WRT Jaccard', plottype='loglog', methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))
    plt, file_title = getPlot(Convdf.iloc[0:], ['udo', 'vces', 'metricIso'] , 'Elements', 'Hausdorff', 'Comparison of Best Methods WRT Hausdorff', plottype='loglog', methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))
    plt, file_title = getPlot(Convdf, ['vcesBR', 'udoBR', 'metricIsoHess'] , 'PreMeshCompTime/Elements', 'L2', 'L2 Time Effeciency of Inactive Set Methods', plottype='loglog',methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))
    plt, file_title = getPlot(Convdf.iloc[0:], ['vces','udo' ,'metricIso'] , 'PreMeshCompTime/Elements', 'Hausdorff', 'Hausdorff Time Effeciency of Inactive Set Methods', plottype='loglog',methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))
    plt, file_title = getPlot(Convdf.iloc[0:], ['vces','udo' ,'metricIso'] , 'PreMeshCompTime/Elements', 'Jaccard', 'Jaccard Time Effeciency of Inactive Set Methods', plottype='loglog',methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))
    plt, file_title = getPlot(Convdf.iloc[0:], ['vces' ,'metricIso'] , 'Elements', 'SolveTime/Elements', 'L Shaped Domain Solve Time Comparison', plottype='loglog',methodcolors=methodcolors, legend = True)
    plt.savefig(os.path.join(plot_dir, file_title))
    # Grid sequencing should make the solve time better for tag and refine methods?? Likely an issue with cross mesh interpolation?