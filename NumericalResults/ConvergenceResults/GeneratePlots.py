import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess  
import os
import argparse


def getPlot(Convdf, methodslist, x, y, title, plottype=None):

      # plotting types
    if plottype == "loglog":
        convrates = []
        for method in methodslist:
            Conv = np.polyfit(np.log(Convdf[method][x].to_numpy()), np.log(Convdf[method][y].to_numpy()), 1)
            convrates.append(Conv[0])

        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methodslist):
            plt.loglog(Convdf[method][x].to_numpy(), Convdf[method][y].to_numpy(), label=f'{method} Convergence Rate: {convrates[i]:.2f}', marker='o')
   
   
    elif plottype == "semilog":
        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methodslist):
            plt.semilogx(Convdf[method][x].to_numpy(), Convdf[method][y].to_numpy(), marker='o')

    else:
        plt.figure(figsize=(10, 6))
        for i, scheme in enumerate(schemes):
            plt.plot(Convdf[method][x].to_numpy(), Convdf[method][y].to_numpy(), marker='o')


    plt.xlabel(x, fontsize=16)
    plt.ylabel(y, fontsize=16)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Ensure the Plots directory exists
    results_dir = "Plots"
    os.makedirs(results_dir, exist_ok=True)

    # Construct the filename and save the plot
    concattitle = ""
    for method in methodslist:
        concattitle += method + "_"
        
    file_title = f"{concattitle}{x}_vs_{y}.png".replace(" ", "_")
    plt.savefig(os.path.join(results_dir, file_title))






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
    #os.chdir("/home/stefano/Desktop/VI-AMR/NumericalResults/ConvergenceResults")
    methodlist = ['vces', 'udo', 'metricIso', 'vcesUnif', 'udoUnif', 'metricIsoHess', 'vcesBR', 'udoBR', 'uniform']


    # flag for running convergence script
    parser = argparse.ArgumentParser(
        description="Script to run Convergence with various parameters")
    parser.add_argument('--runconvergence', action='store_true',
                        help='Run convergence script for all amr_methods and refinements.')
    args = parser.parse_args()

    methodlist = ['vces', 'udo', 'metricIso', 'vcesUnif','udoUnif', 'metricIsoHess', 'vcesBR', 'udoBR', 'uniform']
    print(args.runconvergence)
    
    if args.runconvergence:
        script_path = "Convergence.py"
        for method in methodlist:
            # Construct the command with arguments
            cmd = ["python3", script_path,"-m", method]

            # Print the command to be executed
            print(f"Running command: {' '.join(cmd)}")

            # Call the script using subprocess
            result = subprocess.run(cmd, stdout=True)

    # Generate Convergence dataframe
    current_dir = os.getcwd()
    results_dir = os.path.join(current_dir, "Results")
    methodlist = [os.path.splitext(filename)[0] for filename in os.listdir(results_dir)]
    Convdf = create_multiindex_dataframe(results_dir, methodlist)

    # Compute things on the dataframe
    for method in methodlist:
        Convdf[method, 'TotalTime'] = Convdf[method]['MeshTime'] + Convdf[method]['SolveTime']
        Convdf[method, 'MeshTime/Elements'] = Convdf[method]['MeshTime']/Convdf[method]['Elements']
        Convdf[method, 'SolveTime/Elements'] = Convdf[method]['SolveTime']/Convdf[method]['Elements']
        Convdf[method, 'TotalTime/Elements'] = Convdf[method]['TotalTime']/Convdf[method]['Elements']
    Convdf = Convdf.sort_index(axis=1)
    
    
    getPlot(Convdf, ['vces', 'vcesUnif', 'uniform'],
            'Elements', 'Hausdorff', 'VCES', plottype='loglog')
    getPlot(Convdf, ['udo', 'udoUnif', 'uniform'], 'Elements', 'Hausdorff', 'UDO', plottype='loglog')
    
    getPlot(Convdf, ['vces', 'vcesUnif', 'uniform'], 'Elements', 'Jaccard', 'VCES', plottype='loglog')
    getPlot(Convdf, ['udo', 'udoUnif', 'uniform'], 'Elements','Jaccard', 'UDO', plottype='loglog')

    getPlot(Convdf, ['vces', 'vcesUnif', 'uniform'],'Elements', 'L2', 'VCES', plottype='loglog')
    getPlot(Convdf, ['udo', 'udoUnif', 'uniform'],'Elements', 'L2', 'UDO', plottype='loglog')
    
    getPlot(Convdf, ['vces', 'vcesBR', 'metricIso', 'metricIsoHess'], 'MeshTime', 'L2', 'Meshing Time Comparison for Compiled Codes', plottype='loglog')
    getPlot(Convdf, ['vces', 'vcesBR', 'metricIso', 'metricIsoHess'], 'SolveTime', 'L2', 'Solving Time Comparison for Compiled Codes', plottype='loglog')


    getPlot(Convdf, ['vces', 'vcesBR', 'metricIso', 'metricIsoHess'], 'MeshTime', 'Jaccard', 'Meshing Time Comparison for Compiled Codes', plottype='loglog')
    getPlot(Convdf, ['vces', 'vcesBR', 'metricIso', 'metricIsoHess'], 'SolveTime', 'Jaccard', 'Solving Time Comparison for Compiled Codes', plottype='loglog')
    
    


    
