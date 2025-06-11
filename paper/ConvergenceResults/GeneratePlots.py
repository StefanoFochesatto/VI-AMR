import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess  
import os
import argparse


def getPlot(Convdf, methodslist, x, y, title, plottype=None, methodcolors = None, methodmarkers = None, methodlabels = None, legend = True):
    ms0 = 10.0
    fs0 = 14.0
    # plotting types
    if plottype == "loglog":
        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methodslist):
            plt.loglog(Convdf[method][x].to_numpy(), 
                       Convdf[method][y].to_numpy(), color = methodcolors[method], marker=methodmarkers[method],  ms=ms0 ,label = methodlabels[method])

    elif plottype == "semilogx":
        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methodslist):
            plt.semilogx(Convdf[method][x].to_numpy(),
                         Convdf[method][y].to_numpy(),color = methodcolors[method],  marker=methodmarkers[method],  ms=ms0,label = methodlabels[method])
            
    elif plottype == "semilogy":
        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methodslist):
            plt.semilogy(Convdf[method][x].to_numpy(),
                         Convdf[method][y].to_numpy(),color = methodcolors[method],  marker=methodmarkers[method],  ms=ms0,label = methodlabels[method])

    else:
        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methodslist):
            plt.plot(Convdf[method][x].to_numpy(),
                     Convdf[method][y].to_numpy(), color = methodcolors[method],  marker=methodmarkers[method],  ms=ms0,label = methodlabels[method])

    plt.xlabel(x, fontsize=16)
    plt.ylabel(y, fontsize=16)
    plt.title(title)
    if legend:
        plt.legend(fontsize=fs0)
    plt.xlim(7.0e1,1.0e7)
        
    plt.grid(True)


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
    os.chdir("/home/stefano/Desktop/VI-AMR/paper/ConvergenceResults")
    methodlist = ['vcd', 'udo', 'metricIso', 'metricIsoHess', 'vcdBR', 'udoBR', 'uniform']


    # flag for running convergence script
    parser = argparse.ArgumentParser(
        description="Script to run Convergence with various parameters")
    parser.add_argument('--lshaped', action='store_true',
                        help='run script for lshaped domain problem')
    parser.add_argument('--runconvergence', action='store_true',
                        help='Run convergence script for all amr_methods and refinements.')
    args = parser.parse_args()

    methodlist = ['vcd', 'udo', 'metricIso', 'metricIsoHess', 'vcdBR', 'udoBR', 'uniform']
    
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
        
        # Group 2: vcd* methods (blues)
        'vcd': '#0021f3',          
        'vcdBR': '#0006b1',        
        'vcdUnif': '#05014a',      
        
        # Group 3: udo* methods (reds)
        'udo': '#440909',           
        'udoBR': '#a72b2b',         
        'udoUnif': '#d09c9c',       
        
        # Standalone methods (unique color)
        'uniform': '#9467bd',       
    }
    
    
    methodmarkers = {
        # Group 1: metric* methods (circles/squares)
        'metricIso': 'o',          
        'metricIsoHess': 's',      
        
        # Group 2: vcd* methods (triangles/crosses)
        'vcd': 'o',                
        'vcdBR': 'x',              
        'vcdUnif': 'p',            
        
        # Group 3: udo* methods (diamonds/triangles)
        'udo': 'ko',                
        'udoBR': 'D',              
        'udoUnif': '>',            
        
        # Standalone method (hexagon)
        'uniform': 'p',            
    }
    
    
    methodlabels = {
        # Group 1: metric* methods (circles/squares)
        'metricIso': r'AVM $\gamma = 1$',          
        'metricIsoHess': r'AVM $\gamma = .5$',      
        
        # Group 2: vcd* methods (triangles/crosses)
        'vcd': 'VCD',                
        'vcdBR': 'VCD + BR',              
        'vcdUnif': 'p',            
        
        # Group 3: udo* methods (diamonds/triangles)
        'udo': 'UDO',                
        'udoBR': 'UDO + BR',              
        'udoUnif': '>',            
        
        # Standalone method (hexagon)
        'uniform': 'UNIFORM',            
    }
    
    
    
    if args.lshaped:
        plot_dir = 'PlotsLShaped'
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = 'Plots'
        os.makedirs(plot_dir, exist_ok=True)
    
    # Individual Methods plotted compared to uniform
        
    plt, file_title = getPlot(Convdf, ['vcdBR', 'udoBR', 'metricIsoHess','uniform'] , 'Elements', 'H1', 'H1 Convergence', plottype='loglog', methodcolors=methodcolors, methodmarkers = methodmarkers, methodlabels = methodlabels ,legend = True)
    plt.savefig(os.path.join(plot_dir, 'Test.png'), bbox_inches="tight")
    
    
    
    
    
    #plt, file_title = getPlot(Convdf, ['udo', 'udoBR', 'uniform'] , 'Elements', 'H1', 'UDO', plottype='loglog', methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    #plt, file_title = getPlot(Convdf, ['vcd', 'vcdBR', 'uniform'] , 'Elements', 'H1', 'VCD', plottype='loglog', methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    #plt, file_title = getPlot(Convdf, ['metricIso', 'metricIsoHess', 'uniform'] , 'Elements', 'H1', 'Metric', plottype='loglog', methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    #plt, file_title = getPlot(Convdf, ['udo', 'udoBR', 'uniform'] , 'Elements', 'Jaccard', 'UDO', plottype='loglog', methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    #plt, file_title = getPlot(Convdf, ['vcd', 'vcdBR', 'uniform'] , 'Elements', 'Jaccard', 'VCD', plottype='loglog', methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    #plt, file_title = getPlot(Convdf, ['metricIso', 'metricIsoHess', 'uniform'] , 'Elements', 'Jaccard', 'Metric', plottype='loglog', methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    #plt, file_title = getPlot(Convdf.iloc[1:], ['udo', 'vcd', 'metricIso'] , 'Elements', 'Jaccard', 'Comparison of Best Methods WRT Jaccard', plottype='loglog', methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    #plt, file_title = getPlot(Convdf.iloc[0:], ['udo', 'vcd', 'metricIso'] , 'Elements', 'Hausdorff', 'Comparison of Best Methods WRT Hausdorff', plottype='loglog', methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    #plt, file_title = getPlot(Convdf, ['vcdBR', 'udoBR', 'metricIsoHess'] , 'PreMeshCompTime/Elements', 'L2', 'L2 Time Effeciency of Inactive Set Methods', plottype='loglog',methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    #plt, file_title = getPlot(Convdf.iloc[0:], ['vcd','udo' ,'metricIso'] , 'PreMeshCompTime/Elements', 'Hausdorff', 'Hausdorff Time Effeciency of Inactive Set Methods', plottype='loglog',methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    #plt, file_title = getPlot(Convdf.iloc[0:], ['vcd','udo' ,'metricIso'] , 'PreMeshCompTime/Elements', 'Jaccard', 'Jaccard Time Effeciency of Inactive Set Methods', plottype='loglog',methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    #plt, file_title = getPlot(Convdf.iloc[0:], ['vcd' ,'metricIso'] , 'Elements', 'SolveTime/Elements', 'L Shaped Domain Solve Time Comparison', plottype='loglog',methodcolors=methodcolors, legend = True)
    #plt.savefig(os.path.join(plot_dir, file_title))
    # Grid sequencing should make the solve time better for tag and refine methods?? Likely an issue with cross mesh interpolation?