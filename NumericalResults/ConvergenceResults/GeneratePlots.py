import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess  # for running convergence.py later


def getPlot(df, amrtype, x, y, plottype=None):
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


if __name__ == "__main__":

    # Read in data from each scheme
    AMRdf = pd.read_csv('AMR.csv')
    Hybriddf = pd.read_csv('Hybrid.csv')
    Unifdf = pd.read_csv('Unif.csv')

    AMRdf = AMRdf.assign(Scheme='AMR')
    Hybriddf = Hybriddf.assign(Scheme='Hybrid')
    Unifdf = Unifdf.assign(Scheme='Unif')

    # Concatenate and set MultiIndex
    Convdf = pd.concat([AMRdf, Hybriddf, Unifdf])
    Convdf.set_index(['Scheme', Convdf.index], inplace=True)

    # Sort the MultiIndex
    Convdf.sort_index(inplace=True)

    # loglog Number of elements vs l2 with convergence rate
    # Number of elements vs Hausdorff distance
    # loglog Number of elements vs Hausdorff distance
    # Number of elements vs IoU
    # loglog Number of elements vs 1 - IoU
