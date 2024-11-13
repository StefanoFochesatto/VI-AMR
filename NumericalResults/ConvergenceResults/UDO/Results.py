import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import subprocess


RunExperiment = 1


def run_command(refinement_value):
    # Create the command dynamically
    command = ["python3", "RunSolution.py",
               "--refinement", refinement_value]

    try:
        # Execute the command
        subprocess.run(command, check=True)
        print(f"Successfully ran with refinement: {refinement_value}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Example usage
    refinement = ["Unif", "Hybrid", "AMR"]

    if RunExperiment:
        for i in refinement:
            run_command(i)

    AMRdf = pd.read_csv('AMR.csv')
    Hybriddf = pd.read_csv('Hybrid.csv')
    Unifdf = pd.read_csv('Unif.csv')

    # Extract the the columns \{L2,IoU,Hausdorff,Elements,dof\} into numpy arrays
    l2Unif = Unifdf['L2'].to_numpy()
    l2AMR = AMRdf['L2'].to_numpy()
    l2Hybrid = Hybriddf['L2'].to_numpy()

    iouUnif = Unifdf['IoU'].to_numpy()
    iouAMR = AMRdf['IoU'].to_numpy()
    iouHybrid = Hybriddf['IoU'].to_numpy()

    hausdorffUnif = Unifdf['Hausdorff'].to_numpy()
    hausdorffAMR = AMRdf['Hausdorff'].to_numpy()
    hausdorffHybrid = Hybriddf['Hausdorff'].to_numpy()

    elementsUnif = Unifdf['Elements'].to_numpy()
    elementsAMR = AMRdf['Elements'].to_numpy()
    elementsHybrid = Hybriddf['Elements'].to_numpy()

    dofUnif = Unifdf['dof'].to_numpy()
    dofAMR = AMRdf['dof'].to_numpy()
    dofHybrid = Hybriddf['dof'].to_numpy()

    # create a list which indicates refinement level
    Refinements = np.arange(1, len(l2Unif)+1)

    axis_label_fontsize = 16

    # loglog Number of elements vs l2 with convergence rate
    ConvUnif = np.polyfit(np.log(elementsUnif), np.log(l2Unif), 1)
    ConvAMR = np.polyfit(np.log(elementsAMR), np.log(l2AMR), 1)
    ConvHybrid = np.polyfit(np.log(elementsHybrid), np.log(l2Hybrid), 1)
    plt.figure(figsize=(10, 6))
    plt.loglog(elementsUnif, l2Unif,
               label=f'Uniform Convergence Rate: {ConvUnif[0]:.2f}', marker='o')
    plt.loglog(elementsAMR, l2AMR,
               label=f'Adaptive Convergence Rate: {ConvAMR[0]:.2f}', marker='s')
    plt.loglog(elementsHybrid, l2Hybrid,
               label=f'Hybrid Convergence Rate: {ConvHybrid[0]:.2f}', marker='x')
    plt.xlabel('Number of Elements', fontsize=axis_label_fontsize)
    plt.ylabel('L2 Error', fontsize=axis_label_fontsize)
    plt.title('Convergence Plot UDO - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Number of elements vs Hausdorff distance
    plt.figure(figsize=(10, 6))
    plt.semilogx(elementsUnif, hausdorffUnif,
                 label='Uniform Refinement', marker='o')
    plt.semilogx(elementsAMR, hausdorffAMR,
                 label='Adaptive Refinement', marker='s')
    plt.semilogx(elementsHybrid, hausdorffHybrid,
                 label='Hybrid Refinement', marker='x')
    plt.xlabel('Number of Elements', fontsize=axis_label_fontsize)
    plt.ylabel('Hausdorff Distance', fontsize=axis_label_fontsize)
    plt.title('Convergence Plot UDO - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)
    plt.show()

    # loglog Number of elements vs Hausdorff distance
    ConvUnif = np.polyfit(np.log(elementsUnif), np.log(hausdorffUnif), 1)
    ConvAMR = np.polyfit(np.log(elementsAMR), np.log(hausdorffAMR), 1)
    ConvHybrid = np.polyfit(np.log(elementsHybrid), np.log(hausdorffHybrid), 1)

    plt.figure(figsize=(10, 6))
    plt.loglog(elementsUnif, hausdorffUnif,
               label=f'Uniform Convergence Rate: {ConvUnif[0]:.2f}', marker='o')
    plt.loglog(elementsAMR, hausdorffAMR,
               label=f'Adaptive Convergence Rate: {ConvAMR[0]:.2f}', marker='s')
    plt.loglog(elementsHybrid, hausdorffHybrid,
               label=f'Hybrid Convergence Rate: {ConvHybrid[0]:.2f}', marker='x')
    plt.xlabel('Number of Elements', fontsize=axis_label_fontsize)
    plt.ylabel('Hausdorff Distance', fontsize=axis_label_fontsize)
    plt.title('Convergence Plot UDO - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Number of elements vs IoU
    plt.figure(figsize=(10, 6))
    plt.semilogx(elementsUnif, iouUnif, label='Uniform Refinement', marker='o')
    plt.semilogx(elementsAMR, iouAMR, label='Adaptive Refinement', marker='s')
    plt.semilogx(elementsHybrid, iouHybrid,
                 label='Hybrid Refinement', marker='x')
    plt.xlabel('Number of Elements', fontsize=axis_label_fontsize)
    plt.ylabel('Jaccard index', fontsize=axis_label_fontsize)
    plt.title('Convergence Plot UDO - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)
    plt.show()

    # loglog Number of elements vs 1 - IoU
    ConvUnif = np.polyfit(np.log(elementsUnif), np.log(1 - iouUnif), 1)
    ConvAMR = np.polyfit(np.log(elementsAMR), np.log(1 - iouAMR), 1)
    ConvHybrid = np.polyfit(np.log(elementsHybrid), np.log(1 - iouHybrid), 1)
    plt.figure(figsize=(10, 6))
    plt.loglog(elementsUnif, 1 - iouUnif,
               label=f'Uniform Convergence Rate: {ConvUnif[0]:.2f}', marker='o')
    plt.loglog(elementsAMR, 1 - iouAMR,
               label=f'Adaptive Convergence Rate: {ConvAMR[0]:.2f}', marker='s')
    plt.loglog(elementsHybrid, 1 - iouHybrid,
               label=f'Hybrid Convergence Rate: {ConvHybrid[0]:.2f}', marker='x')
    plt.xlabel('Number of Elements', fontsize=axis_label_fontsize)
    plt.ylabel('1 - Jaccard index', fontsize=axis_label_fontsize)
    plt.title('Convergence Plot UDO - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)
    plt.show()

    print('done')
