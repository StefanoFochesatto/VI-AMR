from firedrake import *
try:
    import netgen
except ImportError:
    import sys
    warning("Unable to import NetGen.")
    sys.exit(0)

from firedrake.petsc import PETSc
from netgen.geom2d import SplineGeometry

import matplotlib.pyplot as plt
import numpy as np
import copy
import time

import pandas as pd
import subprocess
import os

RunExperiment = 0


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

   # Number of elements vs L2 error
    plt.figure(figsize=(10, 6))
    plt.loglog(elementsUnif, l2Unif, label='Uniform Refinement', marker='o')
    plt.loglog(elementsAMR, l2AMR, label='Adaptive Refinement', marker='s')
    plt.loglog(elementsHybrid, l2Hybrid, label='Hybrid Refinement', marker='x')
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

    # refinement vs log dof
    plt.figure(figsize=(10, 6))
    plt.semilogy(Refinements, dofUnif, label='Uniform Refinement', marker='o')
    plt.semilogy(Refinements, dofAMR, label='Adaptive Refinement', marker='s')
    plt.semilogy(Refinements, dofHybrid, label='Hybrid Refinement', marker='x')
    plt.xlabel('Refinement Level', fontsize=axis_label_fontsize)
    plt.ylabel('Log DOF', fontsize=axis_label_fontsize)
    plt.title('Convergence Plot UDO - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Refinement level vs log l2 error
    plt.figure(figsize=(10, 6))
    plt.semilogy(Refinements, l2Unif, label='Uniform Refinement', marker='o')
    plt.semilogy(Refinements, l2AMR, label='Adaptive Refinement', marker='s')
    plt.semilogy(Refinements, l2Hybrid, label='Hybrid Refinement', marker='x')
    plt.xlabel('Refinement Level', fontsize=axis_label_fontsize)
    plt.ylabel('L2 Error', fontsize=axis_label_fontsize)
    plt.title('Convergence Plot UDO - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Refinement level vs IoU
    plt.figure(figsize=(10, 6))
    plt.semilogy(Refinements, iouUnif, label='Uniform Refinement', marker='o')
    plt.semilogy(Refinements, iouAMR, label='Adaptive Refinement', marker='s')
    plt.semilogy(Refinements, iouHybrid, label='Hybrid Refinement', marker='x')
    plt.xlabel('Refinement Level', fontsize=axis_label_fontsize)
    plt.ylabel('Jaccard index', fontsize=axis_label_fontsize)
    plt.title('Convergence Plot UDO - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Refinement level vs Hausdorff distance
    plt.figure(figsize=(10, 6))
    plt.semilogy(Refinements, hausdorffUnif,
                 label='Uniform Refinement', marker='o')
    plt.semilogy(Refinements, hausdorffAMR,
                 label='Adaptive Refinement', marker='s')
    plt.semilogy(Refinements, hausdorffHybrid,
                 label='Hybrid Refinement', marker='x')
    plt.xlabel('Refinement Level', fontsize=axis_label_fontsize)
    plt.ylabel('Hausdorff Distance', fontsize=axis_label_fontsize)
    plt.title('Convergence Plot UDO - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Area Metric vs Dof
    plt.figure(figsize=(10, 6))
    plt.loglog(dofUnif, iouUnif, label='Uniform Refinement', marker='o')
    plt.loglog(dofAMR, iouAMR, label='Adaptive Refinement', marker='s')
    plt.loglog(dofHybrid, iouHybrid, label='Hybrid Refinement', marker='x')
    plt.xlabel('Degrees of Freedom', fontsize=axis_label_fontsize)
    plt.ylabel('IoU', fontsize=axis_label_fontsize)
    plt.title('Convergence Plot UDO - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Hausdorff vs Dof
    plt.figure(figsize=(10, 6))
    plt.loglog(dofUnif, hausdorffUnif, label='Uniform Refinement', marker='o')
    plt.loglog(dofAMR, hausdorffAMR, label='Adaptive Refinement', marker='s')
    plt.loglog(dofHybrid, hausdorffHybrid,
               label='Hybrid Refinement', marker='x')
    plt.xlabel('Degrees of Freedom', fontsize=axis_label_fontsize)
    plt.ylabel('Hausdorff Distance', fontsize=axis_label_fontsize)
    plt.title('Convergence Plot UDO - Uniform vs Adaptive vs Hybrid Refinement')
    plt.legend()
    plt.grid(True)
    plt.show()

    print('done')
