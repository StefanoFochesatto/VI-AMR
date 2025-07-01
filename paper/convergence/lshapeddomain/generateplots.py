import numpy as np
import matplotlib.pyplot as plt
import csv

path = "../lshapeddomain/ResultsLShaped/"

def floatvals(method, field):
    fname = path + method + ".csv"
    with open(fname, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return np.array([float(row[field]) for row in reader], dtype=np.float64)

def intvals(method, field):
    fname = path + method + ".csv"
    with open(fname, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return np.array([int(row[field]) for row in reader], dtype=np.int32)

enorm = ["L2", "H1", "Jaccard"]
ytitle = [r"$L^2$ error in $u$",
          r"$H^1$ error in $u$",
          "Jaccard"]
ms0 = 10.0
fs0 = 14.0
for j in range(3):
    plt.figure()
    plt.loglog(intvals("udoBR", "Elements"), floatvals("udoBR", enorm[j]), 'ko', ms=ms0, label="UDO+BR")
    plt.loglog(intvals("vcdBR", "Elements"), floatvals("vcdBR", enorm[j]), 'o', ms=ms0+2, markerfacecolor="w", alpha=0.5, markeredgecolor="k", label="VCD+BR")
    plt.loglog(intvals("uniform", "Elements"), floatvals("uniform", enorm[j]), 'k+', ms=ms0+2, label="uniform")
    plt.grid(True)
    plt.ylabel(ytitle[j], fontsize=fs0+2.0)
    plt.xlabel("elements", fontsize=fs0+2.0)
    plt.legend(fontsize=fs0)
    plt.xlim(7.0e1,1.0e6)
    if enorm[j] == "Jaccard":
        plt.ylim(8.0e-1, 1.0e0)

    #plt.show()
    plt.savefig(enorm[j].lower()+".png", bbox_inches="tight")
