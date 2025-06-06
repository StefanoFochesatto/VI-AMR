import numpy as np
import matplotlib.pyplot as plt
import csv

path = "../static/glacier/"

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

enorm = ["UERRH1", "HERRINF", "DRMAX"]
ytitle = [r"$H^1$ error in $u$",
          r"$L^\infty$ error in $H$",
          "margin location error (meters)"]
ms0 = 10.0
fs0 = 14.0
for j in range(3):
    plt.figure()
    plt.loglog(intvals("udo", "NE"), floatvals("udo", enorm[j]), 'ko', ms=ms0, label="UDO+GR")
    plt.loglog(intvals("vcd", "NE"), floatvals("vcd", enorm[j]), 'o', ms=ms0+2, markerfacecolor="w", alpha=0.5, markeredgecolor="k", label="VCD+GR")
    plt.loglog(intvals("uniform", "NE"), floatvals("uniform", enorm[j]), 'k+', ms=ms0+2, label="uniform")
    plt.grid(True)
    plt.ylabel(ytitle[j], fontsize=fs0+2.0)
    plt.xlabel("elements", fontsize=fs0+2.0)
    plt.legend(fontsize=fs0)
    plt.xlim(7.0e1,1.0e7)
    if enorm[j] == "HERRINF":
        plt.ylim(1.0e-3, 1.0)
    elif enorm[j] == "DRMAX":
        plt.ylim(1.0e1, 3.0e5)
    #plt.show()
    plt.savefig(enorm[j].lower()+".png", bbox_inches="tight")
