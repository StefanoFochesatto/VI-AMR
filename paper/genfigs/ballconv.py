import numpy as np
import matplotlib.pyplot as plt
import csv

path = "../static/"


def fname(method):
    return path + "sphere_" + method + ".csv"


def floatvals(method, field):
    with open(fname(method), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return np.array([float(row[field]) for row in reader], dtype=np.float64)


def intvals(method, field):
    with open(fname(method), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return np.array([int(row[field]) for row in reader], dtype=np.int32)


ms0 = 10.0
fs0 = 14.0

for meth in ["UDO+BR", "VCD+BR", "AVM"]:
    plt.figure()
    plt.loglog(intvals(meth, "NE"), floatvals(meth, "ENORM"),
               'ko', ms=ms0, label=r"$||u-u_h||_2$")
    plt.loglog(intvals(meth, "NE"), floatvals(meth, "ENORMPREF"),
               'ks', ms=ms0-1, label=r"$||u-\tilde u_h||_2$")
    plt.loglog(intvals("UNI", "NE"), floatvals("UNI", "ENORM"),
               'r+', ms=ms0+4, label=r"$||u-u_h||_2$ (uniform)")
    plt.grid(True)
    plt.title(f"$L^2$ Convergence: {meth}", fontsize=fs0+2.0)
    plt.ylabel(r"$L^2$ error", fontsize=fs0+2.0)
    plt.xlabel("elements", fontsize=fs0+2.0)
    plt.legend(fontsize=fs0, loc='lower left')
    plt.xlim(1.0e2, 1.0e7)
    outname = "convball_" + meth + ".png"
    print(f"writing {outname} ...")
    plt.savefig(outname, bbox_inches="tight")

plt.figure()
meth = "VCD+BR"
plt.loglog(intvals(meth, "NE"), 1.0 - floatvals(meth, "JACCARD"), 'ko', ms=ms0+4,
           markerfacecolor="w", markeredgecolor="k", label=meth)  # actually 2nd
meth = "UDO+BR"
plt.loglog(intvals(meth, "NE"), 1.0 - floatvals(meth, "JACCARD"),
           'ko', ms=ms0, label=meth)  # actually 1st
meth = "AVM"
plt.loglog(intvals(meth, "NE"), 1.0 - floatvals(meth,
           "JACCARD"), 'kd', ms=ms0+2, label=meth)
plt.loglog(intvals("UNI", "NE"), 1.0 - floatvals("UNI",
           "JACCARD"), 'r+', ms=ms0+2, label="uniform")
plt.grid(True)
plt.title("Active Set Localization", fontsize=fs0+2.0)
plt.ylabel("Jaccard distance", fontsize=fs0+2.0)
plt.xlabel("elements", fontsize=fs0+2.0)
# reorder labels
handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 0, 2, 3]
plt.legend([handles[idx] for idx in order], [labels[idx]
           for idx in order], fontsize=fs0)
plt.xlim(1.0e2, 1.0e7)
outname = "jaccball.png"
print(f"writing {outname} ...")
plt.savefig(outname, bbox_inches="tight")
