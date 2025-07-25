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

for meth in ["UDO+BR", "VCD+BR"]:
    plt.figure()
    plt.loglog(intvals(meth, "NE"), floatvals(meth, "ENORM"), 'ko', ms=ms0, label=r"$||u-u_h||_2$")
    plt.loglog(intvals(meth, "NE"), floatvals(meth, "ENORMPREF"), 'ko', ms=ms0+2, markerfacecolor="w", alpha=0.5, markeredgecolor="k", label=r"$||u-\tilde u_h||_2$")
    plt.loglog(intvals("UNI", "NE"), floatvals("UNI", "ENORM"), 'r+', ms=ms0+4, label=r"$||u-u_h||_2$ (uniform)")
    plt.grid(True)
    #plt.ylabel(r"$L^2$ error", fontsize=fs0+2.0)
    plt.xlabel("elements", fontsize=fs0+2.0)
    plt.legend(fontsize=fs0)
    plt.xlim(2.0e2,1.0e7)
    #plt.show()
    outname = "convball_" + meth + ".png"
    print(f"writing {outname} ...")
    plt.savefig(outname, bbox_inches="tight")

plt.figure()
meth = "UDO+BR"
plt.loglog(intvals(meth, "NE"), 1.0 - floatvals(meth, "JACCARD"), 'ko', ms=ms0+4, markerfacecolor="w", markeredgecolor="k", label=meth)
meth = "VCD+BR"
plt.loglog(intvals(meth, "NE"), 1.0 - floatvals(meth, "JACCARD"), 'ko', ms=ms0, label=meth)
plt.loglog(intvals("UNI", "NE"), 1.0 - floatvals("UNI", "JACCARD"), 'r+', ms=ms0+4, label="uniform")
plt.grid(True)
#plt.ylabel("active set Jaccard distance", fontsize=fs0)
plt.xlabel("elements", fontsize=fs0+2.0)
plt.legend(fontsize=fs0)
plt.xlim(2.0e2,1.0e7)
#plt.show()
outname = "jaccball.png"
print(f"writing {outname} ...")
plt.savefig(outname, bbox_inches="tight")
