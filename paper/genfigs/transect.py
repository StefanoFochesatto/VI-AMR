# figure for a transect through the "-prob cap" bed topography
# copy of formulas from examples/glacier/synthetic.py, converted to numpy

import numpy as np
import matplotlib.pyplot as plt

# constants (same for all problems)
L = 1800.0e3        # domain is [0,L]^2, with fields centered at (xc,xc)
xc = L/2
secpera = 31556926.0
n = 3.0
g = 9.81
rho = 910.0
A = 1.0e-16 / secpera
Gamma = 2*A*(rho * g)**n / (n+2)

domeL = 750.0e3
domeH0 = 3600.0

def dome_exact(x, y, n=3.0):
    r = np.sqrt((x - xc) ** 2 + (y - xc) ** 2)
    mm = 1 + 1/n
    qq = n / (2*n + 2)
    CC = domeH0 / (1-1/n)**qq
    z = r[r < domeL] / domeL
    tmp = mm * z - 1/n + (1-z)**mm - z**mm
    expr = CC * tmp**qq
    s = np.zeros(np.shape(r))
    s[r < domeL] = expr
    return s

def bumps(x, y):
    B0 = 200.0  # (m); amplitude of bumps
    xx, yy = x / L, y / L
    b = + 5.0 * np.sin(np.pi*xx) * np.sin(np.pi*yy) \
        + np.sin(np.pi*xx) * np.sin(3*np.pi*yy) - np.sin(2*np.pi*xx) * np.sin(np.pi*yy) \
        + np.sin(3*np.pi*xx) * np.sin(3*np.pi*yy) + np.sin(3*np.pi*xx) * np.sin(5*np.pi*yy) \
        + np.sin(4*np.pi*xx) * np.sin(4*np.pi*yy) - 0.5 * np.sin(4*np.pi*xx) * np.sin(5*np.pi*yy) \
        - np.sin(5*np.pi*xx) * np.sin(2*np.pi*yy) - 0.5 * np.sin(10*np.pi*xx) * np.sin(10*np.pi*yy) \
        + 0.5 * np.sin(19*np.pi*xx) * np.sin(11*np.pi*yy) + 0.5 * np.sin(12*np.pi*xx) * np.sin(17*np.pi*yy)
    return B0 * b

xt = 1.1e6  # transect location
res = 501
y = np.linspace(0.0, L, res)
x = xt * np.ones(np.shape(y))
b = bumps(x, y)

fig, ax = plt.subplots(figsize=[20.0, 3.0])
ykm = y / 1.0e3
plt.plot(ykm, b, 'k')
ydash = np.array([0.0, 1.6e3])
sELA = [600.0, 800.0, 1000.0]
for j in range(3):
    zdash = sELA[j] * np.ones(np.shape(ydash))
    plt.plot(ydash, zdash, 'k--')
    plt.text(1.65e3, sELA[j] - 20.0, f"{sELA[j]:.0f} m", fontsize=18.0)
plt.axis("off")
plt.savefig("transect.png", bbox_inches="tight")
#plt.show()
