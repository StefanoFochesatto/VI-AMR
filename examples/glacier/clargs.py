des="""
Solves a 2D steady, isothermal shallow ice approximation glacier obstacle problem.
See README.md for examples and METHOD.md for documentation of the mathematics.

The domain is a square [0,L]^2 with L = 1800.0 km, except that with -data
the domain is read from the file.

By default (-prob dome) we solve a flat bed case, with surface mass balance
which only depends on horizontal location, where the exact solution is
known.  Option -prob cap modifies this with a random, but smooth, bed topography,
but keeps the dome SMB.  Option -prob range generates a different SMB, still
depending only on horizontal location, and it results in a disconnected glacier.
An elevation-dependent surface mass balance model is also available, with
options -elevdepend (to turn it on) and -sELA to set equilibrium line altitude.
This case does not allow -newton.

We apply the UDO or VCD methods for free-boundary refinement.  The default mode
does n=1 UDO at the free boundary plus gradient-recovery refinement in the
inactive set.

The default VI solver is Picard iteration on the tilt (Jouvet & Bueler, 2012),
and vinewtonrsls (+ mumps) for each tilt.  A full Newton iteration, i.e. simply
vinewtonrsls, is turned on with -newton, but it does not converge in many
harder cases.
"""

from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(description=des, formatter_class=RawTextHelpFormatter)

parser.add_argument(
    "-box",
    metavar="X",
    type=float,
    nargs=4,
    default=[0.0, 1800.0e3, 0.0, 1800.0e3],
    help="bounding box for -extractpvd; ignored if not -extractpvd",
)
parser.add_argument(
    "-csv",
    metavar="FILE",
    type=str,
    default="",
    help="output file name for dome error report (.csv)",
)
parser.add_argument(
    "-data",
    metavar="FILE",
    type=str,
    default="",
    help='read "topg" variable from NetCDF file (.nc)',
)
parser.add_argument(
    "-elevdepend",
    action="store_true",
    default=False,
    help="compute surface mass balance from an elevation-dependent model",
)
parser.add_argument(
    "-extractpvd",
    metavar="FILE",
    type=str,
    default="",
    help="extract a submesh, defined by -box, into Paraview-format file (.pvd)",
)
parser.add_argument(
    "-hmin",
    type=float,
    default=-1,
    help="do not refine below this diameter (default: -1 .. so no hmin)",
)
parser.add_argument(
    "-jaccard",
    action="store_true",
    default=False,
    help="compare successive active sets by Jaccard agreement",
)
parser.add_argument(
    "-m",
    type=int,
    default=10,
    metavar="M",
    help="number of cells in each direction on initial mesh [default=10]",
)
parser.add_argument(
    "-newton",
    action="store_true",
    default=False,
    help="use straight Newton instead of Picard+Newton",
)
parser.add_argument(
    "-opvd",
    metavar="FILE",
    type=str,
    default="",
    help="name for Paraview-format output file (.pvd)",
)
parser.add_argument(
    "-pcount",
    type=int,
    default=10,
    metavar="P",
    help="number of Picard frozen-tilt (and a(s) if -elevdepend) iterations [default=10]",
)
parser.add_argument(
    "-prob",
    type=str,
    default="dome",
    metavar="X",
    choices=["dome", "cap", "range"],
    help="choose problem from {dome, cap, range}",
)
parser.add_argument(
    "-qdegree",
    type=int,
    default=5,
    metavar="Q",
    help="quadrature degree in non-linear part of weak form [default=5]",
)
parser.add_argument(
    "-refine",
    type=int,
    default=3,
    metavar="R",
    help="number of AMR refinements [default 3]",
)
parser.add_argument(
    "-sELA",
    type=float,
    default=1000.0,
    metavar="X",
    help="equilibrium line altitude to use if -elevdepend [default=1000.0]",
)
parser.add_argument(
    "-softening",
    type=float,
    default=1.0,
    metavar="X",
    help="multiply Gamma by softening factor X; X>1 softens, 0<X<1 hardens [default=1.0]",
)
parser.add_argument(
    "-theta",
    type=float,
    default=0.5,
    metavar="X",
    help="theta to use in fixed-rate marking strategy in inactive set [default=0.5]",
)
parser.add_argument(
    "-udo_n",
    type=int,
    default=1,
    metavar="N",
    help="use udomark(.., n=N) [default 1]",
)
parser.add_argument(
    "-uniform",
    type=int,
    default=0,
    metavar="R",
    help="initial R refinements are uniform [default 0]",
)
parser.add_argument(
    "-vcd",
    action="store_true",
    default=False,
    help="apply VCD free-boundary marking (instead of UDO)",
)
