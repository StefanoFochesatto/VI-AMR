# run as
#   bash caps.sh &> caps.txt

MPG="mpiexec --bind-to hwthread --map-by core"
P=12

HMIN="-hmin 500.0"
REFINE="-uniform 3 -refine 12"
OPTS="-prob cap -elevdepend -m 20 -pcount 20 $HMIN $REFINE"

ELA=1000
BOX="-box 1100.0e3 1300.0e3 900.0e3 1100.0e3"
CMD="python3 steady.py $OPTS $BOX -sELA $ELA -extractpvd result_sub_$ELA.pvd -opvd result_cap_$ELA.pvd"
echo $CMD
time $MPG -n $P $CMD

ELA=800
BOX="-box 500.0e3 700.0e3 350.0e3 550.0e3"
CMD="python3 steady.py $OPTS $BOX -sELA $ELA -extractpvd result_sub_$ELA.pvd -opvd result_cap_$ELA.pvd"
echo $CMD
time $MPG -n $P $CMD

ELA=600
BOX="-box 1100.0e3 1300.0e3 1350.0e3 1550.0e3"
CMD="python3 steady.py $OPTS $BOX -sELA $ELA -extractpvd result_sub_$ELA.pvd -opvd result_cap_$ELA.pvd"
echo $CMD
time $MPG -n $P $CMD
