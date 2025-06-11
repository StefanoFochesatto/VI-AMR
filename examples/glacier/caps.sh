# run as
#   bash caps.sh &> caps.txt

MPG="mpiexec --bind-to hwthread --map-by core"
P=12

OPTS="-prob cap -elevdepend -m 20 -udo_n 2 -pcount 20"  # FIXME -udo_n 1 or -udo_n 2 better?
REFINE="-uniform 3 -refine 9"  # FIXME ?

ELA=1000
BOX="-box 1000.0e3 1300.0e3 900.0e3 1200.0e3"
CMD="python3 steady.py $OPTS $REFINE $BOX -sELA $ELA -extractpvd result_sub_$ELA.pvd -opvd result_cap_$ELA.pvd"
echo $CMD
time $MPG -n $P $CMD

ELA=800
BOX="-box 450.0e3 750.0e3 350.0e3 650.0e3"
CMD="python3 steady.py $OPTS $REFINE $BOX -sELA $ELA -extractpvd result_sub_$ELA.pvd -opvd result_cap_$ELA.pvd"
echo $CMD
time $MPG -n $P $CMD

ELA=600
BOX="-box 1100.0e3 1400.0e3 1400.0e3 1700.0e3"
CMD="python3 steady.py $OPTS $REFINE $BOX -sELA $ELA -extractpvd result_sub_$ELA.pvd -opvd result_cap_$ELA.pvd"
echo $CMD
time $MPG -n $P $CMD
