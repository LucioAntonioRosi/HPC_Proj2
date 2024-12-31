# Run per controllare il numero di iterationi di GMRES + usare questi per i plot in sequenziale

python3 codeProj3.py --loc_nx 32 --loc_ny 32
python3 codeProj3.py --loc_nx 64 --loc_ny 64
python3 codeProj3.py --loc_nx 128 --loc_ny 128
python3 codeProj3.py --loc_nx 16 --loc_ny 16
python3 codeProj3.py --loc_nx 64 --loc_ny 16
python3 codeProj3.py --loc_nx 16 --loc_ny 64
python3 codeProj3.py --loc_nx 128 --loc_ny 32
python3 codeProj3.py --loc_nx 32 --loc_ny 128

# Aumentando J

python3 codeProj3.py --loc_nx 32 --loc_ny 32 --J 8
python3 codeProj3.py --loc_nx 32 --loc_ny 32 --J 16
python3 codeProj3.py --loc_nx 32 --loc_ny 32 --J 32
python3 codeProj3.py --loc_nx 32 --loc_ny 32 --J 64

# Aumentando J e loc_ny

python3 codeProj3.py --loc_nx 32 --loc_ny 64 --J 8
python3 codeProj3.py --loc_nx 32 --loc_ny 128 --J 16
python3 codeProj3.py --loc_nx 32 --loc_ny 256 --J 32

# Controlliamo efficienza di parallelizzazione

mpirun -np 2 python3 codeProj3.py --loc_nx 32 --loc_ny 32 --J 8 --both True
mpirun -np 3 python3 codeProj3.py --loc_nx 32 --loc_ny 32 --J 8 --both True
mpirun -np 4 python3 codeProj3.py --loc_nx 32 --loc_ny 32 --J 8 --both True
mpirun -np 5 python3 codeProj3.py --loc_nx 32 --loc_ny 32 --J 8 --both True
mpirun -np 6 python3 codeProj3.py --loc_nx 32 --loc_ny 32 --J 8 --both True
mpirun -np 7 python3 codeProj3.py --loc_nx 32 --loc_ny 32 --J 8 --both True
mpirun -np 8 python3 codeProj3.py --loc_nx 32 --loc_ny 32 --J 8 --both True

