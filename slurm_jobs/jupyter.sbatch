#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=07:00:00
#SBATCH --mail-user=gv2167@nyu.edu

port=$(shuf -i 10000-65500 -n 1)
logs="/scratch/gv2167/meter_augmentation/output"

/usr/bin/ssh -N -f -R $port:localhost:$port log-1
/usr/bin/ssh -N -f -R $port:localhost:$port log-2
/usr/bin/ssh -N -f -R $port:localhost:$port log-3

cat<<EOF
Current Server: $(hostname)  Job starts at: $(date)

Step 1:
Please open a new terminal window and run:
$ ssh -L $port:localhost:$port $USER@greene.hpc.nyu.edu

(Or if you have an alias already setup):
$ ssh -L $port:localhost:$port greene

Keep this terminal window open for as long as you want to access this port.

Step 2:
Open http://localhost:${port} in your browser to access!
EOF

./sing << EOF
jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)

EOF
