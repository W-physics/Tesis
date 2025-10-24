#!/bin/sh

#SBATCH -p gpu  	# Cola para correr el job.  Especificar gpu partition/queue (required)
#SBATCH --gres=gpu:1	# GPUs solicitadas (required), Default=1
#SBATCH -N 1		# Nodos requeridos, Default=1
#SBATCH -n 8		# Cores por nodo requeridos, Default=1
#SBATCH --mem=16G  	# Memoria Virtual/RAM, Default=2048
#SBATCH -t 10:00:00  	# Walltime 
#SBATCH --mail-user=w.montano@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH --job-name=test-1GPU_8threads
#SBATCH -o job_1GPU8threads.log  # Output filename
host=`/bin/hostname`
date=`/bin/date`
echo "Corri el: "$date
echo "Corri en la maquina: "$host

eval "$(micromamba shell hook --shell bash)"

micromamba activate William

echo "Ejecutando main.py"

python3 main.py
