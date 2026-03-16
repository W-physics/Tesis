#!/bin/bash

#SBATCH --output="tunnel_log/tunnel_gpux-%J.log"
#SBATCH --job-name="gpu"
#SBATCH -p gpu          # Cola para correr el job.  Especificar gpu partition/queue (required)
#SBATCH --gres=gpu:1    # GPUs solicitadas (required), Default=1
#SBATCH -N 1            # Nodos requeridos, Default=1
#SBATCH --mail-user=w.montano@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH -n 1            # Cores por nodo requeridos, Default=1
#SBATCH --mem=24G       # Memoria Virtual/RAM
#SBATCH --time 15-00:00:00

module load anaconda

source ~/venvs/diffusion_models/bin/activate

cd ~/Github/Tesis

python3 main.py