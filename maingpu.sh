#!/bin/sh

SBATCH --job-name=Diffusion		#Nombre del job
SBATCH -p short			#Cola a usar, Default=short (Ver colas y límites en /hpcfs/shared/README/partitions.txt)
SBATCH -N 1				#Nodos requeridos, Default=1
SBATCH -n 1				#Tasks paralelos, recomendado para MPI, Default=1
SBATCH --cpus-per-task=8		#Cores requeridos por task, recomendado para multi-thread, Default=1
SBATCH --mem=2000		#Memoria en Mb por CPU, Default=2048
SBATCH --time=10:00:00			#Tiempo máximo de corrida, Default=2 horas
SBATCH --mail-user=w.montano@uniandes.edu.co
#SBATCH --mail-type=ALL			
SBATCH -o TEST_job.o%j			#Nombre de archivo de salida

eval "$(micromamba shell hook --shell bash)"

micromamba activate William

echo "Ejecutando main.py"

python3 main.py
