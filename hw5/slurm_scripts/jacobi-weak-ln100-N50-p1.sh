#SBATCH --node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=jacobi-weak-ln100-N50-p1
#SBATCH --mail-type=END
#SBATCH --mail-user=cy1505@nyu.edu
#SBATCH --output=jacobi-weak-ln100-N50-p1.out

module purge
module load openmpi/gnu/4.0.2
mpiexec jacobi-mpi 100 20000