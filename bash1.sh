#!/bin/bash
#SBATCH --job-name=fine_tuning
#SBATCH --account=def-escheme
#SBATCH --mem-per-cpu=8192M  #16384M                                         # increase as needed
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=4:30:20                                              # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --output=%x-%J.out
#SBATCH --mail-user=saeed.kazemi@unb.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:p100:1


module load python/3.8
module load git-lfs

cd /home/saeed67/projects/def-escheme/saeed67/master-project
# virtualenv ./env
source ./env/bin/activate
# pip install --upgrade pip


# pip install --no-index pandas scikit_learn matplotlib seaborn
# pip install --no-index tensorflow jupyterlab

# pip install --no-index PyWavelets
# pip install --no-index opencv-python
# pip install --no-index openpyxl


# mv *.out ./logs/



python ./Codes/computing_parallel1.py


## $ chmod 755 bash.sh
## $ seff {Job_ID}                                                                                       # list resources used by a completed job 
## $ sacct -j jobID [--format=jobid,maxrss,elapsed]                                                      # list resources used by a completed job
## $ scancel <jobid>                                                                                     # Cancelling jobs
## $ sbatch bash.sh                                                                                      # submit jobs
## $ squeue -u saeed67



## $ scp filename saeed67@cedar.computecanada.ca:/path/to                                               # File transfer
## $ scp saeed67@cedar.computecanada.ca:/path/to/filename localPath                                     # File transfer


## $ nano ./env/bin/notebook.sh

        #### #!/bin/bash
        #### unset XDG_RUNTIME_DIR
        #### jupyter-lab --ip $(hostname -f) --no-browser


## $ chmod u+x $VIRTUAL_ENV/bin/notebook.sh


## salloc --account=def-escheme --cpus-per-task=1 --mem=100M --time=1:10:00 srun ./env/bin/notebook.sh   # intractive mode


## on new terminal: ssh -L 8888:<<gra105.graham.sharcnet:8888>> saeed67@cedar.computecanada.ca
## on browser: http://localhost:8888/?token=<token>


