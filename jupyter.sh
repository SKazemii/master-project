#!/bin/bash

module load python/3.8
cd /home/saeed67/projects/def-escheme/saeed67/master-project


# virtualenv ./env
source  ./env/bin/activate
# pip install --no-index --upgrade pip

# pip install --no-index pandas scikit_learn matplotlib seaborn
# pip install --no-index tensorflow jupyter
# pip install optunity fastparquet ipykernel


# pip install --no-index PyWavelets
# pip install --no-index opencv-python
# pip install --no-index openpyxl

# nano ./env/bin/notebook.sh

#         #/!/bin/bash
#         unset XDG_RUNTIME_DIR
#         jupyter-lab --ip $(hostname -f) --no-browser or 
#         jupyter notebook --ip $(hostname -f) --no-browser

# chmod u+x $VIRTUAL_ENV/bin/notebook.sh
salloc --account=def-escheme --cpus-per-task=32 --ntasks=1 --mem=32G --time=2:10:00 srun ./env/bin/notebook.sh   # intractive mode

## on new terminal: ssh -L 8888:cdr837.int.cedar.computecanada.ca:8888 saeed67@cedar.computecanada.ca
## on browser: http://localhost:8888/?token=0ebee10d86b7149b0ffa4d0677b70145d2f8234d1ce286b8







# ssh saeed67@narval.computecanada.ca
# ssh saeed67@niagara.computecanada.ca

# cd /home/saeed67/projects/def-escheme/saeed67
# git clone https://github.com/SKazemii/master-project.git

## $ scp C:\Project\master-project\Datasets\Casia-D saeed67@narval.computecanada.ca:/home/saeed67/projects/def-escheme/saeed67/master-project/Codes
                                              # local to server File transfer
## $ scp saeed67@cedar.computecanada.ca:/home/saeed67/.ssh/id_rsa.pub ./                                     # File transfer

## $ scp C:\Project\master-project\id_rsa.pub saeed67@narval.computecanada.ca:/home/saeed67/.ssh



