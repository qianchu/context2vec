# set up bash shell
export SHELL=/bin/bash
cd /home/context-embed
# run in the docker image: continuumio/anaconda

# configure environment
env=context2vec
dir="/opt/conda/envs/$env"
if [ -d "$dir" ]
then
	echo "$dir found."
	source activate $env
        conda info --envs
else
    echo "$dir not found."
    conda create --name $env
    source activate $env
    apt-get install vim
    pip install chainer==1.7

fi



# pack python project
cd /home/context-embed/context2vec
python setup.py install

# run jupyter
cd /home/context-embed
jupyter notebook --ip '*' --port=8882 --allow-root &
