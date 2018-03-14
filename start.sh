
# run in the docker image: continuumio/anaconda

# configure environment
env=context2vec
dir="/opt/conda/envs/$env"
if [ -d "$dir" ]
then
	echo "$dir found."
    source activate $env
else
    echo "$dir not found."
    conda create --name $env
    source activate $env
    apt-get install vim
    pip install chainer==1.7
	#conda env create --name context2vec --file /home/context-embed/context2vec/environment.yml
fi



# pack python project and run jupyter
cd /home/context-embed/context2vec
python setup.py install
jupyter notebook --ip '*' --port=8888 --allow-root
