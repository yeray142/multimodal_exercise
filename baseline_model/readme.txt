#
# The baseline was tested using docker, as illustrated below.
#

1) run the 'pytorch' docker image

docker run --gpus '"device=0"' --rm -ti -v "$(pwd)":/home/code -v /my_data_path/First_Impressions_v3_multimodal:/home/data pytorch/pytorch:latest 


2) Install the following libraries inside the docker

pip install torchinfo
pip install torchview
apt-get update
pip install graphviz
apt-get install graphviz -y
pip install matplotlib
pip install facenet-pytorch

#
# Please, let me know (julio.silveira@ub.edu) in case you find any bug or problem in the baseline implementation. 
#