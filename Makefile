TORCH="1.8.0"
CUDA="cu102"

create_env:
	conda create -n egegl python=3.7 -y

install_lib:
	conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch -y
	conda install -c rdkit rdkit=2020.09 -y
	pip install neptune-client
	pip install tqdm
	pip install guacamol
	pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
	pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
	pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
	pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
	pip install torch-geometric
	pip install -e .

dl_zinc:
	curl https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/all.txt --create-dirs -o ./data/datasets/zinc/all.txt
	curl https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/train.txt --create-dirs -o ./data/datasets/zinc/train.txt
	curl https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/valid.txt --create-dirs -o ./data/datasets/zinc/valid.txt
	curl https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/test.txt --create-dirs -o ./data/datasets/zinc/test.txt

dl_guacamol:
	curl https://ndownloader.figshare.com/files/13612745 -L --create-dirs -o ./data/datasets/guacamol/all.txt
	curl https://ndownloader.figshare.com/files/13612760 -L --create-dirs -o ./data/datasets/guacamol/train.txt
	curl https://ndownloader.figshare.com/files/13612766 -L --create-dirs -o ./data/datasets/guacamol/valid.txt
	curl https://ndownloader.figshare.com/files/13612757 -L --create-dirs -o ./data/datasets/guacamol/test.txt

dl_zinc_pretrain:
	curl https://raw.githubusercontent.com/sungsoo-ahn/genetic-expert-guided-learning/main/resource/checkpoint/zinc/generator_config.json --create-dirs -o ./data/pretrained_models/original_benchmarks/zinc/generator_config.json
	curl https://github.com/sungsoo-ahn/genetic-expert-guided-learning/blob/main/resource/checkpoint/zinc/generator_weight.pt?raw=true -L --create-dirs -o ./data/pretrained_models/original_benchmarks/zinc/generator_weight.pt

dl_guacamol_pretrain:
	curl https://raw.githubusercontent.com/sungsoo-ahn/genetic-expert-guided-learning/main/resource/checkpoint/guacamol/generator_config.json --create-dirs -o ./data/pretrained_models/original_benchmarks/guacamol/generator_config.json
	curl https://github.com/sungsoo-ahn/genetic-expert-guided-learning/blob/main/resource/checkpoint/guacamol/generator_weight.pt?raw=true -L --create-dirs -o ./data/pretrained_models/original_benchmarks/guacamol/generator_weight.pt

typehint:
	mypy egegl/ scripts/

lint:
	black egegl/ scripts/

sort:
	isort -m 3 --profile black egegl/ scripts/

checklist: sort lint typehint

.PHONY: create_env install_lib dl_zinc dl_guacamol dl_zinc_pretrain, dl_guacamol_pretrain typehint lint sort checklist
