# enhanced Genetic Expert Guided Learning (eGEGL)

Source code for the **enhanced GEGL framework** based on [Guiding Deep Molecular Optimization with Genetic Exploration](https://arxiv.org/pdf/2007.04897.pdf). GEGL is a powerful generative framework combining Reinforcement Learning, Genetic algorithms and Deep Learning. This repo is heavily inspired by the [original GEGL source code](https://github.com/sungsoo-ahn/genetic-expert-guided-learning). This package contains the following (non-exhaustive) enhancements to the original work:

- [x] Possibility to use Transformers instead of a LSTM-models for the neural apprentice.
- [x] Possibility to use a genetic expert based on the SELFIES chemical-path crossover as proposed in [STONED](https://github.com/aspuru-guzik-group/stoned-selfies).
- [x] Addition of an explainer models (GCN, D-MPNN) which generates attributions using the CAM method for generated molecules.
- [x] Fragment-based genetic expert which generates a fragment library based on the explainer model and recombines them in the SELFIES space to propose new molecules.
- [x] Support for multiple Genetic Experts sharing some alloted sampling budget. The query-size for each expert can be either fixed or dynamicaly recomputed at each optimization step.

## Installing the library

The library can be installed via the following commands:

```
make create_env
conda activate egegl
make install_lib
```

## Downloading the benchmark models and datasets

Datasets (ZINC/Guacamol) and pretrained models (LSTM on ZINC/Guacamol) can be easily obtained via:

```
make dl_zinc
make dl_guacamol
make dl_zinc_pretrain
make dl_guacamol_pretrain
```

## Pretraining own models

Alternatively, models can be pretrained on the downloaded datasets using the pretraining script:

```
CUDA_VISIBLE_DEVICES=<gpu_id> python scripts/run_pretrain.py --dataset zinc --use_cuda --model_type LSTM
```

Please see further available options in the script `scripts/run_pretrain.py`.

## Run optimizations with eGEGL

Optimizations can be run either via the `scripts/run_optimization.py` script or some customized code inspired by the former. 

If you would like to use the premade script, the following arguments can currently be passed:

- `save_root`(str): Where to save the final neural apprentice and explainer model if used. Defaults to `./results/`
- `benchmark_id`(int): The ID of the benchmark task. See `egegl/scoring/benchmarks.py` for the list of available benchmarks. Default to `28` which correponds to the _plogp_ task. 
- `dataset`(str): Dataset to consider for the task. Needs to be either `zinc` or `guacamol` and defaults to `zinc`.
- `dataset_path`(str): Path to the dataset. Defaults to `./data/datasets/zinc/all.txt`.
- `model_type`(str): Model type for the neural apprentice. Must be either `LSTM` or `Transformer`. Defaults to `LSTM`.
- `explainer_type`(str): Model type for the explainer model. Must be either `GCN` or `DMPNN`. Defaults to `GCN`.
- `max_smiles_length`(int): Maximum length for the generated SMILES. Defaults to `100`.
- `apprentice_load_dir`(str): Path to the pretrained neural apprentice. Defaults to `./data/pretrained_models/original_benchmarks/zinc`.
- `explainer_load_dir`(str): Path to the pretrained explainer model. Defaults to `None`.
- `genetic_experts`(List(str)): List of genetic experts to be used. Defaults to `["SELFIES", "SMILES", "ATTR"]`.
- `logger_type`(str): Logger type for the run. Can be either `Neptune` or `CommandLine`. Please ensure that you have a neptune account and have set the credentials correctly in the script first if you would like to use this logger.
- `project_qualified_name`(str): *For the Neptune logger only*, sets the name for neptune logging.
- `learning_rate`(float): Learning rate for the apprentice and explainer model. Defaults to `1e-3`.
- `mutation_initial_rate`(float): Initial mutation rate for the genetic experts. Defaults to `1e-2`.
- `num_steps`(int): Number of optimization steps. Defaults to `200`.
- `num_keep`(int): Number of molecules to keep in each priority queue. Defaults to `1024`.
- `max_sampling_batch_size`(int): Maximum sampling batch size during sampling of the neural apprentice. Defaults to `1024`
- `apprentice_sampling_batch_size`(int): Number of molecules to sample by the apprentice at each round. Defaults to `8192`.
- `expert_sampling_batch_size`(int): Number of molecules to sample by the experts at each round. Defaults to `8192`.
- `apprentice_training_batch_size`(int): Batch size of the data during imitation training of the apprentice. Defaults to `256`.
- `apprentice_training_steps`(int): The number of training steps of the apprentice during imitation training. Defaults to `8`.
- `num_jobs`(int): Number of parallel jobs during expert sampling.
- `record_filtered`(bool): Activates post-hoc filtering of the molecules as described in the original paper. Defaults to `False`.
- `use_cuda`(bool): Activates optimization on the specified CUDA device.
- `use_frozen_explainer`(bool): Whether to freeze the explainer during optimization. Defaults to `False`.
- `seed`(int): Sets the random seed for certain libraries. Defaults to `404`.


An example command to launch optimization can be:

```
CUDA_VISIBLE_DEVICES=0 python scripts/run_optimization.py \
--model_type LSTM --benchmark_id 28 --use_cuda \
--dataset zinc \
--apprentice_load_dir ./data/pretrained_models/original_benchmarks/zinc \
--max_smiles_length 81 --num_jobs 8 --genetic_expert SMILES
```


## Dependencies

Code was tested on

- Python >= 3.6.13
- torch == 1.8.1
- cuda == 10.1
- rdkit == 2020.09


## Copyright notice

```
Copyright (c) 2021 Elix, Inc.
```

The following source code cannot be used for commercial use but can be used freely otherwise. Please refer to the added `LICENSE.txt` file for more details.
