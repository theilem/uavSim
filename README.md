## Old Version

The previous repository to recreate the [ICAR](https://ieeexplore.ieee.org/abstract/document/9659413) results is found
in the icar branch.

## Table of contents

* [Introduction](#introduction)
* [Requirements](#requirements)
* [How to use](#how-to-use)
* [Resources](#resources)
* [Reference](#reference)
* [License](#license)

## Introduction

This repository contains the implementation of the [power-constrained coverage path planning (CPP) with recharge problem](https://arxiv.org/abs/2309.03157)
and the proposed PPO-based deep reinforcement learning (DRL) solution. The DRL approach utilizes map-based observations,
preprocessed as global and local maps, action masking to ensure safety, discount factor scheduling to optimize the
long-horizon problem, and position history observations to avoid state loops.

![Screenshot of the evaluation](./screenshots/example.png)

The agents are stored in a submodule and can be pulled by
```commandline
git submodule init
git submodule pull
```

For questions, please contact Mirco Theile via email mirco.theile@tum.de.

## Requirements

```
tensorflow~=2.11.0
opencv-python==4.7.0.68
scikit-image==0.19.3
gymnasium==0.27.0
pygame==2.5.1
tqdm~=4.64.1
seaborn==0.12.2
dataclasses-json==0.5.7
einops~=0.6.1
```

Developed and tested only on Linux and MacOS.

## How to use

With this repository PPO agents can be trained to solve the power-constrained CPP problem with recharge. Additionally,
newly trained and example agents can be evaluated with a visualization.

### Training

#### General usage:

```
python train.py [-h] [--gpu] [--gpu_id GPU_ID] [--generate] [--verbose] [--params [PARAMS ...]] config

positional arguments:
  config                Path to config file

options:
  -h, --help            show this help message and exit
  --gpu                 Activates usage of GPU
  --gpu_id GPU_ID       Activates usage of GPU on specific GPU id
  --generate            Generate config file for parameter class
  --verbose             Prints the network summary at the start
  --params [PARAMS ...]
                        Override parameters as: path/to/param1 value1 path/to/param2 value2 ...

```

#### How to recreate all the agents used in the paper:

Normal Agents:

- Multi3 ```python train.py --gpu config/multi3.json```
- Multi10 ```python train.py --gpu config/multi10.json```
- Suburban ```python train.py --gpu config/suburban.json```
- Castle ```python train.py --gpu config/castle.json```
- TUM ```python train.py --gpu config/tum.json```
- Cal ```python train.py --gpu config/cal.json```
- Manhattan ```python train.py --gpu config/manhattan.json```

Mask Ablation:

- No
  Mask ```python train.py --gpu config/multi3.json --params gym/action_masking none trainer/gamma/decay_rate 1.0 --id no_mask```
- Valid Mask
  ```python train.py --gpu config/multi3.json --params gym/action_masking valid trainer/gamma/decay_rate 1.0 --id valid```
- Immediate Mask
  ```python train.py --gpu config/multi3.json --params gym/action_masking immediate trainer/gamma/decay_rate 1.0 --id immediate```
- Invariant Mask ```python train.py --gpu config/multi3.json --params trainer/gamma/decay_rate 1.0 --id invariant```

Discount Scheduling Ablation:

- $\gamma_0=0.99$, $\gamma_s=\infty$
  ```python train.py --gpu config/multi3.json --params trainer/gamma/decay_rate 1.0 --id gamma_099```
- $\gamma_0=0.999$, $\gamma_s=\infty$
  ```python train.py --gpu config/multi3.json --params trainer/gamma/base 0.999 trainer/gamma/decay_rate 1.0 --id gamma_0999```
- $\gamma_0=1.0$, $\gamma_s=\infty$
  ```python train.py --gpu config/multi3.json --params trainer/gamma/base 1.0 trainer/gamma/decay_rate 1.0 --id gamma_1```
- $\gamma_0=0.99$, $\gamma_s=8\times 10^7$
  ```python train.py --gpu config/multi3.json --params trainer/gamma/decay_rate 2000 --id gamma_decay_2k```
- $\gamma_0=0.99$, $\gamma_s=2\times 10^7$
  ```python train.py --gpu config/multi3.json```

Position History Ablation:

- No Position History (
  Base) ```python train.py --gpu config/multi3.json --params gym/position_history 0 --id no_history```
- Random
  Layer ```python train.py --gpu config/multi3.json --params gym/position_history 0 gym/random_layer 1 --id random_layer```
- Position History ```python train.py --gpu config/multi3.json```

### Evaluating

#### General Usage

```
python evaluate.py [-h] [-a [A ...]] [-t [T ...]] [-d] [-r [R ...]] [--scenario SCENARIO] [--all_maps] [--heuristic] [--maps_only] [--gpu] [--gpu_id GPU_ID] [--generate] [--verbose] [--params [PARAMS ...]] config

positional arguments:
  config                Path to config file

options:
  -h, --help            show this help message and exit
  -a [A ...]            Add maps
  -t [T ...]            Add timeouts for maps, 1000 otherwise
  -d                    remove all other maps
  -r [R ...]            Record episode only, potentially override render params
  --scenario SCENARIO   Load specific scenario
  --all_maps            Load all maps
  --heuristic           Use Heuristic Only
  --maps_only           Draws maps only
  --gpu                 Activates usage of GPU
  --gpu_id GPU_ID       Activates usage of GPU on specific GPU id
  --generate            Generate config file for parameter class
  --verbose             Prints the network summary at the start
  --params [PARAMS ...]
                        Override parameters as: path/to/param1 value1 path/to/param2 value2 ...
```

For instructions in the interactive evaluation environment press the ```h``` key.

#### Recreate scenarios in the paper:

To record the videos and log the final trajectory and statistics add ```-r```. It will run in the background. 

Figure 2:

- a) ```python evaluate.py multi3_no_hist --scenario short_loop```
- b) ```python evaluate.py multi3_no_hist --scenario long_loop```

Figure 7:

- a) ```python evaluate.py manhattan --scenario decomp2```
- b) ```python evaluate.py manhattan --scenario decomp2 --heuristic```
- c) ```python evaluate.py manhattan --scenario decomp3```
- d) ```python evaluate.py manhattan --scenario decomp3 --heuristic```

Figure 8:

- a) ```python evaluate.py tum --scenario tum1```
- b) ```python evaluate.py tum --scenario tum2```

Figure 9:

- a) ```python evaluate.py multi3 --scenario suburban```
- b) ```python evaluate.py suburban --scenario suburban```
- c) ```python evaluate.py multi3 --scenario castle```
- d) ```python evaluate.py castle --scenario castle```
- e) ```python evaluate.py multi3 --scenario tum```
- f) ```python evaluate.py tum --scenario tum```

Figure 10:

- a) ```python evaluate.py multi10 --scenario castle2 -a castle2```
- b) ```python evaluate.py castle --scenario castle2 -a castle2```

Figure 11:

- d) ```python evaluate.py multi10 --scenario cal -a cal42```
- h) ```python evaluate.py cal --scenario cal```

Figure 12:

- a) ```python evaluate.py multi10 --scenario border -a hard```
- b) ```python evaluate.py border --scenario border```

## Resources

The maps from the paper are included in the 'res' directory. Map information is
formatted as PNG files with one pixel representing one grid-world cell. The pixel color determines the type of cell
according to

* red #ff0000 no-fly zone (NFZ)
* blue #0000ff start and landing zone
* yellow #ffff00 buildings blocking field-of-view (FoV)

If you would like to create a new map, you can use any tool to draw a PNG with the same pixel dimensions as the
desired map and the above color codes.

When maps are loaded for the first time, a model is computed that is later used by the FoV calculation, action mask, and
heuristic. The model is saved as 'res/[map_name]_model.pickle'. For large maps, this process may take a few minutes.

## Reference

If using this code for research purposes, please cite:

```
@misc{theile2023learning,
      title={Learning to Recharge: UAV Coverage Path Planning through Deep Reinforcement Learning}, 
      author={Mirco Theile and Harald Bayerlein and Marco Caccamo and Alberto L. Sangiovanni-Vincentelli},
      year={2023},
      eprint={2309.03157},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## License

This code is under a BSD license.
