## Table of contents

* [Introduction](#introduction)
* [Requirements](#requirements)
* [How to use](#how-to-use)
* [Resources](#resources)
* [Reference](#reference)
* [License](#license)

## Introduction

This repository contains an implementation of the double deep Q-learning (DDQN) approach to control a UAV on a coverage path planning or data harvesting from IoT sensors mission, including global-local map processing. The corresponding paper ["UAV Path Planning using Global and Local Map Information with Deep Reinforcement Learning"](https://ieeexplore.ieee.org/abstract/document/9659413) is available on IEEExplore. A multi-agent version can be found in ["uav_data_harvesting"](https://github.com/hbayerlein/uav_data_harvesting)

For questions, please contact Mirco Theile via email mirco.theile@tum.de. Please also note that due to github's new naming convention, the 'master' branch is now called 'main' branch.


## Requirements

```
python==3.7 or newer
numpy==1.18.5 or newer
keras==2.4.3 or newer
tensorflow==2.3.0 or newer
matplotlib==3.3.0 or newer
scikit-image==0.16.2 or newer
tqdm==4.45.0 or newer
```


## How to use

Train a new DDQN model with the parameters of your choice in the specified config file for Coverage Path Planning (CPP) or Data Harvesting (DH):

```
python main.py --cpp --gpu --config config/manhattan32_cpp.json --id manhattan32_cpp
python main.py --dh --gpu --config config/manhattan32_dh.json --id manhattan32_dh

--cpp|--dh                  Activates CPP or DH
--gpu                       Activates GPU acceleration for DDQN training
--config                    Path to config file in json format
--id                        Overrides standard name for logfiles and model
--generate_config           Enable only to write default config from default values in the code
```

Evaluate a model through Monte Carlo analysis over the random parameter space for the performance indicators 'Successful Landing', 'Collection Ratio', 'Collection Ratio and Landed' as defined in the paper (plus 'Boundary Counter' counting safety controller activations), e.g. for 1000 Monte Carlo iterations:

```
python main_mc.py --cpp --weights example/models/manhattan32_cpp --config config/manhattan32_cpp.json --id manhattan32_cpp_mc --samples 1000
python main_mc.py --dh --weights example/models/manhattan32_dh --config config/manhattan32_dh.json --id manhattan32_dh_mc --samples 1000

--cpp|--dh                  Activates CPP or DH
--weights                   Path to weights of trained model
--config                    Path to config file in json format
--id                        Name for exported files
--samples                   Number of Monte Carlo  over random scenario parameters
--seed                      Seed for repeatability
--show                      Pass '--show True' for individual plots of scenarios and allow plot saving
```

For an example run of pretrained agents the following commands can be used:
```
python main_scenario.py --cpp --config config/manhattan32_cpp.json --weights example/models/manhattan32_cpp --scenario example/scenarios/manhattan_cpp.json --video
python main_scenario.py --cpp --config config/urban50_cpp.json --weights example/models/urban50_cpp --scenario example/scenarios/urban_cpp.json --video

python main_scenario.py --dh --config config/manhattan32_dh.json --weights example/models/manhattan32_dh --scenario example/scenarios/manhattan_dh.json --video
python main_scenario.py --dh --config config/urban50_dh.json --weights example/models/urban50_dh --scenario example/scenarios/urban_dh.json --video
```

## Resources

The city environments from the paper 'manhattan32' and 'urban50' are included in the 'res' directory. Map information is formatted as PNG files with one pixel representing on grid world cell. The pixel color determines the type of cell according to

* red #ff0000 no-fly zone (NFZ)
* blue #0000ff start and landing zone
* yellow #ffff00 buildings blocking wireless links (also obstacles for flying)

If you would like to create a new map, you can use any tool to design a PNG with the same pixel dimensions as the desired map and the above color codes.

The shadowing maps, defining for each position and each IoT device whether there is a line-of-sight (LoS) or non-line-of-sight (NLoS) connection, are computed automatically the first time a new map is used for training and then saved to the 'res' directory as an NPY file. The shadowing maps are further used to determine which cells the field of view of the camera in a CPP scenario.


## Reference

If using this code for research purposes, please cite:

[1] M. Theile, H. Bayerlein, R. Nai, D. Gesbert, M. Caccamo, “UAV Path Planning using Global and Local Map Information with Deep Reinforcement Learning" 20th International Conference on Advanced Robotics (ICAR), 2021. 

```
@inproceedings{theile2021uav,
  title={UAV path planning using global and local map information with deep reinforcement learning},
  author={Theile, Mirco and Bayerlein, Harald and Nai, Richard and Gesbert, David and Caccamo, Marco},
  booktitle={2021 20th International Conference on Advanced Robotics (ICAR)},
  pages={539--546},
  year={2021},
  organization={IEEE}
}
```

for the multi-agent Data Harvesting paper:

[2] H. Bayerlein, M. Theile, M. Caccamo, and D. Gesbert, “Multi-UAV path planning for wireless data harvesting with deep reinforcement learning," IEEE Open Journal of the Communications Society, vol. 2, pp. 1171-1187, 2021.

```
@article{bayerlein2021multi,
  title={Multi-uav path planning for wireless data harvesting with deep reinforcement learning},
  author={Bayerlein, Harald and Theile, Mirco and Caccamo, Marco and Gesbert, David},
  journal={IEEE Open Journal of the Communications Society},
  volume={2},
  pages={1171--1187},
  year={2021},
  publisher={IEEE}
}
```



## License 

This code is under a BSD license.
