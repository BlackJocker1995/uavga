# ICSearcher and LGDFuzzer
This is an approach source code of ICSearcher.

The original code of the [paper(LGDFuzzer)](https://dl.acm.org/doi/10.1145/3510003.3510084) is in branch [lgdfuzzer](https://github.com/BlackJocker1995/uavga/tree/lgdfuzzer)

ICSearcher is an improved version of LGDFuzzer.

# Log
Update: 22-07-15, support px4

## Requirement
Python package requirement: numpy ; pandas ; pymavlink ; pyulog ; keras ; tensorflow

OS: The program is only test in Ubuntu 18.04 and 20.04 (recommend).

`
pip3 install pymavlink pandas pyulog eventlet keras tensorflow
`


Simulation requirement: Ardupilot [SITL](https://github.com/ArduPilot/ardupilot). We suggest applying python3 to run STIL simulator.
Jmavsim for PX4, which requires source build in PX4 file.

The initializer of Ardupilot simulator needs to change the path in the file `Cptool.config.py` with item
`SITL_PATH`.

For example,
`
python3 {Your Ardupilot path}/Tools/autotest/sim_vehicle.py --location=AVC_plane --out=127.0.0.1:14550 -v ArduCopter -w -S {toolConfig.SPEED} "
`


## Deployment
The configuration is in `Cptool.config.py`.

If you want to try PX4 simulation, change the sentence `toolConfig.select_mode("Ardupilot")` to `toolConfig.select_mode("PX4")`

## Configuration of System config.py
* ARDUPILOT_LOG_PATH: log path of ardupilot running, noted that, the path needs to have a flag file "logs/LASTLOG.TXT".
Or you can manually run the simulation at first in {ARDUPILOT_LOG_PATH} to auto generate flag file. 

The log path for PX4 is in `{PX4_Path}/build/px4_sitl_default/logs/`, which is no need to change.

* SIM: simulation type.

* AIRSIM_PATH: if select airsim, you should set the execution path.

* PX4_RUN_PATH: if select PX4, you should set the execution path.

* PARAM: the parameter used in predictor.

* PARAM_PART: the parameter that participate in fuzzing.

* INPUT_LEN: input length of predictor.


## Description

`0.collect.py` start simulation to collect flight logs.

`1.trans_bin2csv.py` transform the bin file to csv.

`2.extract_feature.py` extract feature from csv.

`2.raw_split.py` split the test feature for further searcher.

`2.feature_split.py` split the csv data for train and test.

`2.train_lstm.py` train a model predictor.

`3.lgfuzzer.py` start the fuzzing test.

`4.pre_validate.py` select candidates.

`4.validate.py` validate configurations through simulator.

If you want to validate with multiple simulator, you can use validate.py -- device {xxx} to start multiple SITL

`4.validate_thread.py` validate configurations through multiple simulators, where use --thread {xx} to launch multiple tab validate.py

`5.range.py` summary range guideline by validated result.
