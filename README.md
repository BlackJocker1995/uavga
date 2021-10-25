# uavga
This is an approach source code of LGDFuzzer.

## Requirement
Python package requirement: numpy ; pandas ; pymavlink ; pyulog ; eventlet ; keras ; tensorflow

OS: The program is only test in Ubuntu 18.04.

`
pip3 install pymavlink pandas pyulog eventlet keras tensorflow
`


Simulation requirement: [SITL](https://github.com/ArduPilot/ardupilot).
The initializer of simulator needs to change the path in the file `Cptool.gaSimManager.py` with function `start_sitl` in line34.
For example,
`
python3 {Your Ardupilot path}/Tools/autotest/sim_vehicle.py --location=AVC_plane --out=127.0.0.1:14550 -v ArduCopter -w -S {toolConfig.SPEED} "
`.



## Deployment
The configuration is in `Cptool.config.py` and `ModelFit.config.py`


## Description


`train_Lstm.py` train a model predictor.

`lgfuzzer.py` start the fuzzing test.

## Other

Train Data Set: https://drive.google.com/drive/folders/1bbRqWWUEuyfu8mubMBMaLD_QARP82P4x?usp=sharing

Video of flight test: https://youtube.com/playlist?list=PLDDY9yM5Ac0Dh5o1R40Hs8lobhD8E3yil

error flight example data log: https://drive.google.com/drive/folders/1VTKvvgNNdIG2kvr4cJ2WeiPsi3kpviaS?usp=sharing