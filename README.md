# uavga
This is an approach source code of LGDFuzzer.

## Requirement
Python package requirement: numpy ; pandas ; pymavlink ; pyulog ; eventlet ; keras ; tensorflow

`
pip3 install pymavlink pandas pyulog eventlet keras tensorflow
`


Simulation requirement: [Airsim](https://github.com/Microsoft/AirSim/releases) or [SITL](https://github.com/ArduPilot/ardupilot)

## Deployment
The configuration is in Cptool.config.py

If you use the SITL, please change the start py file path at Cptool.gaSimManager.py Line-49


## Description





train_Lstm.py train a model predictor.

lgfuzzer.oy start the fuzzing test.

## Other

Train Data Set: https://drive.google.com/drive/folders/1bbRqWWUEuyfu8mubMBMaLD_QARP82P4x?usp=sharing

Video of flight test: https://youtube.com/playlist?list=PLDDY9yM5Ac0Dh5o1R40Hs8lobhD8E3yil

LSTM Training data: https://drive.google.com/drive/folders/1VTKvvgNNdIG2kvr4cJ2WeiPsi3kpviaS?usp=sharing