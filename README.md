[![Build Status](https://travis-ci.com/PurdueCAM2Project/Embedded2.svg?branch=master)](https://travis-ci.com/PurdueCAM2Project/Embedded2)

# Embedded Computer Vision 2
System is used to detect usage of Personal Protection Equipment (PPE), specifically goggles, in labs that require them. System is ran in real time on Jetson Nano and uses a Rasspberry Pi camera to record footage in the lab. To ensure individual privacy is protected, system obfuscates faces after detection and classification. Images are stored in a remote storage drive and image metadata are stored on a SQL database.

#### Features
* Retinaface based SSD performs face detection
* CNN performs classification of detected faces to determine if PPE is being used
* Faces are encrypted using AES
* Image metadata is stored on a SQL database server
* Images are transfered to remote computer using SFTP
# Table of Contents
- [Description](#Embedded-Computer-Vision-2)
- [Table of Contents](#Table-of-Contents)
- [Installation](#Installation)
- [Usage](#Contributing)
- [Credits](#Credits)
- [License](#License)
# Installation
1. Clone the project and enter the folder 
```shell
$ git clone https://github.com/PurdueCAM2Project/Embedded2.git
$ cd Embedded2
```
2. The classiifier model (.pth file) can be found on [Drive](https://drive.google.com/drive/u/1/folders/1ZeKVygo-RyIDL_EnxeYJR8tk-xqzgi3Z). Downloadand place it in the ```Embedded2/src/jetson``` directory.
3. There is a requirement.txt file with all the necessary dependencies. We, however, recommend using Conda for this project. Once you have conda installed, run the following command to setup the enviroment with necessary dependicies.
```shell
$ conda env create -f environment.yml
```
4. Add the Embedded2 folder to PYTHONPATH by adding the following line in your .bashrc file:
```export PYTHONPATH=/path/Embedded2```
# Usage
# Contributing
# Credits
# License
