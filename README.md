# Requirement : 

  Ray : 2.8.1
  python : 3.8.10
  Gymnasium : 0.28.1
# Install 

  pip install "ray[rllib]" tensorflow torch
  pip install ray == 2.8.1
  
# Generic Platform
- The goal is to create a generic platform to train plenty IAs on environments.

## Organisation
- First you will find one folder "Scenarios" and 2 files "start.py" and utils_platfrom.py
![Screenshot from 2024-03-07 11-51-39](https://github.com/ELTGR/generic_platform/assets/122261448/4be9f11b-0222-47c6-b61e-dcef707652e0)

### Launch

- Experiments.py is the file where you can find all the train, test function.
  
### Environments
- You will find 2 folders, one is the folder "UUV_Mono_Agent_TSP", it's an example of Scénario.

- 
![Screenshot from 2024-03-07 11-52-47](https://github.com/ELTGR/generic_platform/assets/122261448/8a70cc7a-2d98-494e-9ac2-fa0aaf13fef5)


- The second "Scenario_Exemple" is a esqueleton of your environement, copie and paste it to create your owne environment.

  
![Screenshot from 2024-03-07 11-51-03](https://github.com/ELTGR/generic_platform/assets/122261448/b33b46b3-39cd-4554-b5e7-e2e6d47ceb81)


- Inside bridge.py you will find 2 class Simple Implementation and RealImplementation. Simple is for the training, it's where to fake a real vehicle during the training part. Real is for the testing part, but you can also test with simple implementation


# Prérequis ↓↓ documentation pour installation ↓↓

- Ubuntu 20
- python 3.8
- QgroundControle
- ArduPilot avec ArduSub
- PyMavlink
- PyNav
- Unity 2021.3.18f1
- ROS Noetic



## Visual Studio Code
To work on this projet we use Visual Studio Code. It's a simple IDE witch allow to work with different programming languages ( Where we mainly use Python).  To install Visual studio code on Ubuntu, 2 solution :

- 1 use the Ubuntu Software, search "code" et launch the installation.

- 2 Download the .deb where : https://code.visualstudio.com/download and run this command in a terminal :

         sudo apt install ./<file>.deb
         
         
Once the IDE launch, go into Extensions and install **Python**.


<img src="https://user-images.githubusercontent.com/122261448/231142348-6e99f545-2b9b-48b0-9ec5-f0dc8ae71dc9.png" width="400" />



## Installer Python 3.8 : 
- update to have the last version of Python :

         sudo apt update
         sudo apt -y upgrade
                  
- To verify if the good version of Python is install run : 

         python3 -V
         
the result must be : Python 3.8.10

## Installation de ROS Noetic
- add packages.ros.org to the sources list :

         sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

         sudo apt install curl 

         curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
         
- Install ROS Noetic package :

         sudo apt update
         sudo apt install ros-noetic-desktop-full
         
- Sources automatisation :

         source /opt/ros/noetic/setup.bash

         echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

         source ~/.bashrc
         
- Python3/ROS dependencies

         sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

         sudo apt install python3-catkin-tools python3-osrf-pycommon
         
- Rosdep initialisation  :

         sudo rosdep init

         rosdep update
         
- Cloning the main repository:
 
      git clone  https://github.com/ELTGR/DRL_platform_montpellier.git

- Ceate a dircetory name " catkin_ws " into the main directory in witch one create a directory name " src "

- Clone the ROS TCP Endpoint packages into src :

      cd DRL_platform_montpellier/catkin_ws/src

      git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
      
- Go into the DRL_platform_montpellier/catkin_ws directory and run : 

         catkin build
         
 - You must obtain this:    
      
 <img src="https://user-images.githubusercontent.com/122261448/231142898-501e7eb3-9679-43fc-abac-b45a204b17c0.png" width="600" /> 

- Update the environment variables

      echo 'source $HOME/DRL_platform_montpellier/catkin_ws/devel/setup.bash' >> ~/.bashrc

      source ~/.bashrc
      
## Install Qgroundcontrol 

- In case of probleme refer to this documentation : https://docs.qgroundcontrol.com/master/en/getting_started/download_and_install.html

         sudo usermod -a -G dialout $USER
         
         sudo apt-get remove modemmanager -y
         
         sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl -y
         
         sudo apt install libqt5gui5 -y
         
         sudo apt install libfuse2 -y
         
- Logout and login to make effective the changes.
 
 Download QGroundControl.AppImage : https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage

 - To launch Qground : 
 
         chmod +x ./QGroundControl.AppImage
         ./QGroundControl.AppImage  (or double click)
         
               
## Install ArduPilot

- In case of probleme refer to this documentation : https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux

         sudo apt-get update

         sudo apt-get install git

         sudo apt-get install gitk git-gui       
  
         git clone https://github.com/your-github-userid/ardupilot
         
         cd ardupilot
         
         git submodule update --init --recursive
      
         Tools/environment_install/install-prereqs-ubuntu.sh -y
 
         . ~/.profile
         
- Logout and login to make effective the changes.
## Install Ray
- In case of probleme refer to this documentation : https://ardupilot.org/dev/docs/building-setup-linux.html
  
        pip install "ray[rllib]" tensorflow torch
        pip install ray == 2.8.1

  ## Install Stable Baselines3 
- In case of probleme refer to this documentation : https://stable-baselines3.readthedocs.io/en/master/guide/install.html
  
        pip install stable-baselines3[extra]

## Install PyMavlink
- In case of probleme refer to this documentation : https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux

         sudo python -m pip install --upgrade future lxml

## Install Navpy   :

- Open a terminal and run :

          pip install NavPy

### Install Unity 
:
- Add Unity to the repository : 

         sudo sh -c 'echo "deb https://hub.unity3d.com/linux/repos/deb stable main" > /etc/apt/sources.list.d/unityhub.list'
         
- Public key :

         wget -qO - https://hub.unity3d.com/linux/keys/public | sudo apt-key add -
         
- Install and Update Unity :

         sudo apt update
         sudo apt-get install unityhub
     
### Install 2021.3.18f1:

- Launch UnityHub and ingnore the proposition. In **Installs** click on **Install Editor**.
Go into **Archive**   click **download archive**, on the website look for the **version 2021.3.18**.Ones found click **Unity Hub**. The download must start into UnityHub.

- We clone the repository of the Unity word (attention don't clone it into the Bluerov2_dock_scan):

         git clone https://github.com/Gregtmlg/Mission_Planning.git
         
- We will verify if all usefull Unity packages are install. For this go in Window > Manager. 
- In projet list packages, verify the presence of : Burst, Addressables and ROS TCP Connector. 

### Install Unity Repository :


## Explanation of the communication system

<img src="https://user-images.githubusercontent.com/122261448/230575616-42fc961a-43e4-4340-aa1e-d88e82931272.jpg" width="1000" />

###      Communication between the BlueRov2 and Python script : 

<img src="https://user-images.githubusercontent.com/122261448/230575819-90deafe4-0a40-43dd-98d3-a16354ca939a.jpg" width="450" />

To communicate with the ROV we use PyMavlink. This library provides us some usefull fonction to received and send data. 

Look script " bridge.py " 

First we have to establish the communication settings

<img src="https://user-images.githubusercontent.com/122261448/230570370-56251835-b07b-47f3-926b-f53c2280552f.png" width="600" />

Where we have enter the main communication setting like adresse and baurate.
Next with the **get_data()**, **get_all_msgs()** and **update()** will allow us to received all the ROV data.
In **get_bluerov_position_data** we decide to get some precise data like x, y, z, yaw witch are the rov position and orientation in the NED.
To give some positions order to the ROV, we use the fonction **set_position_target_local_ned()** 

<img src="https://user-images.githubusercontent.com/122261448/231151892-54c8cdc2-0be4-4415-a288-238422a0ce33.png" width="400" />


###       Communication between Python script and ROS : 

<img src="https://user-images.githubusercontent.com/122261448/230576048-76cbda38-1670-4e5e-a84a-b92fcad3e9b2.jpg" width="400" />
ROS allow us to create topics in witch some messages can be write or read. Thanks to that some programme witch don't have the same programmation language can communicate. 
It's exatly what we do here. We transfere some data of our Python script into a C# script to update the ROV position in Unity.

Look script  " bluerov_node.py  " 
In **__init__** into the class **BlueRov** we define a topic named **/local_position**. It's defined in **pud** that means we will write without read what the topic containe.
In **def_create_position_msg(self)** we received the ROV position data from PyMavlink.

At the end of the fonction we publish the data into the topic previously defined




###      Communication between ROS and the C# script of Unity : 

<img src="https://user-images.githubusercontent.com/122261448/230582932-8fa5c320-011b-4071-a8f7-b3d256a838ed.jpg" width="400" />

To update the ROV position into Unity we use a C# script named **UpdateBluerovPose.cs** this script is in the **Asset** directory of the Unity projet.
To make the link between the ROV and the file, you just have to create an empty box, click on it, go into properties and clikc on **add Component** and select the script. Next put the 3D ROV model into the empty box.

<img src="https://user-images.githubusercontent.com/122261448/231139812-f7cb5af1-342d-492d-8968-53d116716424.png" width="1200" />

We now look how read the data of the ROS topic.

Look  " UpdateBluerovPose.cs " script the **Asset** directory of the Unity projet. 

<img src="https://user-images.githubusercontent.com/122261448/230583657-7248f54e-ba7d-4c5c-8d00-0ac3c9f1fcc7.png" width="600" />

In the fonction **void Start()** we create an **Subscribre** witch will read the data of the **/BlueRov2/local_position** topic.
We can found the callback named **OdomChange** witch allow us to received the infomations.The rest of the script make some conversion. 



### Tensorboard : 
  In VScode go in Help -> Show All Commandes -> shearch Python : Launch Tensor Board -> Select an other forlder and select the folder where the modeles are saved. For RLlib its in home/ray_results. For Stable baselines its in the folde where you saved the modeles
  
## Start the simulation 

- Open a terminal, launch the ROS TCP Endpoint with :
 
                  roslaunch ros_tcp_endpoint endpoint.launch

" -S " is the speed control  1 is the value.  if u want to increase it, go into /home/%sessions_name%/.local/bin/mavproxy.py and modify the hearthbeat value. Then launch the command with the same value write in the mavproxy.py script

- Launch QgroundControl.

- Launch the Unity projet.
