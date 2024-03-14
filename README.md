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
- First you will find one folder "environments" and 2 files "start.py" and utils_platfrom.py
![Screenshot from 2024-03-05 13-16-53](https://github.com/ELTGR/generic_platform/assets/122261448/50d39fbc-71fc-4645-af29-9f51f916a1c8)
### Lauchn
- Start. py is useful to train, test your IA in a simple way.
- utils_platfrom. py is the file where you can find all the train, test function but less simple.
### Environments
- You will find 2 folders, one is the folder "UUV_Mono_Agent", it's an example of environment.


![image](https://github.com/ELTGR/generic_platform/assets/122261448/492a543b-f581-4563-b643-62b21f8b270e)









- The second "My_Custom_Env" is a esqueleton of your environement, copie and paste it to create your owne environment.


![Screenshot from 2024-03-05 13-30-54](https://github.com/ELTGR/generic_platform/assets/122261448/3a438b89-a1ef-470f-8e8b-a158e8c8fa72)

- Inside implementations. py you will find 2 class Simple Implementation and RealImplementation. Simple is for the training, it's where to fake a real vehicle during the training part. Real is for the testing part, but you can also test with simple implementation
