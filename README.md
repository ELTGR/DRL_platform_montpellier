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
