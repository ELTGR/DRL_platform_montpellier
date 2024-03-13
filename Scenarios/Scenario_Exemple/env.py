

import gymnasium as gym

from Scenarios.Scenario_Exemple.view2d import MyCustomEnv2dView
from Scenarios.Scenario_Exemple.bridge import Agent
from Scenarios.Scenario_Exemple.utils_env import  UtilsMyCustomEnv
'''
In this file, you must write the 4 functions: init, reset, step and render.
Other functions may be written in the corresponding Utils  to your envy, where UtilsMyCustomEnv
'''
class MyCustomEnv(gym.Env) : 

    def __init__(self,env_config) :
        
        self.agent = Agent(implementation=self.implementation)
        self.utils = UtilsMyCustomEnv()
        pass
  
    def reset(self):
        pass
          
    def step(self, action):
        pass
   
    def render(self):
            self.scenario2DView = MyCustomEnv2dView(self.env_config["implementation"])
            pass

