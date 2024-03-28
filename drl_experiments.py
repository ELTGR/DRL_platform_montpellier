import ray
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import (
    PPOConfig,
    PPOTF1Policy,
    PPOTF2Policy,
    PPOTorchPolicy,
)
#test branche opti
from time import sleep
from gymnasium import spaces
from ray import tune    
from ray.air import CheckpointConfig
import subprocess
import time
import onnxruntime
import numpy as np
import os, sys 
from Scenarios.Multi_Agents_Supervisor_Operators.env import MultiAgentsSupervisorOperatorsEnv 
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
class DrlExperimentsTune():

    def __init__(self, env, env_config) :
       
        self.env_config = env_config
        self.env_type= env

        
    def tune_train(self, train_config) : 

        config = (PPOConfig() 
            
                .environment(self.env_type,env_config=self.env_config,
                                disable_env_checking=True)

                .resources(num_learner_workers=train_config["num_learner_workers"],
                            num_cpus_per_worker=train_config["num_cpus_per_worker"]
                            )
                )



        # Lancez TensorBoard en utilisant subprocess
           
        # Lancez TensorBoard en utilisant un nouveau terminal (Linux/Mac)
        #tensorboard_command = f"x-terminal-emulator -e tensorboard --logdir="+str(self.ray_path)
        
        #process_terminal_1 = subprocess.Popen(tensorboard_command, shell=True)
        #time.sleep(2)
        self.env_config['implementation'] = "simple"
    
        ray.init()
        algo = tune.run("PPO",
                        name = str(self.env_config["num_boxes_grid_width"])+"x"+str(self.env_config["num_boxes_grid_height"])+"_"+str(self.env_config["n_orders"])+"_"+str(self.env_config["step_limit"]),
                        config = config,
                        stop = {"timesteps_total": train_config["stop_step"]}, 
                        checkpoint_config = CheckpointConfig(checkpoint_at_end=True,
                        checkpoint_frequency=train_config["checkpoint_freqency"] ),
                        storage_path=train_config["storage_path"]
                        )
    
    def tune_train_multi_agent(self, train_config) : 
            
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                agent_type = agent_id.split('_')[0]
            
                if agent_type == "supervisor" :
                
                    return "supervisor_policy"
                else :
                
                    return "operator_policy" 


        config2 = (PPOConfig() 
                
                    .environment(MultiAgentsSupervisorOperatorsEnv,env_config=self.env_config,
                                 disable_env_checking=True)

                    .resources(num_learner_workers=train_config["num_learner_workers"],
                               num_cpus_per_worker=train_config["num_cpus_per_worker"]
                               )

                    .multi_agent(policies=env_config["policies"],
                                 policy_mapping_fn=policy_mapping_fn,)
                    )


        algo = tune.run("PPO", name =  str(self.env_config["num_boxes_grid_width"])+"x"+str(self.env_config["num_boxes_grid_height"])+"_"+str(self.env_config["n_orders"])+"_"+str(self.env_config["step_limit"]),
                        config = config2,
                        stop = {"timesteps_total": train_config["stop_step"]}, 
                        checkpoint_config = CheckpointConfig(checkpoint_at_end=True,
                                                             checkpoint_frequency=train_config["checkpoint_freqency"] ),
                        storage_path = train_config["storage_path"])
                                                                      
    def tune_train_from_checkpoint(self, train_config, multi_agent = False, checkpointpath = None):

        self.env_config['implementation'] = "simple"
        ray.init()

        if multi_agent == False : 
            config = (PPOConfig() 
                
                    .environment(self.env_type,env_config=self.env_config,
                                    disable_env_checking=True)

                    .resources(num_learner_workers=train_config["num_learner_workers"],
                                num_cpus_per_worker=train_config["num_cpus_per_worker"]
                                )
                    )
            
        elif multi_agent == True :

            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                agent_type = agent_id.split('_')[0]
            
                if agent_type == "supervisor" :
                
                    return "supervisor_policy"
                else :
                
                    return "operator_policy" 
            
   
           
            config = (PPOConfig() 
                    
                        .environment(MultiAgentsSupervisorOperatorsEnv,env_config=self.env_config,
                                    disable_env_checking=True)

                        .resources(num_learner_workers=train_config["num_learner_workers"],
                                num_cpus_per_worker=train_config["num_cpus_per_worker"]
                                )

                        .multi_agent(policies=env_config["policies"],
                                    policy_mapping_fn=policy_mapping_fn,)
                        )



        algo = tune.run("PPO",
                        name = "from_checkpoint" + str(self.env_config["num_boxes_grid_width"])+"x"+str(self.env_config["num_boxes_grid_height"])+"_"+str(self.env_config["n_orders"])+"_"+str(self.env_config["step_limit"]),
                        config = config,
                        stop = {"timesteps_total": train_config["stop_step"]}, 
                        checkpoint_config = CheckpointConfig(checkpoint_at_end=True,checkpoint_frequency=train_config["checkpoint_freqency"] ),
                        storage_path=train_config["storage_path"],restore=checkpointpath
                        )
           
    def test(self, implementation, multi_agent = False, checkpointpath = None) :
            
        if multi_agent == False :
             # Lancez TensorBoard en utilisant un nouveau terminal (Linux/Mac)
            
            self.env_config['implementation'] = implementation 
            
            if self.env_config['implementation'] == "real": 
                ros_tcp_endpoint = f"x-terminal-emulator -e roslaunch ros_tcp_endpoint endpoint.launch"
                process_terminal_1 = subprocess.Popen(ros_tcp_endpoint, shell=True)

            print("config : ",self.env_config)

            env = self.env_type(env_config = self.env_config)
            loaded_model = Algorithm.from_checkpoint(checkpointpath)
            agent_obs = env.reset()
            print("obs",agent_obs)
            env.render()

            while True : 

                action =  loaded_model.compute_single_action(agent_obs)
                print(action)
                agent_obs, reward, done, info = env.step(action)
                print("obs",agent_obs)
                print("obs",reward)
            
                env.render()

                if done :
                    env = self.env_type(env_config=self.env_config)
                    agent_obs = env.reset()
                    print("obs",agent_obs)
                    env.render()
        elif multi_agent == True : 

            def inference_policy_mapping_fn(agent_id):
                agent_type = agent_id.split('_')[0]
                if agent_type == "supervisor" :

                    return "supervisor_policy"

                else :

                    return "operator_policy"
          

           
            
            algo = Algorithm.from_checkpoint(checkpointpath)
            env = self.env_type(env_config = self.env_config)
            obs = env.reset()
            print(obs)

            num_episodes = 0
            num_episodes_during_inference =100

        
            episode_reward = {}

            while num_episodes < num_episodes_during_inference:
                num_episodes +=1 
                action = {}
                print("next step : ",num_episodes)

                
                for agent_id, agent_obs in obs.items():
                    
                    policy_id = inference_policy_mapping_fn(agent_id)
                    action[agent_id] = algo.compute_single_action(observation=agent_obs, policy_id=policy_id)

                print(action)
                obs, reward, done, info = env.step(action)
                print("next step : ",num_episodes)

                for id, thing in obs.items() :
                    print("id",id,":",thing) 

                for id, thing in reward.items() :
                    print("id :",id,":",thing)
        
                env.render() 

    def export_to_onnx(self, policies = DEFAULT_POLICY_ID , checkpointpath = None, export_dir = None) : 
        for i in range(len(policies)) : 

            loaded_algo = Algorithm.from_checkpoint(checkpointpath)
            loaded_algo.export_policy_model(policy_id=policies[i], export_dir=export_dir,onnx=13)
   
    def import_and_test_onnx_model(self, model_path = None ):
        
        #import onnxruntim   onnx_model_path = "chemin_vers_le_modele.onnx"
        session = onnxruntime.InferenceSession(model_path)
        
        # Récupérer les noms d'entrée et de sortie
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(input_name)
        print(output_name)   
        
        env = self.env_type(env_config = self.env_config)
        # Initialiser l'environnement       
        observation = env.reset()
        print("agent_obs : ", observation)
        
        # Boucle pour tester les actions de l'agent
        done = False
        observation = env.reset()
        env.render()
        while True :
	
            # Préparer les données d'entrée
            input_data = np.array(observation, dtype=np.float32)
            input_data = np.expand_dims(input_data, axis=0) # Ajouter une dimension batch
            
            # Effectuer l'inférence du modèle
            output = session.run([output_name],{input_name: input_data,'state_ins' : [] })
            # Déduire l'action à partir de la sortie du modèle

            action = np.argmax(output)
            print(action)

            # Exécuter l'action dans l'environnement

            observation, reward, done, info = env.step(action)

            # Afficher ou enregistrer les résultats, si nécessaire
            print("agent_obs : ",observation)
            print("agent_reward : ",reward)
            env.render()
            if done :
                env = self.env_type(env_config=self.env_config)
                agent_obs = env.reset()
                print("agent_obs : ",agent_obs)
                env.render()
 
if __name__ == '__main__':

# #FOR MULTI AGENT CHECKPOINT ARE SAVE IN /tmp/tmpxxxxxxxx I work to solve this problem
     
# Train Multi agent    
        

    env_config={ 
                "implementation" : "simple",
                "num_boxes_grid_width":6,
                "num_boxes_grid_height":3,
                "subzones_width":3,
                "num_supervisors" : 1,
                "num_operators" : 1,
                "n_orders" : 3,
                "step_limit": 100,
                "same_seed" : False
                }
    
    taille_map_x = env_config["num_boxes_grid_width"]
    taille_map_y = env_config["num_boxes_grid_height"]
    subzones_size = env_config["subzones_width"]
    nbr_op= env_config["num_operators"]
        

    nbr_of_subzones = taille_map_x/subzones_size + taille_map_y / subzones_size
    tail_obs_sup = 2 + nbr_op + nbr_op * 2
    tail_obs_op = subzones_size * subzones_size *2 + 2        



    obs_supervisor = spaces.Box(low=0, high=taille_map_x, shape=(tail_obs_sup,))
    obs_operator = spaces.Box(low=0, high=taille_map_x, shape=(tail_obs_op,))

    action_supervisor  = spaces.MultiDiscrete([4, nbr_of_subzones-1])
    action_operator  = spaces.Discrete(4)

    env_config["policies"] = {
                        "supervisor_policy": (None,obs_supervisor,action_supervisor,{}),
                        "operator_policy": (None,obs_operator,action_operator,{}),
            }
    
    train_config = {
                
                "checkpoint_freqency" : 5,
                "stop_step" : 10000,
                "num_learner_workers" :2,
                "num_cpus_per_worker": 2,
                "storage_path" : "/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/Multi_Agents_Supervisor_Operators/models"
                }
    
    my_train = DrlExperimentsTune(env=MultiAgentsSupervisorOperatorsEnv,env_config=env_config)
    #my_train.tune_train_multi_agent(train_config=train_config)
    #my_train.tune_train_from_checkpoint(train_config=train_config,multi_agent=True, checkpointpath="/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/Multi_Agents_Supervisor_Operators/models/6x3_3_100/PPO_MultiAgentsSupervisorOperatorsEnv_d607d_00000_0_2024-03-28_10-30-37/checkpoint_000000")
    #my_train.test(implementation="simple",multi_agent=True,checkpointpath="/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/Multi_Agents_Supervisor_Operators/models/6x3_3_100/PPO_MultiAgentsSupervisorOperatorsEnv_d607d_00000_0_2024-03-28_10-30-37/checkpoint_000000")
    my_train.export_to_onnx(policies = ["supervisor_policy","operator_policy"],
                            checkpointpath="/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/Multi_Agents_Supervisor_Operators/models/6x3_3_100/PPO_MultiAgentsSupervisorOperatorsEnv_d607d_00000_0_2024-03-28_10-30-37/checkpoint_000000",
                            export_dir="/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/Multi_Agents_Supervisor_Operators/models/")

# #-----------------------------------------------------------------------------------------------------
# # Train mono agent 
    # from Scenarios.UUV_Mono_Agent_TSP.env import UUVMonoAgentTSPEnv




    # env_config={
    #             "implementation":"simple",
    #             "num_boxes_grid_width":3,
    #             "num_boxes_grid_height":3,
    #             "subzones_width" : 3,
    #             "n_orders" : 3,
    #             "step_limit": 100,
    #             "same_seed" : False
    #             }

    # train_config = {
                   
    #                 "storage_path" : "/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/UUV_Mono_Agent_TSP/models",
    #                 "checkpoint_freqency" : 5,
    #                 "stop_step" : 1000000000000000000000000000000000000000,
    #                 "num_learner_workers" : 2,
    #                 "num_cpus_per_worker": 2,
                   
    # }

    # my_platform = DrlExperimentsTune(env_config=env_config,env = UUVMonoAgentTSPEnv)
    # my_platform.tune_train(train_config=train_config) 
    # #my_platform.tune_train_from_checkpoint(train_config=train_config,checkpointpath="/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/UUV_Mono_Agent_TSP/models/3x3_3_100/PPO_UUVMonoAgentTSPEnv_0a24c_00000_0_2024-03-28_10-17-46/checkpoint_000001")