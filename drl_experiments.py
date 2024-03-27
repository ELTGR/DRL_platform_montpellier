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

from Scenarios.Multi_Agents_Supervisor_Operators.env import MultiAgentsSupervisorOperatorsEnv 

class DrlExperimentsTune():

    def __init__(self, env, env_config) :

        self.env_config = env_config
        self.env_type= env
        
    def tune_train(self, train_config) : 



        # Lancez TensorBoard en utilisant subprocess
           
        # Lancez TensorBoard en utilisant un nouveau terminal (Linux/Mac)
        tensorboard_command = f"x-terminal-emulator -e tensorboard --logdir="+str(train_config["path"])
        
        process_terminal_1 = subprocess.Popen(tensorboard_command, shell=True)
        time.sleep(2)
        self.env_config['implementation'] = "simple"
        
        tune_config={
                                    "env": self.env_type,
                                    "env_config":self.env_config,
                                    "num_workers": train_config["num_workers"],
                                    #"num_learner_workers" : train_config["num_learner_workers"],
                                    "num_gpus": train_config["num_gpus"],
                                    #"num_gpus_per_worker": train_config["num_gpus_per_worker"],
                                    "num_cpus_per_worker": train_config["num_cpus_per_worker"],
                                    "model":train_config["model"],
                                    "optimizer": train_config["optimizer"],
                                }
        
        ray.init()
        algo = tune.run("PPO",name=train_config["name"],config = tune_config,stop = {"timesteps_total": train_config["stop_step"]}, 
                        checkpoint_config = CheckpointConfig(checkpoint_at_end=True,checkpoint_frequency=train_config["checkpoint_freqency"] ),
                        storage_path=train_config["path"]
                        )
                                                                               
    def tune_train_from_checkpoint(self, train_config, path):
         


        

        self.env_config['implementation'] = "simple"
        ray.init()

        tune_config={
                                    "env": self.env_type,
                                    "env_config":self.env_config,
                                    "num_workers": train_config["num_workers"],
                                    #"num_learner_workers" : train_config["num_learner_workers"],
                                    "num_gpus": train_config["num_gpus"],
                                    #"num_gpus_per_worker": train_config["num_gpus_per_worker"],
                                    "num_cpus_per_worker": train_config["num_cpus_per_worker"],
                                    "model":train_config["model"],
                                    "optimizer": train_config["optimizer"],
                                }
        
        
        algo = tune.run("PPO",name=train_config["name"],config = tune_config,stop = {"timesteps_total": train_config["stop_step"]}, 
                        checkpoint_config = CheckpointConfig(checkpoint_at_end=True,checkpoint_frequency=train_config["checkpoint_freqency"] ),
                        storage_path=train_config["path"],restore=path
                        )
           
    def test(self, implementation, path) :
             # Lancez TensorBoard en utilisant un nouveau terminal (Linux/Mac)
            
            self.env_config['implementation'] = implementation 
            
            if self.env_config['implementation'] == "real": 
                ros_tcp_endpoint = f"x-terminal-emulator -e roslaunch ros_tcp_endpoint endpoint.launch"
                process_terminal_1 = subprocess.Popen(ros_tcp_endpoint, shell=True)

            print("config : ",self.env_config)

            env = self.env_type(env_config = self.env_config)
            loaded_model = Algorithm.from_checkpoint(path)
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
 
    def export(self,checkpointpath, export_dir) : 

        loaded_algo = Algorithm.from_checkpoint(checkpointpath)
        loaded_algo.export_policy_model(export_dir=export_dir,onnx=13)
   
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
    
  
class DrlExperimentsPPO():
    def __init__(self,env,env_config,) :

        self.env_config = env_config
        self.env_type= env

    def ppo_train(self,train_config):
         
           
            ray.init()

            def select_policy(algorithm, framework):
                if algorithm == "PPO":
                    if framework == "torch":
                        return PPOTorchPolicy
                    elif framework == "tf":
                        return PPOTF1Policy
                    else:
                        return PPOTF2Policy
                else:
                    raise ValueError("Unknown algorithm: ", algorithm)

            taille_map_x = train_config["taille_map_x"]
            taille_map_y = train_config["taille_map_y"]
            subzones_size = train_config["subzones_size"]
            nbr_sup = train_config["nbr_sup"]
            nbr_op = train_config["nbr_op"]

            nbr_of_subzones = taille_map_x/subzones_size + taille_map_y / subzones_size
            tail_obs_sup = 2 + nbr_op + nbr_op * 2
            tail_obs_op = subzones_size * subzones_size *2 + 2 
            print("trail_obs_sup",tail_obs_sup)

            ppo_config = (
                PPOConfig()
                # or "corridor" if registered above
                .environment(self.env_type,
                            env_config=self.env_config
                            )
                .environment(disable_env_checking=True)

                .framework("torch")

                # disable filters, otherwise we would need to synchronize those
                # as well to the DQN agent
                
                .rollouts(observation_filter="MeanStdFilter")
                .training(
                    model={"vf_share_layers": True},
                    vf_loss_coeff=0.01,
                    num_sgd_iter=6,
                    _enable_learner_api=False)
                
                # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                .resources(num_gpus=0)
                #.rollouts(num_rollout_workers=1)
                .rl_module(_enable_rl_module_api=False)
                
            )
            
            #Creation of observation and action space 
            obs_supervisor = spaces.Box(low=0, high=taille_map_x, shape=(tail_obs_sup,))
            obs_operator = spaces.Box(low=0, high=taille_map_x, shape=(tail_obs_op,))

            action_supervisor  = spaces.MultiDiscrete([4, nbr_of_subzones-1])
            action_operator  = spaces.Discrete(4)


            policies = {
                "supervisor_policy": (None,obs_supervisor,action_supervisor, {}),
                "operator_policy": (None,obs_operator,action_operator, {}),
                
            }

            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                #print("#################",agent_id,"#####################################")
                agent_type = agent_id.split('_')[0]
                if agent_type == "supervisor" :
                    #print(agent_id,"supervisor_policy")
                    return "supervisor_policy"

                else :
                    #print(agent_id,"operator_policy")
                    return "operator_policy"
    
            ppo_config.multi_agent(policies=policies,policy_mapping_fn=policy_mapping_fn, )
            ppo = ppo_config.build()

            i=0 
            j=0
            checkpoint_interval =  train_config["checkpoint_interval"]
            end_interation = train_config["Iteration stop"]
            while i <= end_interation :
                i+=1
                j+=1
                result_ppo = ppo.train()
                print("== Iteration", i, "==")
                
                if j == checkpoint_interval :
                    j=0
                    save_result = ppo.save()  
                    print(pretty_print(result_ppo))
                    
                    path_to_checkpoint = save_result
                    print(
                        "An Algorithm checkpoint has been created inside directory: "
                        f"'{path_to_checkpoint}'."
                    )    
 

    def ppo_train_from_checkpoint(self):


            #================LOAD================

            # Use the Algorithm's `from_checkpoint` utility to get a new algo instance
            # that has the exact same state as the old one, from which the checkpoint was
            # created in the first place:
            my_new_ppo =  Algorithm.from_checkpoint(str(self.last_ppo_checkpoint))
            i=0 
            j=0
            fin = 20
            save_intervalle = 5

            my_new_result_ppo = my_new_ppo
            while i  <= fin :

                i+=1
                j+=1
               
                my_new_result_ppo.train()
                
                if j == save_intervalle :
                    j=0 

                    save_result = my_new_ppo.save()
                    print(pretty_print(my_new_result_ppo))
                    self.last_ppo_checkpoint = save_result
                    
            my_new_result_ppo.stop

    def test(self) :
      
            def inference_policy_mapping_fn(agent_id):
                agent_type = agent_id.split('_')[0]
                if agent_type == "supervisor" :

                    return "supervisor_policy"

                else :

                    return "operator_policy"
          

           
            
            algo = Algorithm.from_checkpoint("/tmp/tmp7mm65dei")
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

if __name__ == '__main__':

#FOR MULTI AGENT CHECKPOINT ARE SAVE IN /tmp/tmpxxxxxxxx I work to solve this problem
     
# #Train Multi agent    
#     taille_map_x = 6
#     taille_map_y = 3
#     subzones_size = 3
#     nbr_sup = 1
#     nbr_op= 1 
#     nbr_goals = 3 
#     env_config={ 
#                 "implementation" : "simple",
#                 "num_boxes_grid_width":taille_map_x,
#                 "num_boxes_grid_height":taille_map_y,
#                 "subzones_width":subzones_size,
#                 "num_supervisors" : nbr_sup,
#                 "num_operators" : nbr_op,
#                 "n_goals": nbr_goals,
#                 "num_directions" : 4,
#                 "step_limit": 100,
#                 "same_seed" : False
#                 }
    
#     train_config = {
#                 "checkpoint_interval" : 2,
#                 "Iteration stop": 3,
#                 "taille_map_x" : 6,
#                 "taille_map_y" : 3,
#                 "subzones_size" : 3,
#                 "nbr_sup" : 1,
#                 "nbr_op" : 1
#                 }
#     my_train = DrlExperimentsPPO(env=MultiAgentsSupervisorOperatorsEnv,env_config=env_config).ppo_train(train_config= train_config)

##-----------------------------------------------------------------------------------------------------
#Train mono agent 
    from Scenarios.UUV_Mono_Agent_TSP.env import UUVMonoAgentTSPEnv


    taille_map_x = 3
    taille_map_y = 3
    n_orders = 3
    step_limit = 100


    env_config={
                "implementation":"simple",
                
                "num_boxes_grid_width":taille_map_x,
                "num_boxes_grid_height":taille_map_y,
                "n_orders" : n_orders,
                "step_limit": step_limit,
                "same_seed" : False
                }

    train_config = {
                    "name" : str(taille_map_x)+"x"+str(taille_map_y)+"_"+str(n_orders)+"_"+str(step_limit),
                    "path" : "/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/UUV_Mono_Agent_TSP/models",
                    "checkpoint_freqency" : 5,
                    "stop_step" : 1000000000000000000000000000000000000000,
                    "num_workers": 1,
                    "num_learner_workers" : 0,
                    "num_gpus": 0,
                    "num_gpus_per_worker": 0,
                    "num_cpus_per_worker": 5,
                    "model": {"fcnet_hiddens": [128, 128, 128]},  # Architecture du réseau de neurones (couches cachées) 
                    "optimizer": {"learning_rate": 0.001,} # Taux d'apprentissage
    }

    my_platform = DrlExperimentsTune(env_config=env_config,env = UUVMonoAgentTSPEnv)
    #my_platform.tune_train(train_config=train_config) 
    my_platform.export(checkpointpath="/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/UUV_Mono_Agent_TSP/models/3x3_3_100/PPO_UUVMonoAgentTSPEnv_ad9f3_00000_0_2024-03-26_10-10-23/checkpoint_000010",  export_dir="/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/UUV_Mono_Agent_TSP/models/3x3_3_100/")
    #my_platform.tune_train_from_checkpoint(train_config=train_config,path="/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/UUV_Mono_Agent_TSP/models/4x4_3_100/PPO_UUVMonoAgentTSPEnv_4d474_00000_0_2024-03-26_13-42-27/checkpoint_000000")

    #my_platform.import_and_test_onnx_model(model_path ="/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/model.onnx")
    #my_platform.export_checkpoint(checkpoint_path = "/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/UUV_Mono_Agent_TSP/models/4x4_3_100/PPO_UUVMonoAgentTSPEnv_b166e_00000_0_2024-03-26_13-45-15/checkpoint_000002", export_dir = "/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/UUV_Mono_Agent_TSP/models/4x4_3_100/")
    #my_platform.test(implementation="simple",path="/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/UUV_Mono_Agent_TSP/models/4x4_3_100/PPO_UUVMonoAgentTSPEnv_b166e_00000_0_2024-03-26_13-45-15/checkpoint_000002")

    #loaded_algo = Algorithm.from_checkpoint("/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/UUV_Mono_Agent_TSP/models/4x4_3_100/PPO_UUVMonoAgentTSPEnv_b166e_00000_0_2024-03-26_13-45-15/checkpoint_000002")
    #loaded_algo.export_policy_model(export_dir="/home/ia/Desktop/DRL_platform/DRL_platform_montpellier/Scenarios/UUV_Mono_Agent_TSP/models/4x4_3_100/",onnx=13)
