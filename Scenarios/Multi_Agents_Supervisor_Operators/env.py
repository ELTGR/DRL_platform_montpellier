
from copy import deepcopy
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import random
import pygame
import numpy as np
from Scenarios.Multi_Agents_Supervisor_Operators.view2d import UXVSupervisorOperators2DView
from Scenarios.Multi_Agents_Supervisor_Operators.bridge import UXV
from pygame.locals import QUIT
import gymnasium as gym



#from utils_env import  TSP_Utils
from Scenarios.Multi_Agents_Supervisor_Operators.utils_env import  UtimsMultiAgentsSupervisorOperatorsEnv


class MultiAgentsSupervisorOperatorsEnv(MultiAgentEnv):


    def __init__(self, env_config):
        self.utils = UtimsMultiAgentsSupervisorOperatorsEnv()

        # Paramètres de l'environnement
        

        # Dimensions de la fenêtre pygame
        self.env_config = env_config
        self.implementation = "simple"
        self.subzones_width = env_config["subzones_width"]          # Taille des subzones
        self.largeur_grille = env_config["num_boxes_grid_width"] #Nombre de colonnes de la grille
        self.hauteur_grille = env_config["num_boxes_grid_height"]  #Nombre de lignes de la grille
        self.same_seed = env_config["same_seed"]
        self.num_targets = 1
        self.subzones_checked = []

        self.n_sup = env_config["num_supervisors"]  # Nombre de superviseurs
        self.n_op = env_config["num_operators"]     # Nombre d'operateurs

        self.goals_prob =  0.2 #env_config["goals_probability"]   # Probabilité d'apparition d'une mine sur chaque case

        #self.n_dir = env_config['n_dir']            # Nombre de directions des agents

        self.step_limit = env_config["step_limit"]  # Nombre d'itérations max par step
        self.nx = int(self.largeur_grille/self.subzones_width)            # Nombre de subzones en X
        self.ny = int(self.hauteur_grille/self.subzones_width)            # Nombre de subzones en Y
        self.ns = int(self.nx*self.ny)          # Nombre total de subzones
        self.n_agents = self.n_op + self.n_sup
            # Création des agents :

        self.sup_ids = self.utils.create_supervisors_id(self.n_sup)    # Superviseurs ids
        self.op_ids = self.utils.create_operators_id(self.n_op)        # Opérateurs ids
        self.agent_ids =  self.op_ids +self.sup_ids 

        self.subzones = self.utils.subzones(self.largeur_grille,self.hauteur_grille,self.subzones_width)

        self.sup_agents = {self.sup_ids[i]:UXV(self.implementation) for i in range(self.n_sup)}  # Création des superviseurs
        self.utils.reset_sup_pos(self)        # Initialisation de leur position en (0,0)
        self.op_agents = {self.op_ids[i]:UXV(self.implementation) for i in range(self.n_op) }        # Création des opérateurs
        self.utils.reset_op_pos(self)         # Initialisation de leur position en (0,0)
        self.supervisors = []
        self.operators = []
      
        for agent_id, agent in self.sup_agents.items() :
            self.supervisors.append(agent)
        for agent_id, agent in self.op_agents.items() :
            self.operators.append(agent)
        
        # Autres paramètres : Subzones

        
        #print("self.subzones",self.subzones) 
        #print("len(self.subzones)",len(self.subzones))         # dictionnaire des souszone (numéro:position)

        self.nbr_of_subzones = len(self.subzones)
    
        self.subzones_center = {i:None for i in range(self.ns)} # dictionnaire des centre des souszone (numéro:centre)
        
        self.centers_list_x = []    # Liste du centre des souszones X
        self.centers_list_y = []    # Liste du centre des souszones Y
        self.utils.init_subzones_center(self) # Initialise le centre des souszones

        # Autres paramètres : Goals
        self.x_goals = []   # Position X des goals : list
        self.y_goals = []   # Position Y des goals : list
        self.subzones_goals = {i:None for i in range(self.ns)}  # Position des goals de chaque sous-zone 

       
        self.utils.goals_generation(self)     # Génération des goals

        self.utils.reset_sup_goals(self)
        self.utils.reset_op_goals(self)


        self.nbr_actialisation =0
        self.check_sup_goal = self.sup_goals
        self.check_op_goal = self.op_goals

        #print("\ninit sup goal", self.check_sup_goal)
        #print("init op  goal", self.check_op_goal)
        #print("subzone goals",self.subzones_goals)

        self.operator_end = {self.op_ids[i]:0 for i in range(self.n_op)}    # Dictionnaire d'état des opérateurs (0/1) si fini ou non fini

        self.len_of_sup_space = 2 + self.n_op + self.n_op * 2
        self.len_of_op_space = 2 + self.subzones_width * self.subzones_width * 2 

        self.pygame_init = False
        super().__init__()

    def reset(self):
        #print("**********reset***********")
        self.step_counter=0
        self.zone_counter=0
        self.change_goal=False
        self.zone_center_counter=0
        self.supervisor_check = False
        self.subzones_checked = []

        if not(self.same_seed) : 

            self.utils.goals_generation(self)

        self.utils.reset_sup_goals(self)
        self.utils.reset_op_goals(self)

        self.check_op_goal =  self.op_goals
        
        #print("\reset sup goal", self.check_sup_goal)
        #print("reset op  goal", self.check_op_goal)
        #print("subzone goals",self.subzones_goals)

        self.operator_end = {self.op_ids[i]:0 for i in range(self.n_op)}

        observations ={self.agent_ids[i] :self.utils._get_observation(self,self.agent_ids[i]) for i in range(0,self.n_agents) }
        ##print(observations)
        #print(" ")
        return observations

    def step(self, action_dict):
        #print(" ")
        self.step_counter += 1
        self.operator_end_task = 0


        observations ={self.agent_ids[i] :None for i in range(0,self.n_agents) }
       

        rewards = {self.agent_ids[i] :0 for i in range(0,self.n_agents) }
     

        terminated = {self.agent_ids[i] :False for i in range(0,self.n_agents) }
       

        terminated.update({"__all__" : False })
      
       

        for agent_id, action in action_dict.items() :
            agent_type = agent_id.split('_')[0]
            #print(" ")
            #print("agent : ",agent_id)
            if agent_type == "operator":
                #=================Move========================#

                pre_x,pre_y = self.op_agents[agent_id].get_pos()

                self.utils.op_move(self,agent_id,pre_x,pre_y,action)

                now_x,now_y = self.op_agents[agent_id].get_pos()

                if self.operator_end[agent_id] == 0 : 

                        terminated[agent_id]=False
                        rewards[agent_id]=0

                #=================Check goals========================#
               
                rewards= self.utils.update_op_reward(self,agent_id,now_x,now_y,rewards)
                rewards, terminated = self.utils.try_step_limit(self,agent_id,rewards,terminated)
               

                #check if the operator has seen all the objectives
                if self.operator_end[agent_id] == 1 :
                    self.operator_end_task+=1
                
                #check if all the operator has seen all the objectives
                if self.operator_end_task == self.n_op :
                    #print("CHECK START")
                    self.supervisor_check = True

              
                observations[agent_id]=self.utils._get_observation(self,agent_id)

            elif agent_type == "supervisor":

                            ############ PAS COMPRIS CETTE PARTIE #############

                            #desired pos of the sup
                            action_move = action[0]
                            next_subzone = action[1:]

                            #=================Move========================#
                            pre_x,pre_y = self.sup_agents[agent_id].get_pos()


                            self.utils.sup_move(self,agent_id,pre_x,pre_y,action_move)

                          
                            now_x,now_y = self.sup_agents[agent_id].get_pos()


                            #=================Check goals========================#
                            rewards = self.utils.update_sup_reward(self,agent_id,now_x,now_y,rewards)
                            rewards, terminated = self.utils.try_step_limit(self,agent_id, rewards,terminated)
                            observations[agent_id]=self.utils._get_observation(self,agent_id)
            #change_goal is True when all opérator seen their goals 
            #and superivoses went on the center of the operator's subzone
        if self.change_goal :
                        self.change_goal = False
                        self.supervisor_check = False
                        self.nbr_of_subzones -= self.n_op
                        #print("self.nbr_of_subzones",self.nbr_of_subzones)
                        if self.nbr_of_subzones > 0 : 
                        

                            #Give next subzone goal's to the opérator
                            
                            self.utils.new_subzone_for_operator(self,next_subzone)
                            self.utils.new_subzone_for_supervisor(self,next_subzone)
                            self.operator_end = {self.op_ids[i]:0 for i in range(self.n_op)}
                            
                            rewards[self.sup_ids[0]],terminated["__all__"] = self.utils.new_subzone_reward(self,next_subzone)
                            
                            for i in range(self.n_agents):
                                
                                observations[self.agent_ids[i]]=self.utils._get_observation(self,self.agent_ids[i])
                            
                        else :
                            for i in range(self.n_agents):
                                rewards[self.agent_ids[i]]=1000 
                            terminated["__all__"]=True

        ##print("observations",observations)
        ##print("rewards",rewards)
        ##print("terminated",terminated)

        return observations, rewards, terminated, {}

    def render(self):
        #print("in render")
        
        if self.pygame_init==False : 
            ##print("ini render")
        ############################################################################################   
            self.pygame_init = True
            # Initialisation de Pygame
            pygame.init()

            self.scenario2DView = UXVSupervisorOperators2DView(self.env_config["implementation"],self.op_ids,self.sup_ids)


            new_op_agents = self.scenario2DView.create_inference_op()
            new_sup_agents = self.scenario2DView.create_inference_sup()

           

            # for i in range(len(self.op_ids)) : 
            #     #print("type of "+str(self.op_ids[i]),":", self.operators[i].get_info())
            # for i in range(len(self.sup_ids)) : 
            #     #print("type of "+str(self.sup_ids[i]),":", self.supervisors[i].get_info())
            # #print(" ")
            self.operators = []
            self.supervisors = []

         

                  
            for id_agent,new_op in new_op_agents.items() :
                    
                new_op.set_pos(self.op_agents[id_agent].get_pos())
                self.op_agents[id_agent] = new_op
                self.operators.append(new_op )
          


            for id_agent,new_sup in new_sup_agents.items() :
                    
                new_sup.set_pos(self.sup_agents[id_agent].get_pos())
                self.sup_agents[id_agent] = new_sup
                self.supervisors.append(new_sup)
                
            #print("=========================remplacment=================================")
          
            # for i in range(len(self.op_ids)) : 
            #     #print("type of "+str(self.op_ids[i]),":", self.operators[i].get_info())
            # for i in range(len(self.sup_ids)) : 
            #     #print("type of "+str(self.sup_ids[i]),":", self.supervisors[i].get_info())
            #     #print(" ")

            if self.env_config["implementation"] == "simple" :
                # Création de la fenêtre
                

    
                self.subzones_width = self.env_config["subzones_width"]
                centre = self.subzones_width // 2  
                self.plage_coords = [i - centre for i in range(self.subzones_width)] # plage de coordonnées utile pour dessiner les sous-zones
                
                self.num_subzones_grid_width = self.env_config["num_boxes_grid_width"] // self.subzones_width
                self.num_subzones_grid_height = self.env_config["num_boxes_grid_height"] // self.subzones_width 
                self.num_subzones = self.num_subzones_grid_width * self.num_subzones_grid_height 
                self.largeur_fenetre = self.env_config["num_boxes_grid_width"] * 40
                self.hauteur_fenetre = self.env_config["num_boxes_grid_height"] * 40
            
                # Taille de la case
                self.taille_case_x = self.largeur_fenetre // self.largeur_grille
                self.taille_case_y = self.hauteur_fenetre // self.hauteur_grille
                # Initialisation de la liste de coordonnées des centres des sous-zones jaunes et vertes
                self.centres_sous_zones = []
                pas = self.env_config["subzones_width"]
                # Boucles pour générer les coordonnées
                for x in range(1, self.largeur_grille, pas):
                    for y in range(1, self.hauteur_grille, pas):
                        self.centres_sous_zones.append((x, y))
                    
                # Liste de croix représentant la case où se trouvent les cibles
                        
                self.croix = []
                # Générer 60 croix aléatoirement
                
                for i in range(len(self.goals['x'])):
                    self.croix.append((self.goals['x'][i],self.goals['y'][i]))
                   
                # Initialisation de la liste de coordonnées des sous-zones visitées pour l'exemple        
                self.centres_sous_zones_visitees = [] 
                self.centres_sous_zones_visitees = self.centres_sous_zones[0:2]
                
                self.num_operators = self.env_config["num_operators"]

                self.fenetre = pygame.display.set_mode((self.largeur_fenetre, self.hauteur_fenetre))
                pygame.display.set_caption("Multi-Agent Supervisor Workers Environment")

        if self.env_config["implementation"] == "simple" :        
            # Couleurs
            blanc = (255, 255, 255)
            noir = (0, 0, 0)
            bleu_clair = (173, 216, 230)
            bleu_fonce = (0, 0, 128)
            rouge = (255, 0, 0)
            jaune = (255, 255, 0)
            vert = (0, 255, 0)
            orange = (255, 128, 0)
            
            clock = pygame.time.Clock()

            #while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                        pygame.quit()
                        
            # Efface la fenêtre
            self.fenetre.fill(blanc)
            ##print("sup ", self.supervisors)
            ##print("ops ", self.operators)
            # Dessine les sous-zones en damier
            self.scenario2DView.draw_subzones(pygame, self.fenetre, vert, jaune, self.num_subzones_grid_height, self.centres_sous_zones, self.plage_coords, self.subzones_width, self.taille_case_x, self.taille_case_y )

            # Dessine les sous-zones visitées pour l'exemple
            self.scenario2DView.draw_visited_subzones(pygame, self.fenetre, orange, self.centres_sous_zones_visitees, self.plage_coords, self.subzones_width, self.taille_case_x, self.taille_case_y )

            # Dessine la grille
            self.scenario2DView.draw_grid(pygame, self.fenetre, noir, self.hauteur_fenetre, self.largeur_fenetre, self.taille_case_x, self.taille_case_y)

            # Dessine les robots
            self.scenario2DView.draw_supervisor(pygame, self.fenetre, bleu_clair, self.supervisors, self.taille_case_x, self.taille_case_y )
            self.scenario2DView.draw_operators(pygame, self.fenetre, bleu_fonce, self.operators, self.taille_case_x, self.taille_case_y)

            # Dessine les croix
            self.scenario2DView.draw_crosses(pygame, self.fenetre, rouge, self.croix, self.taille_case_x, self.taille_case_y)
            
            # Met à jour la fenêtre
            pygame.display.flip()

            # Limite la fréquence de rafraîchissement
            clock.tick(1)




