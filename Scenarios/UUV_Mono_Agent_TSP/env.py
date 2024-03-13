
from copy import deepcopy
from gymnasium import spaces

import pygame
import numpy as np
from pygame.locals import QUIT
import gymnasium as gym

from Scenarios.UUV_Mono_Agent_TSP.view2d import UUVMonoAgentTSPEnv2dView
from Scenarios.UUV_Mono_Agent_TSP.bridge import UUV
from Scenarios.UUV_Mono_Agent_TSP.utils_env import  UtilsUUVMonoAgentTspEnv

class UUVMonoAgentTSPEnv(gym.Env) : 

    def __init__(self,env_config) :
        
        self.pygame_init = False
        self.env_config = env_config
        self.largeur_grille = env_config["num_boxes_grid_width"] #Nombre de colonnes de la grille
        self.hauteur_grille = env_config["num_boxes_grid_height"]  #Nombre de lignes de la grille
        self.randomized_orders = not(env_config["same_seed"])
        self.step_limit = env_config["step_limit"]
        self.n_orders = env_config["n_orders"]
    
        self.map_min_x = 0
        self.map_max_x = self.largeur_grille
        self.map_min_y = 0
        self.map_max_y = self.hauteur_grille
        self.goals_prob = 0.05
        self.implementation = "simple" 
        self.starting_point = [0,0]
        self.agent = UUV(implementation=self.implementation)

  
        

       
        #agent coord,goals coord, starting_point,agent_at_starting_point, time ,  #step_limit, 
        self.observation_space = spaces.Box(low=0, high=self.step_limit, shape=(7+2*self.n_orders,))
        self.action_space = spaces.Discrete(4)
        self.utils = UtilsUUVMonoAgentTspEnv()

    def reset(self):

            self.current_step = 0 

            self.starting_point = [0,0]
            self.agent = UUV(implementation=self.implementation)
            self.agent.set_pos(self.starting_point)
            self.agent_at_starting_point = 1
            
            self.goals, self.goals_cord = self.utils.create_goals(self)
            
            self.goals_to_check = self.goals_cord
            self.goal_checked = []
            
            

            
            observation = self.utils._get_observation(self)

            self.cumul_reward = 0
            return observation
 
    def step(self, action):

        self.current_step +=1 

        done = False
        self.utils.agent_move(self,action)
        
        reward = self.utils._get_reward(self)

        self.cumul_reward +=reward

        info = {}
        if self.current_step == self.step_limit :
            reward = -1000 
            done = True
            info = {}

        if self.agent_at_starting_point == 1 and len(self.goals_to_check) == 0 :
            reward = 1000 - self.cumul_reward 
            
            done = True
            

        obs = self.utils._get_observation(self)
        return obs, reward, done, info

    def render(self):
        #print("in render")
        
        if self.pygame_init==False : 
            ##print("ini render")
        ############################################################################################   
            self.pygame_init = True
            # Initialisation de Pygame
            pygame.init()

            self.scenario2DView = UUVMonoAgentTSPEnv2dView(self.env_config["implementation"])


            new_agents = self.scenario2DView.create_inference_agent()
            

           

            # for i in range(len(self.op_ids)) : 
            #     #print("type of "+str(self.op_ids[i]),":", self.operators[i].get_info())
            # for i in range(len(self.sup_ids)) : 
            #     #print("type of "+str(self.sup_ids[i]),":", self.supervisors[i].get_info())
            # #print(" ")

            new_agents.set_pos(self.agent.get_pos())
         
            self.agent = new_agents
                  

            if self.env_config["implementation"] == "simple" :
                # Création de la fenêtre
                

    
                self.subzones_width =  self.env_config["num_boxes_grid_width"]
                centre = self.subzones_width // 2  
                self.plage_coords = [i - centre for i in range(self.subzones_width)] # plage de coordonnées utile pour dessiner les sous-zones
                
                self.num_subzones_grid_width = self.env_config["num_boxes_grid_width"] // self.subzones_width
                self.num_subzones_grid_height = self.env_config["num_boxes_grid_height"] // self.subzones_width 
                 
                self.largeur_fenetre = self.env_config["num_boxes_grid_width"] * 100
                self.hauteur_fenetre = self.env_config["num_boxes_grid_height"] * 100
            
                # Taille de la case
                self.taille_case_x = self.largeur_fenetre // self.largeur_grille
                self.taille_case_y = self.hauteur_fenetre // self.hauteur_grille
                # Initialisation de la liste de coordonnées des centres des sous-zones jaunes et vertes
                self.centres_sous_zones = []
                pas =  self.env_config["num_boxes_grid_width"]
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
            
            # Dessine les sous-zones visitées pour l'exemple
            self.scenario2DView.draw_visited_subzones(pygame, self.fenetre, orange, self.centres_sous_zones_visitees, self.plage_coords, self.subzones_width, self.taille_case_x, self.taille_case_y )

            # Dessine la grille
            self.scenario2DView.draw_grid(pygame, self.fenetre, noir, self.hauteur_fenetre, self.largeur_fenetre, self.taille_case_x, self.taille_case_y)

            # Dessine les robots
            self.scenario2DView.draw_agent(pygame, self.fenetre, bleu_fonce, self.agent, self.taille_case_x, self.taille_case_y)

            # Dessine les croix
            self.scenario2DView.draw_crosses(pygame, self.fenetre, rouge, self.croix, self.taille_case_x, self.taille_case_y)
            
            # Met à jour la fenêtre
            pygame.display.flip()

            # Limite la fréquence de rafraîchissement
            clock.tick(1)

