#File for your env's specifiques fonctions.
import numpy as np


class UtilsUUVMonoAgentTspEnv():

    def __init__(self):     
        pass

    def create_goals(self,Env_self):
        if Env_self.randomized_orders : 

            self.x_goals = [0]
            self.y_goals = [0]
            self.goals_cord = [[0,0]]

            n_goals=0

            while True :

                for x_s in range(Env_self.largeur_grille):
                    for y_s in range(Env_self.hauteur_grille):

                        if [x_s,y_s] in  self.goals_cord :
                            pass

                        else : 
                            goals = np.random.choice([0,1],p=[1-Env_self.goals_prob,Env_self.goals_prob])

                            if goals == 1:  
                                self.x_goals.append(x_s)
                                self.y_goals.append(y_s)
                                self.goals_cord.append([x_s,y_s])
                                n_goals +=1
                                
                                if n_goals==Env_self.n_orders :
                                    self.goals = {'x': self.x_goals[1:],'y' :self.y_goals[1:]}    
                                    
                                    return self.goals, self.goals_cord[1:]
                                    
        else : 
            self.goals = {'x': [0,1,2],'y' :[2,0,1]}    
            self.goals_cord = [[0,2],[1,0],[2,1]]
            return self.goals, self.goals_cord
                                        
    def agent_move(self,Env_self, action) : 

        pre_x,pre_y= Env_self.agent.get_pos()

        new_x,new_y = pre_x,pre_y

        if action == 0:  # UP
            #print("UP")
            new_y = (min(Env_self.hauteur_grille-1, pre_y+1))
        elif action == 1:  # DOWN
            #print("DOWN")
            new_y = (max(0, pre_y-1 ))
        elif action == 2:  # LEFT
            #print("LEFT")
            new_x = (max(0, pre_x-1))
        elif action == 3:  # RIGHT
            #print("RIGHT")
            new_x = (min(Env_self.largeur_grille-1, pre_x+1))
        else:
            raise Exception("action: {action} is invalid")
        #print("starting_point :", [new_x,new_y],"=",self.starting_point)
        if [new_x,new_y] == Env_self.starting_point :
            #print("in")
            Env_self.agent_at_starting_point = 1 
        else :
            Env_self.agent_at_starting_point = 0
        Env_self.agent.set_pos( [new_x,new_y])

    def _get_observation(self,Env_self):
       
        obs = []

        obs.append(Env_self.agent.get_pos()[0])
        obs.append(Env_self.agent.get_pos()[1])

        for i in range(len(Env_self.goals_to_check)) :

            obs.append(Env_self.goals_to_check[i][0])
            obs.append(Env_self.goals_to_check[i][1])

        for i in range(len(Env_self.goal_checked)) :
            obs.append(0)
            obs.append(0)

        obs.append(Env_self.starting_point[0])
        obs.append(Env_self.starting_point[1])
        obs.append(Env_self.agent_at_starting_point)
        obs.append(Env_self.current_step)
        obs.append(Env_self.step_limit)
        
        return (obs)
    
    def _get_reward(self,Env_self):
     
        #print("goals: ",self.goals_to_check)
        #print("x,y : ",self.agent.get_pos())
        #print("len  :",len(self.goals_to_check))
    
        for i in range(0,len(Env_self.goals_to_check)) :
                   
            goal = Env_self.goals_to_check[i]
           
            agent_x,agent_y = Env_self.agent.get_pos()

            agent_coord = [agent_x,agent_y]

            #print(agent_coord,"==",goal)
            if agent_coord==goal :
                
                reward=10
                #print(self.goals_to_check[i])
                Env_self.goal_checked.append(Env_self.goals_to_check[i])
                del Env_self.goals_to_check[i]
                #print("remain goal :", self.goals_to_check )
                
                
                return reward

        reward = -1 
        return reward

