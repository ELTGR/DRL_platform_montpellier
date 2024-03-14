#File for your env's specifiques fonctions.
import numpy as np


class UtilsMonoAgent():


    def __init__(self):     
        pass

    def create_goals(self,Env_self):
        if Env_self.randomized_orders : 

            self.x_goals = [0]
            self.y_goals = [0]
            self.goals_cord = []

            n_goals=0

            while True : 
                
                for x_s in range(Env_self.largeur_grille):
                    for y_s in range(Env_self.hauteur_grille):

                        for i in range(len(self.x_goals)) :


                            print(x_s ,'== ',self.x_goals[i],"and",self.x_goals[i],'==',self.y_goals[i])

                            if x_s == self.x_goals[i] and y_s == self.y_goals[i]:
                                print("pass")

                            else : 
                                print("add")

                                goals = np.random.choice([0,1],p=[1-Env_self.goals_prob,Env_self.goals_prob])
                                if goals == 1:
                                    
                                    self.x_goals.append(x_s)
                                    self.y_goals.append(y_s)
                                    self.goals_cord.append([x_s,y_s])

                                    n_goals +=1
                                    
                                    if n_goals==Env_self.n_orders :
                                        self.goals = {'x': self.x_goals[1:],'y' :self.y_goals[1:]}    
                                        return self.goals_cord,self.goals
                                    
        else : 
            self.goals = {'x': [0,1,2],'y' :[2,0,1]}    
            self.goals_cord = [[0,2],[1,0],[2,1]]
            return self.goals,self.goals_cord
                                        
    def agent_move(self,Env_self, action) : 

        pre_x,pre_y = Env_self.agent.get_pos()

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

class UtimsMultiAgentsSupervisorOperatorsEnv() :
 
    def __init__(self):
      pass

    def create_supervisors_id(self,n_sup):
        id_list = []
        for i in range(n_sup):
            id_list.append("supervisor_{}".format(i))
        return id_list

    def create_operators_id(self,n_op):
        id_list = []
        for i in range(n_op):
            id_list.append("operator_{}".format(i))
        return id_list


    def subzones(self,x,y,s):
        n_s = (x/s)*(y/s)   # Nombre de subzones
        centers = {i:self.find_subzone(x,y,s,i) for i in range(int(n_s))}
        return centers

    def find_subzone(self,x,y,s,i):
        n_x = x/s

        x_c = s* int(i%n_x)

        y_c = s* int(i/n_x)
        return [x_c,y_c]


    def reset_sup_pos(self,Env_self):
        """
        set la position de chaque superviseur en (0,0)
        Voir pour ajouter une position aléatoire / définie à l'avance
        """
        for id in Env_self.sup_ids:
            Env_self.sup_agents[id].set_pos([0,0])
       
    def reset_op_pos(self,Env_self):
        """
        set la position de chaque operateur en (0,0)
        Voir pour ajouter une position aléatoire / définie à l'avance
        """
        for i in range(Env_self.n_op) :
            
            id = Env_self.op_ids[i]
            position = Env_self.subzones[i]
            #print("pos", position)
            Env_self.op_agents[id].set_pos(position)


    def init_subzones_center(self,Env_self):
        for i in range(0, len(Env_self.subzones)) :
            half_s = (Env_self.subzones_width-1)/2    # demi-longueur d'une sous-zone
            center = (half_s+Env_self.subzones[i][0],half_s+Env_self.subzones[i][1])
            Env_self.subzones_center[i]=(center[0],center[1])
            Env_self.centers_list_x.append(center[0])
            Env_self.centers_list_y.append(center[1])

    def goals_generation(self,Env_self):
        self.x_goals = []
        self.y_goals = []
        
        for i in range(0, len(Env_self.subzones)) :
            goals_subzone = []

            for x_s in range(Env_self.subzones_width):
                for y_s in range(Env_self.subzones_width):
                    goals = np.random.choice([0,1],p=[1-Env_self.goals_prob,Env_self.goals_prob])
                    if goals == 1:
                        #goals_x.append(self.subzones[i][0] + x_s)
                        #goals_y.append(self.subzones[i][1] + y_s)
                        goals_subzone.append([Env_self.subzones[i][0] + x_s,Env_self.subzones[i][1] + y_s])
                        self.x_goals.append(Env_self.subzones[i][0] + x_s)
                        self.y_goals.append(Env_self.subzones[i][1] + y_s)
        
            Env_self.subzones_goals[i]=goals_subzone 
        Env_self.goals = {'x': self.x_goals,'y' :self.y_goals}


    def reset_sup_goals(self,Env_self):

        for i in range(int(Env_self.n_sup)) : 
            centers = []
            for j in range(0,int(Env_self.n_op)) :
                centers.append([Env_self.centers_list_x[j], Env_self.centers_list_y[j]])
            Env_self.sup_goals = { Env_self.sup_ids[i] : centers}

    def reset_op_goals(self,Env_self):
        Env_self.op_goals = {Env_self.op_ids[i]:Env_self.subzones_goals[i] for i in range(Env_self.n_op)}
#========obs==============#  
    def _get_observation(self,Env_self,agent_id):

        agent_type = agent_id.split('_')[0]
        
        if agent_type == "supervisor" :

            observation =   [Env_self.sup_agents[agent_id].get_pos()[0], Env_self.sup_agents[agent_id].get_pos()[1]]  # Position du superviseur
            #print(observation)

            for i in range(Env_self.n_op):

                observation.append(Env_self.operator_end[Env_self.op_ids[i]])   # Etat des opérateurs
            #print(observation)
           
            for i in range(len(Env_self.sup_goals[agent_id])) : 
                
                observation.extend(Env_self.sup_goals[agent_id][i])
            
            #print(observation)
            if len(observation)< Env_self.len_of_sup_space :

                
                for i in range(len(observation),Env_self.len_of_sup_space ) :
                    observation.append(0.0)

        elif agent_type == "operator" :


            observation =   [Env_self.op_agents[agent_id].get_pos()[0],
                             Env_self.op_agents[agent_id].get_pos()[1],]
            ##print(observation)
            
         
            for i in range(len(Env_self.op_goals[agent_id])) : 
                ##print("agent_id",agent_id)
               
                ##print("observationextend",self.op_goals[agent_id][i])
                observation.extend(Env_self.op_goals[agent_id][i])
            
       
            if len(observation) < Env_self.len_of_op_space :
                
                
                for i in range(len(observation),Env_self.len_of_op_space ) :
                    observation.append(0.0) 
        else:
            raise("Agent type error")
        #print("id :",agent_id)
        #print("observation :",observation)
        #print("observation len :",len(observation))
        return observation

    def op_move(self,Env_self,agent_id,pre_x,pre_y,action):
        #print("x/y before",pre_x,pre_y)
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
        elif action == 4 :
            #print("STAY")
            pass
        else:
            raise Exception("action: {action} is invalid")
        Env_self.op_agents[agent_id].set_pos([new_x,new_y])

    def sup_move(self,Env_self,agent_id,pre_x,pre_y,action):
        #print("x/y before",pre_x,pre_y)
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
        elif action == 4 :
            #print("STAY")
            pass
        else:
            raise Exception("action: {action} is invalid")
        Env_self.sup_agents[agent_id].set_pos([new_x,new_y])

    def update_op_reward(self,Env_self,agent_id,now_x,now_y,rewards):
     
        #print("goals",self.check_op_goal[agent_id])
        #print("x/y",now_x,now_y)
        #print("len : ",len(self.check_op_goal[agent_id]))
        if len(Env_self.check_op_goal[agent_id]) == 0 :
            Env_self.operator_end[agent_id]= 1 
            #print("opérator end")
            rewards[agent_id] = 0
            return rewards
        for i in range(0,len(Env_self.check_op_goal[agent_id])) :
                   
            
            goal_x = Env_self.check_op_goal[agent_id][i][0]
            goal_y = Env_self.check_op_goal[agent_id][i][1]
            
           
            #print((now_x,now_y),'=',(goal_x,goal_y))
            if (now_x,now_y)==(goal_x,goal_y) :
                #print(goal_x,goal_y)
                #print("on goal get 10")
                rewards[agent_id]+=100
                #print(self.check_op_goal[agent_id][i])
                del Env_self.check_op_goal[agent_id][i]
                #print("remain goal :", self.check_op_goal )
                goal_uncheck = len(Env_self.check_op_goal[agent_id])
                

                if goal_uncheck == 0 :
                    #print("check all goal get 100")
                    Env_self.operator_end[agent_id]=1
                    rewards[agent_id]+=200
                break
       
        return rewards

    def update_sup_reward(self,Env_self,agent_id,now_x,now_y,rewards):
      
        #print("goals",self.check_sup_goal[agent_id])
        #print("x/y",now_x,now_y)
        for i in range(0,len(Env_self.check_sup_goal[agent_id])) :

            goal_x = Env_self.check_sup_goal[agent_id][i][0]
            goal_y = Env_self.check_sup_goal[agent_id][i][1]

            if (now_x,now_y)==(goal_x,goal_y) and Env_self.supervisor_check :
                #print(goal_x,goal_y)
                #print("on goal get 10")
                rewards[agent_id]=100
                del Env_self.check_sup_goal[agent_id][i]
                #print("del self.check_sup_goal[agent_id][i]", self.check_sup_goal )
                goal_uncheck = len(Env_self.check_sup_goal[agent_id])
                Env_self.subzones_checked.append([goal_x,goal_y])

                if goal_uncheck == 0 :
                    Env_self.change_goal = True
                    rewards[agent_id]=200
                break
        
        return rewards
     
    def new_subzone_reward(self,Env_self, next_subzone) : 
        #print("inside new subzone reward")
        reward = 100*Env_self.n_op
        end = False
        #print("new_zone_coord",next_subzone)
        #print("old_zone_coord",self.subzones_checked)
        for new_zone in next_subzone :
            for old_center_coord in Env_self.subzones_checked : 
                centers=[Env_self.centers_list_x[new_zone], Env_self.centers_list_y[new_zone]]
                if centers == old_center_coord :
                    #print("false choise of new zone ")
                    reward += -250
                    end = True 
                
        return reward,end
        
    def try_step_limit(self,Env_self,agent_id,rewards,terminated):
      
        if Env_self.step_counter >= Env_self.step_limit:
            rewards[agent_id]=-100
            terminated["__all__"]=True
        return rewards,terminated
 
    def new_subzone_for_operator(self,Env_self,next_subzone):
        
        Env_self.op_goals = {Env_self.op_ids[i]:Env_self.subzones_goals[next_subzone[i]] for i in range(Env_self.n_op)}  
        Env_self.check_op_goal = Env_self.op_goals
 
    def new_subzone_for_supervisor(self,Env_self,next_subzone):
        
       
        #print("self.centers_list_x",self.centers_list_x)
        #print("self.centers_list_y",self.centers_list_y)


        for i in range(int(Env_self.n_sup)) : 
            centers = []
            for j in next_subzone :
                
                centers.append([Env_self.centers_list_x[j], Env_self.centers_list_y[j]])
                

            Env_self.sup_goals = { Env_self.sup_ids[i] : centers}
       

        Env_self.check_sup_goal = Env_self.sup_goals


