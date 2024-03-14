from Scenarios.Multi_Agents_Supervisor_Operators.bridge import UXV



class UXVMonoAgent2DView:

    def __init__(self,implementation):
        # Initialisation de la classe MobileSupervisor
            self.implementation = implementation
            print("self.implementation : ",self.implementation)
            
            self.i = 0
            
    def create_inference_agent(self):
        i=0
        agent = UXV(self.implementation,i) 
        
        return agent
   

    # Fonction pour dessiner les sous-zones visités
    def draw_visited_subzones(self, pygame, fenetre, orange, centres_sous_zones_visitees, plage_coords, subzones_width, taille_case_x, taille_case_y ):
        
        for centre in centres_sous_zones_visitees:
            for dx in range(plage_coords[0], plage_coords[subzones_width -1] + 1):
                for dy in range(plage_coords[0], plage_coords[subzones_width -1] + 1):
                    x = centre[0] + dx
                    y = centre[1] + dy
                    pygame.draw.rect(fenetre, orange, (x * taille_case_x, y * taille_case_y, taille_case_x, taille_case_y))
                 
    # Fonction pour dessiner la grille
    def draw_grid(self, pygame, fenetre, noir, hauteur_fenetre, largeur_fenetre, taille_case_x, taille_case_y):
        for x in range(0, largeur_fenetre, taille_case_x):
            pygame.draw.line(fenetre, noir, (x, 0), (x, hauteur_fenetre))
        for y in range(0, hauteur_fenetre, taille_case_y):
            pygame.draw.line(fenetre, noir, (0, y), (largeur_fenetre, y))

   
    # Fonction pour dessiner les operators
    def draw_agent(self, pygame, fenetre, bleu_fonce, agent, taille_case_x, taille_case_y):
       
            pygame.draw.rect(fenetre, bleu_fonce, (agent.get_pos()[0] * taille_case_x, agent.get_pos()[1] * taille_case_y, taille_case_x, taille_case_y))
            
    # Fonction pour dessiner une croix
    def draw_crosses(self, pygame, fenetre, rouge, croix, taille_case_x, taille_case_y):
        for x, y in croix:
            x_pos = x * taille_case_x + taille_case_x // 2
            y_pos = y * taille_case_y + taille_case_y // 2

            pygame.draw.line(fenetre, rouge, (x_pos - 10, y_pos - 10), (x_pos + 10, y_pos + 10), 2)
            pygame.draw.line(fenetre, rouge, (x_pos + 10, y_pos - 10), (x_pos - 10, y_pos + 10), 2)




class UXVSupervisorOperators2DView:

    def __init__(self,implementation,op_ids,sup_ids):
        # Initialisation de la classe MobileSupervisor
            self.implementation = implementation
            print("self.implementation : ",self.implementation)
            self.op_ids = op_ids 
            self.sup_ids = sup_ids
            self.i = 0
            
    def create_inference_op(self):
        op_agents = {self.op_ids[i]:UXV(self.implementation,i) for i in range(len(self.op_ids)) }
        self.i = len(self.op_ids)
        return op_agents
    
    def create_inference_sup(self):
        sup_agents = {self.sup_ids[i]:UXV(self.implementation,i+len(self.op_ids)) for i in range(len(self.sup_ids))}
        return sup_agents

    # Fonction pour dessiner les sous-zones en damier
    def draw_subzones(self, pygame, fenetre, vert, jaune, num_subzones_grid_height, centres_sous_zones, plage_coords, subzones_width, taille_case_x, taille_case_y ):
        couleur1 = jaune 
        couleur2 = vert
           
        for i, centre in enumerate(centres_sous_zones):
        
            if i % 2 == 0 : couleur = couleur1  
            else : couleur = couleur2
            
            for dx in range(plage_coords[0], plage_coords[subzones_width -1] + 1):
                for dy in range(plage_coords[0], plage_coords[subzones_width -1] + 1):
                    x = centre[0] + dx
                    y = centre[1] + dy
                    pygame.draw.rect(fenetre, couleur, (x * taille_case_x, y * taille_case_y, taille_case_x, taille_case_y))
                     
            multiple = (i+1) % num_subzones_grid_height
            if multiple == 0  :
                temp = couleur1
                couleur1 = couleur2 
                couleur2 = temp
            
    # Fonction pour dessiner les sous-zones visités
    def draw_visited_subzones(self, pygame, fenetre, orange, centres_sous_zones_visitees, plage_coords, subzones_width, taille_case_x, taille_case_y ):
        
        for centre in centres_sous_zones_visitees:
            for dx in range(plage_coords[0], plage_coords[subzones_width -1] + 1):
                for dy in range(plage_coords[0], plage_coords[subzones_width -1] + 1):
                    x = centre[0] + dx
                    y = centre[1] + dy
                    pygame.draw.rect(fenetre, orange, (x * taille_case_x, y * taille_case_y, taille_case_x, taille_case_y))
                 
    # Fonction pour dessiner la grille
    def draw_grid(self, pygame, fenetre, noir, hauteur_fenetre, largeur_fenetre, taille_case_x, taille_case_y):
        for x in range(0, largeur_fenetre, taille_case_x):
            pygame.draw.line(fenetre, noir, (x, 0), (x, hauteur_fenetre))
        for y in range(0, hauteur_fenetre, taille_case_y):
            pygame.draw.line(fenetre, noir, (0, y), (largeur_fenetre, y))

    # Fonction pour dessiner le supervisor
    def draw_supervisor(self,pygame, fenetre, bleu_clair, mobileSupervisor, taille_case_x, taille_case_y ):
        for i in range(len(mobileSupervisor)):
            #print("op x : ",mobileSupervisor[i].get_pos()[0])
            #print("op y : ",mobileSupervisor[i].get_pos()[1])
            pygame.draw.rect(fenetre, bleu_clair, (mobileSupervisor[i].get_pos()[0] * taille_case_x, mobileSupervisor[i].get_pos()[1] * taille_case_y, taille_case_x, taille_case_y))
    
    # Fonction pour dessiner les operators
    def draw_operators(self, pygame, fenetre, bleu_fonce, mobileOperators, taille_case_x, taille_case_y):
        for i in range(len(mobileOperators)):
            #print("op x : ",mobileOperators[i].get_pos()[0])
            #print("op y : ",mobileOperators[i].get_pos()[1])
            pygame.draw.rect(fenetre, bleu_fonce, (mobileOperators[i].get_pos()[0] * taille_case_x, mobileOperators[i].get_pos()[1] * taille_case_y, taille_case_x, taille_case_y))
            
    # Fonction pour dessiner une croix
    def draw_crosses(self, pygame, fenetre, rouge, croix, taille_case_x, taille_case_y):
        for x, y in croix:
            x_pos = x * taille_case_x + taille_case_x // 2
            y_pos = y * taille_case_y + taille_case_y // 2

            pygame.draw.line(fenetre, rouge, (x_pos - 10, y_pos - 10), (x_pos + 10, y_pos + 10), 2)
            pygame.draw.line(fenetre, rouge, (x_pos + 10, y_pos - 10), (x_pos - 10, y_pos + 10), 2)


