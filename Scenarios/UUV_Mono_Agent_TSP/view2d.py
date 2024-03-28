from Scenarios.UUV_Mono_Agent_TSP.bridge import UUV



class UUVMonoAgentTSPEnv2dView:

    def __init__(self,implementation):
        # Initialisation de la classe MobileSupervisor
            self.implementation = implementation
            print("self.implementation : ",self.implementation)
            
            self.i = 0
            
    def create_inference_agent(self):
        
        agent = UUV(self.implementation) 
        
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

