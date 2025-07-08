from typing import List
from typing import Tuple
import pygame
from hexagon_env_parallel.hexagon import FlatTopHexagonTile
from hexagon_env_parallel.hexagon import HexagonTile
import random 
import time
import numpy as np

honey_color = tuple([255,195,11])
honey_color_border = tuple([181,101,29])
col = tuple([137,207,240])
rosso_corsa = tuple([212,0,0])
light_red = tuple([215,95,86])
green = tuple([0,86,63])

SIZEOFGRID = (79,39)
PREDATORCOLOR = tuple([255,0,0])
PREDATORVISIONCOLOR = tuple([50,74,178])
PREYCOLOR = tuple([0,255,0])
PREYVISIONCOLOR = tuple([64,224,208])


# prey energy gain while staying
ALPHA = 8   # natural growth
# predator energy gain while eating
DELTA = 30   # 
# predator energy loss while moving
GAMMA =.5

# LIST OF LENGTH 79 * 39 = 3081

"""
list of agents(type)
1. predator agents [list] stores 3-tuple (x,y,z) where x,y represents axial coordinates and 
   z represents direction
2. prey agents [list] stores 2-tuple (x,y) where x,y represent axial coordinates
"""
predatorAgents = []
preyAgents = []

def initializeAgent(noOfPredator,noOfPrey):
    """
    initializes agents and returns them
    prey agent = (x,y, actionCount, energy)
    predator agent = (x,y, actionCount(as a tuple defined below), energy, direction)
    actionCount = (-1,0) initially 
    actionCount[1] = 1/2... implies the action needed to be taken after count reaches 0
    # predator has 2 more possible actions 8,9 that orientation/pov of the predator agent
    actionCount[0] = -1 implies new action can be allocated
    actionCount[0] = 0 implies action has to be implemented now which is stored on actionCount[1] = 0/1/2...
    actionCount[0] = n implies n steps/ticks needed for the action to be completed
    """
    predatorAgents = []
    preyAgents = []
    for agent in range(noOfPrey):
        temp = list_to_axial(random.randint(0,3081))
        preyAgents.append([temp[0],temp[1],(-1,1),random.randint(0,100)])

    for agent in range(noOfPredator):
        temp = list_to_axial(random.randint(0,3081))
        predatorAgents.append([temp[0],temp[1],(-1,1),random.randint(80,100),random.randint(1,6)])
    return predatorAgents,preyAgents


    
     

        
def randomMovement(preyAgents,predatorAgents,hexagon):
    """ randomly moves predator and prey agents if counter is == -1
    """
    for agentIndex,agent in enumerate(preyAgents):
        if agent[2][0] == -1:
            # allocate action 
            action = random.randint(1,7)
            agent[2] = (2,action)

        elif agent[2][0] == 0:
            # do action
            agent[0],agent[1] = directionGeneratorv2(agent[0],agent[1],agent[2][1])
            
            # if staying in one place then energy increases by 15
            if agent[2][1] == 7:
                agent[3] += 10
                if agent[3] >= 100:
                    x,y = direction_generator(agent[0],agent[1])
                    preyAgents.append([x,y,(-1,0),int(agent[3]/2)])
                    agent[3] = int(agent[3]/2)
            
            agent[2] = (-1,1)
            
        else:
            agent[2] = (agent[2][0]-1,agent[2][1])

    for agentIndex,agent in enumerate(predatorAgents):
        if agent[2][0] == -1:
            # allocate action
            action = random.randint(1,9)
            agent[2] = (1,action)


        elif agent[2][0] == 0:
            # do action

            if agent[2][1]<8:
                agent[0],agent[1] = directionGeneratorv2(agent[0],agent[1],agent[2][1])
            elif agent[2][1]== 8:
                if (agent[4]-1) <1:
                    agent[4] = 6
                agent[4] = agent[4] -1
            else:
                if (agent[4]+1) >6:
                    agent[4] = 1
                agent[4] = agent[4] +1
            
            
            agent[2] = (-1,1)

        else:
            agent[2] = (agent[2][0]-1,agent[2][1])

def list_to_axial(index):
    """from list/index to axial coordinates
    """
    x = 0
    y = 0
    #code here
    quotient = int(index/79)
    index = index % 79
    x = index
    y = -1 * int(index/2) + quotient
    return (x,y)

def axial_to_list(axial):
    """axial to list/index 
    """
    index = axial[0] + (axial[1]+int(axial[0]/2))*79
    index = index % ((79*39))  
    return int(index)


def create_hexagon(position, radius=10, flat_top=False) -> HexagonTile:
    """Creates a hexagon tile at the specified position"""
    class_ = FlatTopHexagonTile if flat_top else HexagonTile
    return class_(radius, position, colour=honey_color)

def init_hexagons(num_x=78, num_y=39, flat_top=False) -> List[HexagonTile]:
    """Creates a hexaogonal tile map of size num_x * num_y"""
    # pylint: disable=invalid-name
    leftmost_hexagon = create_hexagon(position=(10,10), flat_top=flat_top)
    hexagons = [leftmost_hexagon]
    for x in range(num_y):
        if x:
            # alternate between bottom left and bottom right vertices of hexagon above
            index = 2 if x % 2 == 1 or flat_top else 4
            position = leftmost_hexagon.vertices[index]
            leftmost_hexagon = create_hexagon(position, flat_top=flat_top)
            hexagons.append(leftmost_hexagon)

        # place hexagons to the left of leftmost hexagon, with equal y-values.
        hexagon = leftmost_hexagon
        for i in range(num_x):
            x, y = hexagon.position  # type: ignore
            if flat_top:
                if i % 2 == 1:
                    position = (x + hexagon.radius * 3 / 2, y - hexagon.minimal_radius)
                else:
                    position = (x + hexagon.radius * 3 / 2, y + hexagon.minimal_radius)
            else:
                position = (x + hexagon.minimal_radius * 2, y)
            hexagon = create_hexagon(position, flat_top=flat_top)
            hexagons.append(hexagon)

    return hexagons

def render(screen, hexagons):
    """Renders hexagons on the screen"""
    screen.fill(honey_color)
    for hexagon in hexagons:
        hexagon.render(screen)

    # draw borders around colliding hexagons and neighbours
    mouse_pos = pygame.mouse.get_pos()
    colliding_hexagons = [
        hexagon for hexagon in hexagons if hexagon.collide_with_point(mouse_pos)
    ] 
    for hexagon in hexagons:
        hexagon.render_highlight(screen, border_colour=honey_color_border)
    for hexagon in colliding_hexagons:
        # for neighbour in hexagon.compute_neighbours(hexagons):
        #     neighbour.render_highlight(screen, border_colour=(100, 100, 100))
        hexagon.render_highlight(screen, border_colour=(255, 255, 255))
        
    pygame.display.flip()

def preyDirectionGenerator(currentX,currentY,action):
    """given current x and y returns nexy x and y in random direction
      """
    x = currentX
    y = currentY
    if (action == 1):
        x = x
        y = y - 1
    elif (action == 2):
        x=x+1
        y=y-1
    elif (action == 3):
        x=x+1
        y=y
    elif (action == 4):
        x=x
        y=y+1
    elif (action == 5):
        x=x-1
        y=y+1
    elif (action == 6):
        x=x-1
        y=y
    elif (action==7):
        x=x
        y=y
    x,y = list_to_axial(axial_to_list((x,y)))
    return x,y

def predatorDirectionGenerator(currentX,currentY,dir,action):
    """given current x and y returns nexy x and y in random direction
      """
    x = currentX
    y = currentY
    
    # if (currentX % 79 == 0) and (action == 5 or action ==  6):
    #     action == 7
    #     return x,y,dir
    # elif (currentX % 78 == 0) and (action == 2 or action == 3):
    #     action == 7
    #     return x,y,dir
        
    if (action == 1):
        x = x
        y = y - 1
    elif (action == 2):
        x=x+1
        y=y-1
    elif (action == 3):
        x=x+1
        y=y
    elif (action == 4):
        x=x
        y=y+1
    elif (action == 5):
        x=x-1
        y=y+1
    elif (action == 6):
        x=x-1
        y=y
    elif (action==7):
        x=x
        y=y
    elif (action==8):
        x=x
        y=y
        dir = (dir +1) % 7
    elif (action==9):
        if dir > 2:
            dir = dir -1
        elif dir == 1:
            dir = 6
        x=x
        y=y
    x,y = list_to_axial(axial_to_list((x,y)))
    return x,y,dir

def direction_generator(currentx,currenty,action):
    """given current x and y returns nexy x and y in random direction
      """
    dir = action
    x = currentx
    y = currenty
    if (dir == 1):
        x = x
        y = y - 1
    elif (dir == 2):
        x=x+1
        y=y-1
    elif (dir == 3):
        x=x+1
        y=y
    elif (dir == 4):
        x=x
        y=y+1
    elif (dir == 5):
        x=x-1
        y=y+1
    elif (dir == 6):
        x=x-1
        y=y
    elif (dir==7):
        x=x
        y=y
    return x,y

def predator_vision(axialCoordinate, dir_vec):
    """ given axial coordinate and direction returns all axial coordinates that predetor can see
    """
    axial_x , axial_y = axialCoordinate
    initial_posx, initial_posy = axial_x , axial_y
    a,b,c,d = 0,0,0,0
    predVis = []
    if (dir_vec == 1):
        a=1
        b=-1
        c=-1
        d=0
    elif (dir_vec == 2):
        a=1
        b=0
        c=0
        d=-1
    elif (dir_vec == 3):
        a=0
        b=1
        c=1
        d=-1
    elif (dir_vec == 4):
        a=-1
        b=1
        c=1
        d=0
    elif (dir_vec == 5):
        a=-1
        b=0
        c=0
        d=1
    elif (dir_vec == 6):
        a=0
        b=-1
        c=-1
        d=1
    for i in range(0,3):
        for j in range(0,3):
            check_x, check_y = (initial_posx + j*c), (initial_posy +j*d) #rightshift 
            predVis.append((check_x,check_y))
        initial_posx, initial_posy= initial_posx + a, initial_posy+ b  #leftshift
    
    # for vis in predVis:
    #     x,y = list_to_axial(axial_to_list(vis))
        
    return predVis

def predator_visionV2(axialCoordinate, dir_vec):
    """ given axial coordinate and direction returns all axial coordinates that predetor can see
    """
    axial_x , axial_y = axialCoordinate
    initial_posx, initial_posy = axial_x , axial_y
    a,b,c,d = 0,0,0,0
    predVis = []
    if (dir_vec == 1):
        a=1
        b=-1
        c=-1
        d=0
    elif (dir_vec == 2):
        a=1
        b=0
        c=0
        d=-1
    elif (dir_vec == 3):
        a=0
        b=1
        c=1
        d=-1
    elif (dir_vec == 4):
        a=-1
        b=1
        c=1
        d=0
    elif (dir_vec == 5):
        a=-1
        b=0
        c=0
        d=1
    elif (dir_vec == 6):
        a=0
        b=-1
        c=-1
        d=1
    for i in range(0,3):
        for j in range(0,3):
            check_x, check_y = (initial_posx + j*c), (initial_posy +j*d) #rightshift 
            predVis.append((check_x,check_y))
        initial_posx, initial_posy= initial_posx + a, initial_posy+ b  #leftshift
    return predVis

def list_neighbours(axial):
    """ returns a list of neighbours of the initial axial point
    """
    initial_posx, initial_posy = axial[0], axial[1]
    preyvis = []
    p1_x, p1_y = (initial_posx + 0),(initial_posy - 1)
    p2_x, p2_y = (initial_posx + 1),(initial_posy - 1)
    p3_x, p3_y = (initial_posx + 1 ),(initial_posy + 0)
    p4_x, p4_y = (initial_posx + 0),(initial_posy + 1)
    p5_x, p5_y = (initial_posx - 1),(initial_posy + 1)
    p6_x, p6_y = (initial_posx - 1),(initial_posy + 0)
    preyvis.append((p1_x,p1_y))
    preyvis.append((p2_x,p2_y))
    preyvis.append((p3_x,p3_y))
    preyvis.append((p4_x,p4_y))
    preyvis.append((p5_x,p5_y))
    preyvis.append((p6_x,p6_y))
    return preyvis

def prey_vision(axial,radius):
    """ given axial, radius
        i.e if radius is 3 then loop gardai append gardai set nikalera firta dine
    """
    nodes = {(axial[0],axial[1])}

    for i in range(radius+1):
        if i == 1:
            neighbours = list_neighbours(axial)
            for neighbour in neighbours:
                nodes.add(neighbour)
            
        elif i>1:
            for neighbour in neighbours:
                temp_neighbours = list_neighbours(neighbour)
                for temp_neighbour in temp_neighbours:
                    nodes.add(temp_neighbour)
            neighbours = list(nodes)
        
    nodes.remove((axial[0],axial[1]))
    return list(nodes)

def reset_env ():
    return 0

def clearGrid(hexagons):
    for hexagon in hexagons:
        hexagon.colour = honey_color
         
def coordinate_movement(axial_x, axial_y,hexa,color):
    """ given axial coordinates, moves the prosthetic agent """
    x,y = direction_generator(axial_x,axial_y)
    hexa[axial_to_list(axial_x,axial_y)].colour = honey_color
    hexa[axial_to_list(x,y)].colour = color
    return x,y

# def main():
#     """Main function"""
#     a = 0
#     pygame.init()
#     screen = pygame.display.set_mode((1200, 700))
#     clock = pygame.time.Clock()
#     hexagons = init_hexagons(flat_top=True)
#     terminated = False
#     predatorAgents,preyAgents = initializeAgent(500,500)

#     while not terminated:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 terminated = True

        
#         randomMovement(preyAgents,predatorAgents,hexagons)
#         predatorBehaviourCheck(preyAgents,predatorAgents)
#         clearGrid(hexagons)
#         renderAgents(preyAgents,predatorAgents,hexagons)
#         render(screen, hexagons)
        
#         clock.tick(3 )
#     pygame.display.quit()

# if __name__ == "__main__":
#     main()

