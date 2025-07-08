# def list_to_axial(index):
#     x = 0
#     y = 0
#     #code here
#     quotient = int(index/79)
#     index = index % 79
#     x = index
#     y = -1 * int(index/2) + quotient
#     return x,y

# def axial_to_list(axial_x,axial_y):
#     #code here
#     index = axial_x + (axial_y+int(axial_x/2))*79
#     return index

# def main():
#     print(list_to_axial(79*38 -1 -79))
#     print(axial_to_list(0,-1))
    
# if __name__ == "__main__":
#     main()

# def axial_to_cube(axial_x, axial_y):
#     cube_s = 0
#     cube_q = axial_x
#     cube_r = axial_y
#     cube_s = - cube_q - cube_r
#     return (cube_q, cube_r, cube_s)

# def check_prey(x,y):
#     return 0



# def axial_to_cube(axial_x, axial_y):
#     cube_s = 0
#     cube_q = axial_x
#     cube_r = axial_y
#     cube_s = - cube_q - cube_r
#     return (cube_q, cube_r, cube_s)



# def predator_vision(axial_coordinate, dir_vec):
#     """ given axial coordinate and direction returns all axial coordinates that predetor can see
#     """
#     axial_x , axial_y = axial_coordinate
#     initial_posx, initial_posy = axial_x , axial_y
#     a,b,c,d = 0,0,0,0
#     predVis = []
#     if (dir_vec == 1):
#         a=1
#         b=-1
#         c=-1
#         d=0axial
#     elif (dir_vec == 2):
#         a=1
#         b=0
#         c=0
#         d=-1
#     elif (dir_vec == 3):
#         a=0
#         b=1
#         c=1
#         d=-1
#     elif (dir_vec == 4):
#         a=-1
#         b=1
#         c=1
#         d=0
#     elif (dir_vec == 5):
#         a=-1
#         b=0
#         c=0
#         d=1
#     elif (dir_vec == 6):
#         a=0
#         b=-1
#         c=-1
#         d=1
#     for i in range(0,3):
#         for j in range(0,3):
#             check_x, check_y = (initial_posx + j*c), (initial_posy +j*d) #rightshift 
#             predVis.append((check_x,check_y))
#         initial_posx, initial_posy= initial_posx + a, initial_posy+ b  #leftshift
        
#     return predVis

# print (predator_vision((0,0),1))


def list_neighbours(axial):
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
    """ given axial,radius
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
        
    return nodes


total = []
axial = (0,0)

a = prey_vision(axial,3)
a = list(a)
print(a)
# for b in a:
#     print(b)


# a = prey_vision(axial)
# for i in a:
#     total.append(prey_vision((i[0],i[1])))

# final = []
# for i in total:
#     for j in i:
#         final.append(j)

# print(final)
# print(set(final))


# import numpy as np
# Color = np.linspace(0,255,num = 101,dtype=np.int16)
# print(len(Color))