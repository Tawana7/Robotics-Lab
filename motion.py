#!/usr/bin/env python

#Students
#2680747 - Gopolang Mohlakola
#2588206 - Tawananyasha Kadango

"""
RRT was chosen because of single query efficiency. RRT builds a map from the start to the goal, 
instead of building a full global map first, which saves computation time. Since the global map is a simple
1000x1000 grid, it need not be built using PRM algorithms. It also avoids creating a large network of waypoints,
since the only relevant way points in this scenario are the ones that get you from the start to the goal.

Note that RTT uses a sampling strategy so it will generate a new wavepoints each run, and these are valid 
traversals towards the goal.
"""
# Testing on a 1000 x 1000 obstacle grid; RTT provides stochastic sampling
import numpy as np
import random
import math
import sys

class RRTNode:
    def __init__(self, x, y, parent=None):
        self.x= x
        self.y= y
        self.parent= parent

def read_in_input():
    lines= []
    while True:
        try:
            line =input().strip()
            lines.append(line)
            if line== '-1':
                break
        except EOFError:
            break
    return lines

def format_input(lines):
    if not lines:
        return None, None, None
    
    start_goal =lines[0].split(';')
    start =[float(x) for x in start_goal[0].split(',')]
    goal =[float(x) for x in start_goal[1].split(',')]
    
    obstacles_raw =[]
    for line in lines[1:-1]:
        if not line or line == '-1':
            continue
        p1, p2 =line.split(';')
        x1, y1 =[float(x) for x in p1.split(',')]
        x2, y2 = [float(x) for x in p2.split(',')]
        obstacles_raw.append((x1, y1, x2, y2))
    
    return start,goal,obstacles_raw

def mark_obstacles(grid, obstacles_raw):
    for (x1,y1,x2,y2) in obstacles_raw:
        x_min =int(min(x1, x2))
        x_max=int(max(x1, x2))
        y_min = int(min(y1, y2))
        y_max = int(max(y1, y2))
        
        expansion =2
        x_min=max(0,x_min-expansion)
        x_max =min(999, x_max+expansion)
        y_min = max(0, y_min-expansion)
        y_max =min(999, y_max + expansion)
        
        
        for x in range(x_min,x_max + 1):
            for y in range(y_min,y_max +1):
                grid[x][y] =1

def bool_free_from_collision(x1,y1,x2, y2, grid):
    x1,y1 =int(round(x1)),int(round(y1))
    x2,y2 =int(round(x2)),int(round(y2))
    
    dx=abs(x2-x1)
    dy =abs(y2-y1)
    sx =1 if x1<x2 else -1
    sy =1 if y1<y2 else -1
    err= dx-dy

    x, y =x1, y1
    while True:
        #Bresenham algorith: between 2 points in a grid, check if theres an obstacle between them
        # acccomodates for squeeze throughs
        if not (0 <= x<1000 and 0 <=y<1000):
            return False
        if grid[x][y] ==1:
            return False
        if x == x2 and y == y2:
            break
        e2 = 2*err
        if e2 >-dy:
            err-=dy
            x+=sx
        if e2<dx:
            err+=dx
            y+=sy
    return True

def smooth_path(path,grid):
    if len(path)<=2:
        return path
    
    smoothed =[path[0]]
    current = 0
    
    while current< len(path)-1:
        for i in range(len(path)-1,current,-1):
            if bool_free_from_collision(path[current][0],path[current][1],path[i][0], path[i][1], grid):
                smoothed.append(path[i])
                current = i
                break
    
    return smoothed

def find_waypoints(grid,start,goal,step_size=15.0,max_iter=80000):
    tree =[RRTNode(start[0],start[1])]
    
    #create sampling points in the grid to traverse in a tree-like manner while avoiding obstacles
    for i in range(max_iter):
        if random.random()< 0.2:
            x_rand, y_rand = goal[0], goal[1]
        else:
            x_rand =random.uniform(0,999)
            y_rand =random.uniform(0,999)
        
        #The RTT algo: move towards random sampled point in the direction to the goal, by increments of step_size
        nearest_node = min(tree,key=lambda node: math.hypot(node.x -x_rand, node.y - y_rand))
        
        #Calc the angle of traversal, then move towards the next sampled point in a direction specified by the angle
        angle =math.atan2(y_rand -nearest_node.y,x_rand -nearest_node.x)
        x_new =nearest_node.x + step_size*math.cos(angle)
        y_new = nearest_node.y+ step_size*math.sin(angle)
        
        x_new = max(0,min(999,x_new))
        y_new = max(0,min(999,y_new))
        
        if bool_free_from_collision(nearest_node.x, nearest_node.y, x_new, y_new, grid):
            new_node =RRTNode(x_new,y_new,parent=nearest_node)
            tree.append(new_node)
            
            #check if we have reached goal node, if not continue sampling
            if math.hypot(new_node.x-goal[0], new_node.y-goal[1]) <=step_size:
                if bool_free_from_collision(new_node.x, new_node.y, goal[0], goal[1], grid):
                    path =[]
                    curr =new_node
                    while curr is not None:
                        path.append((curr.x,curr.y))
                        curr= curr.parent
                    path.reverse()
                    path.append((goal[0], goal[1]))
                    
                   #smooth the planned path around obstacles (avoids erratic behaviour of rrt sampling)
                    path =smooth_path(path, grid)
                    
                    int_path =[]
                    for x, y in path:
                        ix, iy =int(round(x)),int(round(y))
                        if not int_path or (int_path[-1][0] != ix or int_path[-1][1] != iy):
                            int_path.append((ix, iy))
                    
                    #removes redundant points in sampling (very small incremental points are not very important)
                    if len(int_path)> 2:
                        filtered= [int_path[0]]
                        for i in range(1, len(int_path) - 1):
                            prev= filtered[-1]
                            curr= int_path[i]
                            nxt = int_path[i+ 1]
                        
                            cross =abs((curr[0]-prev[0])*(nxt[1]-curr[1])-(curr[1] - prev[1]) * (nxt[0] - curr[0]))
                            
                            if cross>0.5:
                                filtered.append(curr)
                        filtered.append(int_path[-1])
                        int_path = filtered
                    
                    return int_path
    
    return None

def main():
    lines =read_in_input()
    
    start,goal,obstacles_raw =format_input(lines)
    
    if start is None:
        print("Error: Start coordinate not read")
        sys.exit(1)
    
    # Initialise the obstacle grid,this is a 1000x1000 grid
    grid= np.zeros((1000,1000))
    mark_obstacles(grid, obstacles_raw)
    
    #Run RRT 10 times and sample our best path wavepoints
    best_path =None
    for i in range(10):
        result = find_waypoints(grid, start, goal, step_size=15.0, max_iter=80000)
        if result:
            best_path = result
            break
    
    if best_path:
        for wave_point in best_path:
            print("{},{}".format(wave_point[0],wave_point[1]))
    else:
        print("No path found", file=sys.stderr)
        sys.exit(1)

main()
