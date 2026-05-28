import numpy as np
import random
import math

class RRTNode:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent



points_str = input().strip()
points = [[float(x) for x in pair.split(",")] for pair in points_str.split(";")]
start = points[0]
goal = points[1]


obstacles = []
while True:
    line = input().strip()
    if line == "-1":
        break
    obstacles.append([[float(x) for x in pair.split(",")] for pair in line.split(";")])

grid = np.zeros((1000, 1000))

def mark_obstacles(grid, obstacles):
    for o in obstacles:
        x_min = int(min(o[0][0], o[1][0]))
        x_max = int(max(o[0][0], o[1][0]))
        y_min = int(min(o[0][1], o[1][1]))
        y_max = int(max(o[0][1], o[1][1]))
        
        grid[x_min:x_max+1, y_min:y_max+1] = 1

mark_obstacles(grid, obstacles)

def is_collision_free(x1, y1, x2, y2, grid):

    distance = math.hypot(x2 - x1, y2 - y1)
    steps = int(distance)  
    if steps == 0:
        return True
        
    for i in range(steps + 1):
        t = i / steps
        curr_x = int(round(x1 + t * (x2 - x1)))
        curr_y = int(round(y1 + t * (y2 - y1)))
        
        if 0 <= curr_x < 1000 and 0 <= curr_y < 1000:
            if grid[curr_x][curr_y] == 1:
                return False
        else:
            return False 
    return True

def find_waypoints(grid, start, goal, step_size=5.0, max_iter=5000):

    tree = [RRTNode(start[0], start[1])]
    
    for _ in range(max_iter):
        if random.random() < 0.10:
            x_rand, y_rand = goal[0], goal[1]
        else:
            x_rand = random.uniform(0, 999)
            y_rand = random.uniform(0, 999)
            
        nearest_node = min(tree, key=lambda node: math.hypot(node.x - x_rand, node.y - y_rand))
        
        angle = math.atan2(y_rand - nearest_node.y, x_rand - nearest_node.x)
        
        x_new = nearest_node.x + step_size * math.cos(angle)
        y_new = nearest_node.y + step_size * math.sin(angle)
        
        if is_collision_free(nearest_node.x, nearest_node.y, x_new, y_new, grid):
            new_node = RRTNode(x_new, y_new, parent=nearest_node)
            tree.append(new_node)
            
            if math.hypot(new_node.x - goal[0], new_node.y - goal[1]) <= step_size:
                if is_collision_free(new_node.x, new_node.y, goal[0], goal[1], grid):
                    goal_node = RRTNode(goal[0], goal[1], parent=new_node)
                    tree.append(goal_node)
                    return goal_node
                    
    return None 

goal_node = find_waypoints(grid, start, goal)

if goal_node:
    path = []
    curr = goal_node
    while curr is not None:
        path.append((curr.x, curr.y))
        curr = curr.parent

    path.reverse()
    
    for wp in path:

        print(f"{int(round(wp[0]))},{int(round(wp[1]))}")
else:
    print("No path found.")