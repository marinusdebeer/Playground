import pygame
from random import randint
from collections import deque
import numpy as np
import cv2
import time
import random
# Define constants
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
BLOCK_SIZE = 20
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Define the Snake class
class Snake:
    def __init__(self):
        self.positions = [(100, 100), (80, 100), (60, 100)]
        self.direction = RIGHT
        self.length = 3
        self.color = GREEN
    
    def get_head_position(self):
        return self.positions[0]
    
    def get_tail_position(self):
        return self.positions[-1]
    
    def turn(self, direction):
        # if self.length > 1 and (direction[0] * -1, direction[1] * -1) == self.direction:
            # return
        # else:
        self.direction = direction
    
    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = ((cur[0] + (x * BLOCK_SIZE)), (cur[1] + (y * BLOCK_SIZE)))
        
        self.positions = [new] + self.positions
        if len(self.positions) > self.length:
            self.positions.pop()
        return self.positions[-1]
    
    def reset(self):
        self.positions = [(100, 100), (80, 100), (60, 100)]
        self.direction = RIGHT
        self.length = 3
        self.color = GREEN
    
    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, BLACK, r, 1)
    
    def check_collision(self):
        head = self.get_head_position()
        if head[0] < 0 or head[0] > WINDOW_WIDTH - BLOCK_SIZE or head[1] < 0 or head[1] > WINDOW_HEIGHT - BLOCK_SIZE:
            print("before death", head, self.direction)
            return True
        for p in self.positions[1:]:
            if head == p:
                print("before death", head, self.direction)
                return True
        return False
class Apple:
    def __init__(self):
        self.position = (0, 0)
        self.randomize_position(None)
    
    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]), (BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(surface, RED, r)
        pygame.draw.rect(surface, BLACK, r, 1)
    
    def randomize_position(self, snake):
        while True:
            x = randint(0, (WINDOW_WIDTH-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = randint(0, (WINDOW_HEIGHT-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if snake is None or (x, y) not in snake.positions:
                self.position = (x, y)
                break
from collections import deque







def ai_next_moves(snake, target, grid):
    head = snake.get_head_position()
    start = (head[0] // BLOCK_SIZE, head[1] // BLOCK_SIZE)
    target = (target[0] // BLOCK_SIZE, target[1] // BLOCK_SIZE)
    print("start", start, "target", target)
    queue = deque([start])
    path = {start: [start]}
    
    while queue:
        curr = queue.popleft()
        if curr == target:
            break
        x, y = curr
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < GRID_WIDTH) and (0 <= ny < GRID_HEIGHT) and (grid[nx][ny] == 0) and ((nx, ny) not in path) and ((nx, ny) not in snake.positions):
                queue.append((nx, ny))
                # visited.add((nx, ny))
                path[(nx, ny)] = path[curr] + [(nx, ny)]
    
    if target in path:
        return [tuple(val * BLOCK_SIZE for val in node) for node in path[target][1:]]
    else:
        print("NOT FOUND")
        return None

""" def ai_next_moves(snake, target, grid):
    def bfs(queue, visited, path):
        print('hi')
        if not queue:
            return None
        curr = queue.popleft()
        if curr == target:
            return path[curr][1:]
        x, y = curr
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 < nx < GRID_WIDTH and 0 < ny < GRID_HEIGHT and grid[nx][ny] == 0 and (nx, ny) not in visited:
                queue.append((nx, ny))
                new_visited = visited.copy()  # create a copy of the visited set
                new_visited.add((nx, ny))  # add the new position to the copy
                path[(nx, ny)] = path[curr] + [(nx, ny)]
                result = bfs(queue, new_visited, path)  # recursive call
                if result is not None:
                    return result
        return None
    head = snake.get_head_position()
    start = (head[0] // BLOCK_SIZE, head[1] // BLOCK_SIZE)
    target = (target[0] // BLOCK_SIZE, target[1] // BLOCK_SIZE)
    queue = deque([start])
    visited = set(snake.positions)
    path = {start: [start]}
    return bfs(queue, visited, path) """
def generate_grid(snake, apple):
    grid = [[0 for _ in range(GRID_HEIGHT)] for _ in range(GRID_WIDTH)]
    
    for x, y in snake.positions:
        try:
            grid[y//BLOCK_SIZE][x//BLOCK_SIZE] = 1
        except:
            print("DIVISION", x, y)
    grid[apple[1]//BLOCK_SIZE][apple[0]//BLOCK_SIZE] = 2
    
    return grid

# def generate_grid(snake, apple):
#     grid = [[0 for y in range(GRID_HEIGHT+1)] for x in range(GRID_WIDTH+1)]
#     for segment in snake.positions:
#         grid[segment[1]//BLOCK_SIZE][segment[0]//BLOCK_SIZE] = 1
#     grid[apple.position[1]//BLOCK_SIZE][apple.position[0]//BLOCK_SIZE] = 2
#     return grid
def show_grid():
    np_grid = np.array(grid)
    img = np.uint8(np_grid * 127)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Grid Image", img)

def random_action():
    head = snake.get_head_position()
    actions = [UP, DOWN, LEFT, RIGHT]
    action_probs = np.ones(len(actions)) / len(actions)
    for i, action in enumerate(actions):
        next_pos = (head[0] + action[0]*BLOCK_SIZE, head[1] + action[1]*BLOCK_SIZE)
        if (next_pos[0] <= 0 or next_pos[0] >= WINDOW_WIDTH or next_pos[1] <= 0 or next_pos[1] >= WINDOW_HEIGHT):
            action_probs[i] = 0
        if next_pos in snake.positions:
            action_probs[i] = 0
    
    action_probs /= np.sum(action_probs)
    
    try:
        action = actions[np.random.choice(len(actions), p=action_probs)]
    except:
        action = UP
    # print("PROBS: ", action_probs, "action: ", action, next_pos)
    return action
pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
snake = Snake()
apple = Apple()
highscore = 0
frame = 0
GRID_WIDTH = WINDOW_WIDTH // BLOCK_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // BLOCK_SIZE
grid = generate_grid(snake, apple.position)
game_over = False
ai_player = True
SPEED = 10

while True:
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                action = UP
            elif event.key == pygame.K_s:
                action = DOWN
            elif event.key == pygame.K_a:
                action = LEFT
            elif event.key == pygame.K_d:
                action = RIGHT
            elif event.key == pygame.K_SPACE:
                ai_player = not ai_player
            # snake.turn(action)
    if ai_player:
        grid = generate_grid(snake, apple.position)
        # path = ai_next_moves(snake.get_head_position(), GRID_WIDTH, GRID_HEIGHT, snake.positions, apple.position)
        path = ai_next_moves(snake, apple.position, grid)
        # print(path)
        if path is not None and len(path) > 0:
            head = snake.get_head_position()
            next_cell = path.pop(0)
            
            direction = (next_cell[0] - head[0], next_cell[1] - head[1])
            if direction == (0, -BLOCK_SIZE):
                action = UP
            elif direction == (0, BLOCK_SIZE):
                action = DOWN
            elif direction == (-BLOCK_SIZE, 0):
                action = LEFT
            elif direction == (BLOCK_SIZE, 0):
                action = RIGHT
            # print("next_cell", next_cell, "action", action)
        else:
            # apple.randomize_position(snake)
            print("APPLE",apple.position)
            # continue
    if action == None:
        action = random_action()
        head = snake.get_head_position()
        next_pos = (head[0] + action[0]*BLOCK_SIZE, head[1] + action[1]*BLOCK_SIZE)
        print("RANDOM NEXT_CELL", next_pos)
    print("next_cell", next_cell, "action", action)
    # snake.turn(action)
    snake.direction = action
    print("DIRECTION", snake.direction)
    print("SNAKE", snake.positions)
    print("CHECK", next_cell in snake.positions)
    if next_cell not in snake.positions:
        snake.move()
    
    
    if snake.check_collision():
        highscore = max(highscore, snake.length)
        print("head", snake.get_head_position(), "path", path, "apple", apple.position)
        print("DIED")
        # snake.reset()
        # show_grid()
        time.sleep(100000000)
        game_over = True
        # apple.randomize_position(snake)
    elif snake.get_head_position() == apple.position:
        snake.length += 1
        apple.randomize_position(snake)
    
    screen.fill(BLACK)
    snake.draw(screen)
    apple.draw(screen)
    grid = generate_grid(snake, apple.position)
    show_grid()
    pygame.display.update()
    clock.tick(SPEED)
    # if game_over:
        # time.sleep(100000000)
# pygame.quit()