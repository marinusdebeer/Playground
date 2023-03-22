import pygame
from random import randint
from collections import deque
import numpy as np
import cv2
from tqdm import tqdm
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
        if self.length > 1 and (direction[0] * -1, direction[1] * -1) == self.direction:
            return
        else:
            self.direction = direction
    
    def move(self):
        self.move_counter = 0
        cur = self.get_head_position()
        x, y = self.direction
        new = ((cur[0] + (x * BLOCK_SIZE)), (cur[1] + (y * BLOCK_SIZE)))
        if new in self.positions[2:]:
            self.length = 1
            # self.positions = [new]
            self.reset()
        else:
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
        if self.get_head_position()[0] < 0 or self.get_head_position()[0] > WINDOW_WIDTH - BLOCK_SIZE or \
            self.get_head_position()[1] < 0 or self.get_head_position()[1] > WINDOW_HEIGHT - BLOCK_SIZE:
            return True
        for p in self.positions[1:]:
            if self.get_head_position() == p:
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

def ai_next_moves(snake, target, graph):
    head = snake.get_head_position()
    start = tuple(val // BLOCK_SIZE for val in head)
    target = tuple(val // BLOCK_SIZE for val in target)
    queue = deque([start])
    visited = set((pos[0] // BLOCK_SIZE, pos[1] // BLOCK_SIZE) for pos in snake.positions)
    parent = {start: None}
    while queue:
        curr = queue.popleft()
        if curr == target:
            break
        x, y = curr
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 25 and 0 <= ny < 25 and graph[nx][ny] == 0 and (nx, ny) not in visited:
                queue.append((nx, ny))
                visited.add((nx, ny))
                parent[(nx, ny)] = curr
    path = []
    curr = target
    while curr:
        path.append(tuple(val * BLOCK_SIZE for val in curr))
        if curr in parent:
            curr = parent[curr]
        else:
            curr = None
            print("NOT FOUND")
    path.reverse()
    path.pop(0)
    return path

def select_action(state, q_table):
    actions = [UP, DOWN, LEFT, RIGHT]
    opposite_actions = {
        LEFT: RIGHT,
        RIGHT: LEFT,
        UP: DOWN,
        DOWN: UP
    }
    head = snake.get_head_position()
    action = None
    if np.random.random() > epsilon or not training:
        global choice
        choice += 1
        action = actions[np.argmax(q_table[state])]
        return action
    else:
        action_probs = np.ones(len(actions)) / len(actions)
        for i, action in enumerate(actions):
            next_pos = (head[0] + action[0]*BLOCK_SIZE, head[1] + action[1]*BLOCK_SIZE)
            if (next_pos[0] < 0 or next_pos[0] >= WINDOW_WIDTH or
                next_pos[1] < 0 or next_pos[1] >= WINDOW_HEIGHT):
                action_probs[i] = 0
            if next_pos in snake.positions[1:]:
                action_probs[i] = 0
        
        action_probs /= np.sum(action_probs)
        # print("PROBS: ", action_probs, "action: ", action)
        try:
            action = actions[np.random.choice(len(actions), p=action_probs)]
        except:
            action = UP
        
        return action

def show_grid(grid):
    np_grid = np.array(grid)
    img = np.uint8(np_grid * 127)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Grid Image", img)
def generate_grid():
    grid = [[0 for y in range(GRID_HEIGHT)] for x in range(GRID_WIDTH)]
    for segment in snake.positions:
        grid[segment[1]//BLOCK_SIZE][segment[0]//BLOCK_SIZE] = 1
    grid[apple.position[1]//BLOCK_SIZE][apple.position[0]//BLOCK_SIZE] = 2
    return grid
def get_state(grid):
    q_grid = np.array(grid)
    q_grid = q_grid.reshape((GRID_HEIGHT, GRID_WIDTH))
    snake_x, snake_y = np.where(q_grid == 1)
    apple_x, apple_y = np.where(q_grid == 2)
    state_indices = (snake_x[0], snake_y[0], apple_x[0], apple_y[0])
    return state_indices
def save_q_table(q_table, filename):
    np.save(filename, q_table)
pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
snake = Snake()
apple = Apple()
highscore = 0
frame = 0
epsilon = 1
EPSILON_DECAY = 0.99997
episodes = 100_000
alpha = 0.4
gamma = 0.995
choice = 0
rand = 0
SAVE_LOCATION = "snake_q_tables/"
SAVE_EVERY = 100
GRID_WIDTH = WINDOW_WIDTH // BLOCK_SIZE     #25
GRID_HEIGHT = WINDOW_HEIGHT // BLOCK_SIZE   #25

num_snake_x = GRID_WIDTH 
num_snake_y = GRID_HEIGHT
num_apple_x = GRID_WIDTH
num_apple_y = GRID_HEIGHT
num_actions = 4
SPEED = 10000

grid = generate_grid()
state_indices = get_state(grid)
game_over = False
ai_player = False
ml_player = True
training = True
load_model = False
MODEL_LOCATION = 'snake_q_tables/q_table_episode1130.npy'
apples = 0
q_table = np.zeros((num_snake_x, num_snake_y, num_apple_x, num_apple_y, num_actions))
if not training or load_model:
    q_table = np.load(MODEL_LOCATION)
game_over = False
died = 0
saved = None
while not game_over:
    if apples >= episodes:
        game_over = True
    action = UP
    # print(snake.get_head_position())
    reward = 0
    grid = generate_grid()
    state_indices = get_state(grid)
    if ml_player:
        # state_indices = get_state(grid)
        action = select_action(state_indices, q_table)
        # print(action)
        snake.turn(action)
    elif ai_player:
        path = ai_next_moves(snake, apple.position, grid)
        if path is not None and len(path) >= 1:
            head = snake.get_head_position()
            next_cell = path.pop(0)
            direction = (next_cell[0] - head[0], next_cell[1] - head[1])
            if direction == (0, -BLOCK_SIZE):
                action = UP
                snake.turn(UP)
            elif direction == (0, BLOCK_SIZE):
                action = DOWN
                snake.turn(DOWN)
            elif direction == (-BLOCK_SIZE, 0):
                action = LEFT
                snake.turn(LEFT)
            elif direction == (BLOCK_SIZE, 0):
                action = RIGHT
                snake.turn(RIGHT)
        else:
            print("PATH EMPTY")
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                snake.turn(UP)
            elif event.key == pygame.K_s:
                snake.turn(DOWN)
            elif event.key == pygame.K_a:
                snake.turn(LEFT)
            elif event.key == pygame.K_d:
                snake.turn(RIGHT)
            elif event.key == pygame.K_SPACE:
                ai_player = not ai_player
    
    snake.move()
    
    if snake.check_collision():
        reward = -1
        highscore = max(highscore, snake.length)
        print("DIED")
        snake.reset()
        apple.randomize_position(snake)
        died += 1
        epsilon = pow(EPSILON_DECAY, apples)
        
    
    # Check if the snake has eaten the apple
    elif snake.get_head_position() == apple.position:
        reward = 1
        snake.length += 1
        apple.randomize_position(snake)
        epsilon = pow(EPSILON_DECAY, apples)
        print("EPSILON: ", epsilon)
        apples += 1

    # if ml_player:
    grid = generate_grid()
    next_state_indices = get_state(grid)
    
    if training and ml_player or ai_player:
        q_table[state_indices + (action,)] += alpha * (reward + gamma * np.max(q_table[next_state_indices]) - q_table[state_indices + (action,)])
    
    screen.fill(BLACK)
    # show_grid(grid)
    snake.draw(screen)
    apple.draw(screen)
    pygame.display.update()
    clock.tick(SPEED)

    if apples % SAVE_EVERY == 0 and training and apples != saved:
        saved = apples
        q_table_filename = f"{SAVE_LOCATION}q_table_episode{apples}_died{died}.npy"
        save_q_table(q_table, q_table_filename)
        died = 0
        file_reward = 0
q_table_filename = f"{SAVE_LOCATION}q_table_episode{apples}_died{died}.npy"
save_q_table(q_table, q_table_filename)
pygame.quit()