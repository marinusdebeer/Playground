import pygame
from random import randint
from collections import deque
import numpy as np
import cv2
# Define constants
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
BLOCK_SIZE = 20
SPEED = 10
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

pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
snake = Snake()
apple = Apple()
highscore = 0
frame = 0
GRID_WIDTH = WINDOW_WIDTH // BLOCK_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // BLOCK_SIZE
grid = [[0 for y in range(GRID_HEIGHT+1)] for x in range(GRID_WIDTH+1)]
for segment in snake.positions:
    grid[segment[1]//BLOCK_SIZE][segment[0]//BLOCK_SIZE] = 1
grid[apple.position[1]//BLOCK_SIZE][apple.position[0]//BLOCK_SIZE] = 2
game_over = False
ai_player = True

while not game_over:
    grid = [[0 for y in range(GRID_HEIGHT+1)] for x in range(GRID_WIDTH+1)]
    for segment in snake.positions:
        grid[segment[1]//BLOCK_SIZE][segment[0]//BLOCK_SIZE] = 1
    grid[apple.position[1]//BLOCK_SIZE][apple.position[0]//BLOCK_SIZE] = 2
    np_grid = np.array(grid)
    img = np.uint8(np_grid * 127)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_NEAREST)
    path = ai_next_moves(snake, apple.position, grid)
    
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
    if ai_player:
        if path is not None and len(path) >= 1:
            head = snake.get_head_position()
            next_cell = path.pop(0)
            direction = (next_cell[0] - head[0], next_cell[1] - head[1])
            if direction == (0, -BLOCK_SIZE):
                snake.turn(UP)
            elif direction == (0, BLOCK_SIZE):
                snake.turn(DOWN)
            elif direction == (-BLOCK_SIZE, 0):
                snake.turn(LEFT)
            elif direction == (BLOCK_SIZE, 0):
                snake.turn(RIGHT)
        else:
            print("PATH EMPTY")
    snake.move()
    grid = [[0 for y in range(GRID_HEIGHT+1)] for x in range(GRID_WIDTH+1)]
    for segment in snake.positions:
        grid[segment[1]//BLOCK_SIZE][segment[0]//BLOCK_SIZE] = 1
    grid[apple.position[1]//BLOCK_SIZE][apple.position[0]//BLOCK_SIZE] = 2
    np_grid = np.array(grid)
    img = np.uint8(np_grid * 127)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Grid Image", img)
    if snake.check_collision():
        highscore = max(highscore, snake.length)
        print("DIED")
        snake.reset()
        apple.randomize_position(snake)
        # grid = [[0 for y in range(GRID_HEIGHT+1)] for x in range(GRID_WIDTH+1)]
        # for segment in snake.positions:
        #     grid[segment[1]//BLOCK_SIZE][segment[0]//BLOCK_SIZE] = 1
        # grid[apple.position[1]//BLOCK_SIZE][apple.position[0]//BLOCK_SIZE] = 2
    
    # Check if the snake has eaten the apple
    elif snake.get_head_position() == apple.position:
        snake.length += 1
        apple.randomize_position(snake)
        # grid = [[0 for y in range(GRID_HEIGHT+1)] for x in range(GRID_WIDTH+1)]
        # for segment in snake.positions:
        #     grid[segment[1]//BLOCK_SIZE][segment[0]//BLOCK_SIZE] = 1
        # grid[apple.position[1]//BLOCK_SIZE][apple.position[0]//BLOCK_SIZE] = 2
    
    screen.fill(BLACK)
    snake.draw(screen)
    apple.draw(screen)
    pygame.display.update()
    clock.tick(SPEED)
pygame.quit()