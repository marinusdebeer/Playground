import pygame
import random
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BLOCK_SIZE = 20
FPS = 10
FOOD_SPAWN_RATE = 200  # Food spawn rate in frames

# Colors
COLOR_PLAYER = (0, 0, 255)       # Blue
COLOR_PLAYER_HEAD = (0, 0, 128)  # Dark Blue
COLOR_AI_IMMORTAL = (255, 0, 0)  # Red
COLOR_AI_HEAD = (128, 0, 0)      # Dark Red
COLOR_FOOD = (255, 255, 0)       # Yellow
COLOR_BG = (173, 216, 230)       # Light Blue

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Immortal ticks for new AI snakes
IMMORTAL_TICKS = 10

class Snake:
    def __init__(self, x, y, color, head_color, is_player=False):
        self.body = deque([(x, y)])
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.color = color
        self.head_color = head_color
        self.is_player = is_player
        self.alive = True
        self.immortal_ticks = IMMORTAL_TICKS if not is_player else 0

    def move(self):
        if not self.alive:
            return
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0] * BLOCK_SIZE, head_y + self.direction[1] * BLOCK_SIZE)
        self.body.appendleft(new_head)
        self.body.pop()

    def grow(self):
        tail = self.body[-1]
        self.body.append(tail)

    def set_direction(self, new_direction):
        # Prevent reversing direction
        if (self.direction[0] * -1, self.direction[1] * -1) != new_direction:
            self.direction = new_direction

    def check_collision(self, snakes):
        if self.immortal_ticks > 0:
            self.immortal_ticks -= 1
            return
        head_x, head_y = self.body[0]
        # Check boundary collision
        if head_x < 0 or head_x >= SCREEN_WIDTH or head_y < 0 or head_y >= SCREEN_HEIGHT:
            self.alive = False
            return
        # Check collision with other snakes
        for snake in snakes:
            if not snake.alive:
                continue
            if snake != self and (head_x, head_y) in snake.body:
                self.alive = False
                return

    def find_nearest_food(self, foods):
        head_x, head_y = self.body[0]
        nearest_food = min(foods, key=lambda food: abs(food.position[0] - head_x) + abs(food.position[1] - head_y))
        return nearest_food

    def move_towards_food(self, food):
        possible_directions = [UP, DOWN, LEFT, RIGHT]
        # Exclude reverse direction
        reverse_direction = (-self.direction[0], -self.direction[1])
        possible_directions = [d for d in possible_directions if d != reverse_direction]

        head_x, head_y = self.body[0]

        # Move towards food without worrying about self-collision
        dx = food.position[0] - head_x
        dy = food.position[1] - head_y

        if abs(dx) >= abs(dy):
            if dx > 0:
                preferred_direction = RIGHT
            elif dx < 0:
                preferred_direction = LEFT
            else:
                preferred_direction = None
        else:
            if dy > 0:
                preferred_direction = DOWN
            elif dy < 0:
                preferred_direction = UP
            else:
                preferred_direction = None

        if preferred_direction and preferred_direction in possible_directions:
            self.set_direction(preferred_direction)
        else:
            # If preferred direction is not possible, choose any valid direction
            if possible_directions:
                self.set_direction(random.choice(possible_directions))

class Food:
    def __init__(self, position=None):
        self.position = position

    @staticmethod
    def spawn(occupied_positions):
        # All possible positions
        all_positions = set(
            (x * BLOCK_SIZE, y * BLOCK_SIZE)
            for x in range(SCREEN_WIDTH // BLOCK_SIZE)
            for y in range(SCREEN_HEIGHT // BLOCK_SIZE)
        )
        # Available positions
        available_positions = list(all_positions - occupied_positions)
        if not available_positions:
            return None  # No available positions
        return random.choice(available_positions)

    def draw(self, screen):
        pygame.draw.rect(screen, COLOR_FOOD, pygame.Rect(self.position[0], self.position[1], BLOCK_SIZE, BLOCK_SIZE))

def get_occupied_positions(snakes, foods, snake_foods):
    occupied = set()
    for snake in snakes:
        if snake.alive:
            occupied.update(snake.body)
    for food in foods + snake_foods:
        occupied.add(food.position)
    return occupied

def main():
    # Initialize game elements
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    player_snake = Snake(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, COLOR_PLAYER, COLOR_PLAYER_HEAD, is_player=True)
    ai_snake_colors = [
        ((0, 128, 128), (0, 64, 64)),    # Teal and Dark Teal
        ((128, 0, 128), (64, 0, 64)),    # Purple and Dark Purple
        ((255, 165, 0), (200, 100, 0)),  # Orange and Dark Orange
        ((0, 255, 255), (0, 128, 128)),  # Cyan and Dark Cyan
        ((255, 20, 147), (128, 10, 73))  # Pink and Dark Pink
    ]
    ai_snakes = [
        Snake(
            random.randint(0, SCREEN_WIDTH // BLOCK_SIZE - 1) * BLOCK_SIZE,
            random.randint(0, SCREEN_HEIGHT // BLOCK_SIZE - 1) * BLOCK_SIZE,
            color[0],
            color[1]
        )
        for color in ai_snake_colors
    ]
    foods = []
    snake_foods = []

    snakes = [player_snake] + ai_snakes

    # Initial food spawning
    occupied_positions = get_occupied_positions(snakes, foods, snake_foods)
    for _ in range(20):
        position = Food.spawn(occupied_positions)
        if position:
            foods.append(Food(position=position))
            occupied_positions.add(position)

    frame_count = 0

    running = True
    while running:
        screen.fill(COLOR_BG)

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    player_snake = Snake(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, COLOR_PLAYER, COLOR_PLAYER_HEAD, is_player=True)
                    snakes[0] = player_snake  # Update the snakes list
                elif event.key == pygame.K_UP:
                    player_snake.set_direction(UP)
                elif event.key == pygame.K_DOWN:
                    player_snake.set_direction(DOWN)
                elif event.key == pygame.K_LEFT:
                    player_snake.set_direction(LEFT)
                elif event.key == pygame.K_RIGHT:
                    player_snake.set_direction(RIGHT)

        # Update occupied positions before moves
        occupied_positions = get_occupied_positions(snakes, foods, snake_foods)

        # Move snakes
        for snake in snakes:
            if snake.alive:
                if snake != player_snake:
                    if foods:
                        nearest_food = snake.find_nearest_food(foods)
                        snake.move_towards_food(nearest_food)
                snake.move()

        # Check collisions
        for snake in snakes:
            snake.check_collision(snakes)

        # Update occupied positions after moves
        occupied_positions = get_occupied_positions(snakes, foods, snake_foods)

        # Food consumption
        for food in foods[:]:
            for snake in snakes:
                if snake.alive and snake.body[0] == food.position:
                    snake.grow()
                    foods.remove(food)
                    occupied_positions.remove(food.position)  # Update occupied positions
                    # Spawn new food
                    position = Food.spawn(occupied_positions)
                    if position:
                        new_food = Food(position=position)
                        foods.append(new_food)
                        occupied_positions.add(new_food.position)
                    break  # Only one snake can eat the food

        for food in snake_foods[:]:
            for snake in snakes:
                if snake.alive and snake.body[0] == food.position:
                    snake.grow()
                    snake_foods.remove(food)
                    occupied_positions.remove(food.position)  # Update occupied positions
                    # Optionally, spawn new food if desired
                    break  # Only one snake can eat the food

        # Turn dead snakes into food and respawn AI snakes
        for i, snake in enumerate(snakes[1:], start=1):
            if not snake.alive:
                # Convert dead snake body to food
                for segment in snake.body:
                    if segment not in occupied_positions:
                        snake_foods.append(Food(position=segment))
                        occupied_positions.add(segment)
                # Respawn AI snake
                new_snake = Snake(
                    random.randint(0, SCREEN_WIDTH // BLOCK_SIZE - 1) * BLOCK_SIZE,
                    random.randint(0, SCREEN_HEIGHT // BLOCK_SIZE - 1) * BLOCK_SIZE,
                    ai_snake_colors[i - 1][0],
                    ai_snake_colors[i - 1][1]
                )
                snakes[i] = new_snake
                ai_snakes[i - 1] = new_snake  # Keep ai_snakes list updated
                occupied_positions.update(new_snake.body)

        # Handle player snake death
        if not player_snake.alive:
            for segment in player_snake.body:
                if segment not in occupied_positions:
                    snake_foods.append(Food(position=segment))
                    occupied_positions.add(segment)
            # Uncomment the next lines to respawn the player snake automatically
            # player_snake = Snake(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, COLOR_PLAYER, COLOR_PLAYER_HEAD, is_player=True)
            # snakes[0] = player_snake  # Update the snakes list

        # Spawn new food at a constant rate
        frame_count += 1
        if frame_count % FOOD_SPAWN_RATE == 0:
            position = Food.spawn(occupied_positions)
            if position:
                foods.append(Food(position=position))
                occupied_positions.add(position)

        # Draw elements
        # First, draw all food items
        for food in foods + snake_foods:
            food.draw(screen)

        # Then draw snake bodies (excluding heads)
        for snake in snakes:
            if snake.alive:
                body_segments = list(snake.body)[1:]
                current_color = snake.color
                if snake.immortal_ticks > 0:
                    current_color = COLOR_AI_IMMORTAL
                for segment in body_segments:
                    pygame.draw.rect(screen, current_color, pygame.Rect(segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE))

        # Finally, draw snake heads to ensure they are on top
        for snake in snakes:
            if snake.alive:
                head = snake.body[0]
                head_color = snake.head_color
                if snake.immortal_ticks > 0:
                    head_color = COLOR_AI_IMMORTAL
                pygame.draw.rect(screen, head_color, pygame.Rect(head[0], head[1], BLOCK_SIZE, BLOCK_SIZE))

        # Update display and tick
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == '__main__':
    main()
