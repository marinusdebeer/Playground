import pygame
import random
import sys
import numpy as np

# Global Constants for the game
WIDTH = 800
HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
DARK_GRAY = (30, 30, 30)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)

# Paddle Settings
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 15
PADDLE_SPEED = 10

# Ball Settings
INITIAL_BALL_SPEED = 5
BALL_RADIUS = 10

# Brick Settings (no gaps between bricks)
BRICK_ROWS = 6
BRICK_COLS = 10
BRICK_WIDTH = 70
BRICK_HEIGHT = 20
BRICK_OFFSET_TOP = 60
BRICK_OFFSET_LEFT = (WIDTH - (BRICK_COLS * BRICK_WIDTH)) // 2

class Paddle:
    def __init__(self):
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
        self.x = (WIDTH - self.width) // 2
        self.y = HEIGHT - self.height - 30
        self.speed = PADDLE_SPEED
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.dx = 0

    def move(self, direction):
        # direction: -1 (left), 0 (stay), 1 (right)
        self.dx = direction * self.speed
        self.x += self.dx
        if self.x < 0:
            self.x = 0
        if self.x + self.width > WIDTH:
            self.x = WIDTH - self.width
        self.rect.x = self.x

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self.rect)

class Ball:
    def __init__(self):
        self.radius = BALL_RADIUS
        self.reset()

    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = INITIAL_BALL_SPEED
        angle = random.uniform(-0.5, 0.5)
        self.vx = self.speed * random.choice([-1, 1]) * (1 + abs(angle))
        self.vy = -self.speed

    def move(self):
        self.x += self.vx
        self.y += self.vy

    def draw(self, surface):
        pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), self.radius)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius,
                           self.radius * 2, self.radius * 2)

    def increase_speed(self, factor=1.01):
        speed = (self.vx ** 2 + self.vy ** 2) ** 0.5
        new_speed = speed * factor
        if new_speed > 15:
            new_speed = 15
        norm = speed if speed != 0 else 1
        self.vx = (self.vx / norm) * new_speed
        self.vy = (self.vy / norm) * new_speed

class Brick:
    def __init__(self, x, y, color):
        self.rect = pygame.Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT)
        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)

def create_bricks():
    bricks = []
    colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
    for row in range(BRICK_ROWS):
        for col in range(BRICK_COLS):
            x = BRICK_OFFSET_LEFT + col * BRICK_WIDTH
            y = BRICK_OFFSET_TOP + row * BRICK_HEIGHT
            color = colors[row % len(colors)]
            bricks.append(Brick(x, y, color))
    return bricks

class BreakoutEnv:
    """
    A custom environment for Breakout with a simplified state:
      [paddle_x, ball_x, ball_y, ball_vx, ball_vy]
    All positions are normalized.
    Action space: 0 (move left), 1 (stay), 2 (move right)
    """
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Breakout DQN")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.paddle = Paddle()
        self.ball = Ball()
        self.bricks = create_bricks()
        self.lives = 3
        self.score = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        state = np.array([
            self.paddle.x / (WIDTH - self.paddle.width),
            self.ball.x / WIDTH,
            self.ball.y / HEIGHT,
            self.ball.vx / 15.0,  # normalized by max speed
            self.ball.vy / 15.0
        ], dtype=np.float32)
        return state

    def step(self, action):
        if self.done:
            return self._get_state(), 0, self.done, {}

        # Map action: 0 -> left, 1 -> no move, 2 -> right
        if action == 0:
            direction = -1
        elif action == 2:
            direction = 1
        else:
            direction = 0

        self.paddle.move(direction)
        self.ball.move()

        reward = 0

        # Ball collision with walls
        if self.ball.x - self.ball.radius <= 0:
            self.ball.x = self.ball.radius
            self.ball.vx *= -1
        if self.ball.x + self.ball.radius >= WIDTH:
            self.ball.x = WIDTH - self.ball.radius
            self.ball.vx *= -1
        if self.ball.y - self.ball.radius <= 0:
            self.ball.y = self.ball.radius
            self.ball.vy *= -1

        # Ball collision with paddle
        if self.paddle.rect.collidepoint(self.ball.x, self.ball.y + self.ball.radius):
            self.ball.vy = -abs(self.ball.vy)
            hit_pos = (self.ball.x - self.paddle.x) / self.paddle.width
            self.ball.vx = INITIAL_BALL_SPEED * ((hit_pos - 0.5) * 2) + 0.2 * self.paddle.dx
            reward += 0.1

        # Handle brick collisions (process one brick per step)
        ball_rect = self.ball.get_rect()
        for brick in self.bricks:
            if brick.rect.colliderect(ball_rect):
                self.ball.vy *= -1
                self.bricks.remove(brick)
                reward += 10
                self.score += 10
                self.ball.increase_speed()
                break

        # Ball falls below paddle
        if self.ball.y - self.ball.radius > HEIGHT:
            self.lives -= 1
            reward -= 5
            if self.lives <= 0:
                self.done = True
            else:
                self.ball.reset()
                self.paddle = Paddle()

        # Win condition: all bricks cleared
        if not self.bricks:
            reward += 50
            self.done = True

        if self.render_mode:
            self.render()

        next_state = self._get_state()
        return next_state, reward, self.done, {}

    def render(self):
        self.clock.tick(FPS)
        self.screen.fill(DARK_GRAY)
        self.paddle.draw(self.screen)
        self.ball.draw(self.screen)
        for brick in self.bricks:
            brick.draw(self.screen)
        font = pygame.font.SysFont("Arial", 24)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        lives_text = font.render(f"Lives: {self.lives}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (WIDTH - 100, 10))
        pygame.display.flip()

    def close(self):
        if self.render_mode:
            pygame.quit()
