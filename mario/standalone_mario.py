import pygame
import sys
import pickle

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# World dimensions
WORLD_WIDTH = 3000  # Width of the level in pixels

# Player properties
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 60
PLAYER_COLOR = (255, 0, 0)  # Red

# Colors
PLATFORM_COLOR = (0, 255, 0)       # Green
COIN_COLOR = (255, 223, 0)         # Gold
ENEMY_COLOR = (0, 0, 255)          # Blue
POWERUP_COLOR = (0, 255, 255)      # Cyan
INVINCIBILITY_COLOR = (255, 215, 0)  # Gold
SPEED_BOOST_COLOR = (255, 105, 180)  # Pink
HAZARD_COLOR = (165, 42, 42)       # Brown
TURRET_COLOR = (128, 0, 128)       # Purple
PROJECTILE_COLOR = (255, 0, 0)     # Red

# Gravity
GRAVITY = 0.8

# Initialize screen and clock
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Super Mario Clone")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

# Load images (replace with actual images if available)
def load_image(color, width, height):
    image = pygame.Surface((width, height))
    image.fill(color)
    return image

# Classes

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = load_image(PLAYER_COLOR, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.rect = self.image.get_rect()
        self.rect.x = 100
        self.rect.y = SCREEN_HEIGHT - PLAYER_HEIGHT - 40
        self.velocity_y = 0
        self.on_ground = False
        self.score = 0
        self.lives = 3
        self.level = 1
        self.direction = 1  # 1 for right, -1 for left
        self.fireballs = pygame.sprite.Group()
        self.last_shot = 0
        self.is_invincible = False
        self.invincibility_timer = 0
        self.invincibility_duration = 0
        self.speed_boost = False
        self.speed_boost_timer = 0
        self.speed_boost_duration = 0
        self.normal_speed = 5
        self.boosted_speed = 8
        self.can_double_jump = True
        self.double_jump_used = False
        self.is_big = False
        self.jump_pressed = False

    def update(self, platforms, coins, enemies, breakable_blocks, powerups, hazards, projectiles):
        keys = pygame.key.get_pressed()
        dx = 0

        # Adjust speed based on power-ups
        speed = self.boosted_speed if self.speed_boost else self.normal_speed

        # Horizontal movement
        if keys[pygame.K_LEFT]:
            dx = -speed
            self.direction = -1
        if keys[pygame.K_RIGHT]:
            dx = speed
            self.direction = 1

        # Apply horizontal movement
        self.rect.x += dx

        # Gravity
        self.velocity_y += GRAVITY
        self.rect.y += self.velocity_y

        # Collision with platforms
        self.on_ground = False
        for platform in platforms:
            if self.rect.colliderect(platform.rect):
                if self.velocity_y > 0:
                    self.rect.bottom = platform.rect.top
                    self.on_ground = True
                    self.velocity_y = 0
                elif self.velocity_y < 0:
                    self.rect.top = platform.rect.bottom
                    self.velocity_y = 0

        # Collision with moving platforms
        for platform in platforms:
            if isinstance(platform, MovingPlatform):
                if self.rect.colliderect(platform.rect):
                    self.rect.x += platform.speed_x
                    self.rect.y += platform.speed_y

        # Handle jumping
        self.handle_jumping(keys)

        # Break blocks when jumping from below
        for block in breakable_blocks:
            if self.rect.colliderect(block.rect):
                if self.velocity_y < 0:  # Moving upward
                    if self.rect.top <= block.rect.bottom:
                        block.break_block(powerups)
                        self.rect.top = block.rect.bottom
                        self.velocity_y = 0

        # Collect power-ups
        powerup_hits = pygame.sprite.spritecollide(self, powerups, True)
        for powerup in powerup_hits:
            self.collect_power_up(powerup)

        # Collect coins
        coin_hits = pygame.sprite.spritecollide(self, coins, True)
        self.score += len(coin_hits) * 10

        # Collision with hazards
        if not self.is_invincible:
            hazard_hits = pygame.sprite.spritecollide(self, hazards, False)
            if hazard_hits:
                self.lives -= 1
                self.reset_position()

        # Collision with enemy projectiles
        if not self.is_invincible:
            projectile_hits = pygame.sprite.spritecollide(self, projectiles, True)
            if projectile_hits:
                self.lives -= 1
                self.reset_position()

        # Collision with enemies
        if not self.is_invincible:
            enemy_hits = pygame.sprite.spritecollide(self, enemies, False)
            if enemy_hits:
                self.lives -= 1
                self.reset_position()

        # Keep player within world bounds
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > WORLD_WIDTH:
            self.rect.right = WORLD_WIDTH

        # Shooting fireballs
        if keys[pygame.K_f]:
            self.shoot_fireball()

        # Update fireballs
        self.fireballs.update(platforms)

        # Check fireball collisions with enemies
        for fireball in self.fireballs:
            enemy_hits = pygame.sprite.spritecollide(fireball, enemies, True)
            if enemy_hits:
                fireball.kill()
                self.score += 50  # Award points for defeating an enemy

        # Handle power-up timers
        self.handle_power_up_timers()

        # Check if player fell off the world
        if self.rect.top > SCREEN_HEIGHT:
            self.lives -= 1
            self.reset_position()

    def handle_jumping(self, keys):
        if keys[pygame.K_SPACE]:
            if self.on_ground:
                self.velocity_y = -15
                self.on_ground = False
                self.double_jump_used = False
                self.jump_pressed = True
            elif self.can_double_jump and not self.double_jump_used and not self.jump_pressed:
                self.velocity_y = -15
                self.double_jump_used = True
                self.jump_pressed = True
        else:
            self.jump_pressed = False

    def handle_power_up_timers(self):
        now = pygame.time.get_ticks()
        if self.is_invincible and now - self.invincibility_timer > self.invincibility_duration:
            self.is_invincible = False
        if self.speed_boost and now - self.speed_boost_timer > self.speed_boost_duration:
            self.speed_boost = False

    def collect_power_up(self, powerup):
        if isinstance(powerup, InvincibilityPowerUp):
            self.is_invincible = True
            self.invincibility_timer = pygame.time.get_ticks()
            self.invincibility_duration = powerup.duration
        elif isinstance(powerup, SpeedBoostPowerUp):
            self.speed_boost = True
            self.speed_boost_timer = pygame.time.get_ticks()
            self.speed_boost_duration = powerup.duration
        else:
            self.grow()

    def grow(self):
        if not self.is_big:
            current_center = self.rect.center
            self.image = load_image(PLAYER_COLOR, PLAYER_WIDTH, int(PLAYER_HEIGHT * 1.5))
            self.rect = self.image.get_rect()
            self.rect.center = current_center
            self.is_big = True

    def reset_position(self):
        self.rect.x = 100
        self.rect.y = SCREEN_HEIGHT - PLAYER_HEIGHT - 40
        self.velocity_y = 0
        self.is_invincible = False
        self.speed_boost = False
        self.is_big = False
        self.image = load_image(PLAYER_COLOR, PLAYER_WIDTH, PLAYER_HEIGHT)
        if self.lives <= 0:
            game_over_screen()

    def shoot_fireball(self):
        # Limit fireball shooting rate
        now = pygame.time.get_ticks()
        if now - self.last_shot < 500:  # 500 ms cooldown
            return
        self.last_shot = now
        fireball = Fireball(self.rect.centerx, self.rect.centery, self.direction)
        self.fireballs.add(fireball)

    def save_state(self):
        return {
            'score': self.score,
            'lives': self.lives,
            'level': self.level,
            'position': (self.rect.x, self.rect.y),
            'is_big': self.is_big
        }

    def load_state(self, state):
        self.score = state['score']
        self.lives = state['lives']
        self.level = state['level']
        self.rect.x, self.rect.y = state['position']
        self.is_big = state['is_big']
        if self.is_big:
            self.grow()

class Fireball(pygame.sprite.Sprite):
    def __init__(self, x, y, direction):
        super().__init__()
        self.image = load_image((255, 165, 0), 20, 20)  # Orange color for fireball
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.direction = direction  # 1 for right, -1 for left
        self.velocity_x = 8 * self.direction
        self.velocity_y = -5  # Initial upward velocity
        self.bounces = 0
        self.max_bounces = 3  # Fireball disappears after 3 bounces

    def update(self, platforms):
        # Apply gravity
        self.velocity_y += GRAVITY
        self.rect.x += self.velocity_x
        self.rect.y += self.velocity_y

        # Bounce off platforms
        for platform in platforms:
            if self.rect.colliderect(platform.rect):
                if self.velocity_y > 0:
                    self.rect.bottom = platform.rect.top
                    self.velocity_y = -10  # Bounce up
                    self.bounces += 1
                    if self.bounces >= self.max_bounces:
                        self.kill()  # Remove fireball after max bounces

        # Remove fireball if it goes off-screen
        if self.rect.right < 0 or self.rect.left > WORLD_WIDTH:
            self.kill()

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = load_image(PLATFORM_COLOR, width, height)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class MovingPlatform(Platform):
    def __init__(self, x, y, width, height, path, speed):
        super().__init__(x, y, width, height)
        self.path = path  # List of points to move between
        self.speed = speed
        self.current_target = 0
        self.speed_x = 0
        self.speed_y = 0

    def update(self):
        target_x, target_y = self.path[self.current_target]
        dx = target_x - self.rect.x
        dy = target_y - self.rect.y
        distance = (dx**2 + dy**2) ** 0.5
        if distance < self.speed:
            self.rect.x = target_x
            self.rect.y = target_y
            self.current_target = (self.current_target + 1) % len(self.path)
            self.speed_x = 0
            self.speed_y = 0
        else:
            self.speed_x = self.speed * dx / distance
            self.speed_y = self.speed * dy / distance
            self.rect.x += self.speed_x
            self.rect.y += self.speed_y

class BreakableBlock(pygame.sprite.Sprite):
    def __init__(self, x, y, contains_powerup=None):
        super().__init__()
        self.image = load_image((139, 69, 19), 40, 40)  # Brown color for block
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.contains_powerup = contains_powerup

    def break_block(self, powerups_group):
        if self.contains_powerup:
            # Release a power-up
            if self.contains_powerup == 'grow':
                powerup = PowerUp(self.rect.centerx, self.rect.top)
            elif self.contains_powerup == 'invincibility':
                powerup = InvincibilityPowerUp(self.rect.centerx, self.rect.top)
            elif self.contains_powerup == 'speed':
                powerup = SpeedBoostPowerUp(self.rect.centerx, self.rect.top)
            else:
                powerup = PowerUp(self.rect.centerx, self.rect.top)
            powerups_group.add(powerup)
        self.kill()  # Remove the block

class PowerUp(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = load_image(POWERUP_COLOR, 30, 30)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.velocity_y = -5  # Initial upward movement

    def update(self, platforms):
        # Apply gravity
        self.velocity_y += GRAVITY
        self.rect.y += self.velocity_y

        # Stop at platforms
        for platform in platforms:
            if self.rect.colliderect(platform.rect):
                if self.velocity_y > 0:
                    self.rect.bottom = platform.rect.top
                    self.velocity_y = 0

class InvincibilityPowerUp(PowerUp):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.image.fill(INVINCIBILITY_COLOR)
        self.duration = 5000  # Duration in milliseconds

class SpeedBoostPowerUp(PowerUp):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.image.fill(SPEED_BOOST_COLOR)
        self.duration = 5000  # Duration in milliseconds

class Coin(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = load_image(COIN_COLOR, 20, 20)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, left_bound, right_bound):
        super().__init__()
        self.image = load_image(ENEMY_COLOR, 40, 40)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.speed = 2

    def update(self):
        self.rect.x += self.speed
        if self.rect.right > self.right_bound or self.rect.left < self.left_bound:
            self.speed *= -1

class FlyingEnemy(Enemy):
    def __init__(self, x, y, left_bound, right_bound, top_bound, bottom_bound):
        super().__init__(x, y, left_bound, right_bound)
        self.top_bound = top_bound
        self.bottom_bound = bottom_bound
        self.vertical_speed = 2

    def update(self):
        super().update()
        self.rect.y += self.vertical_speed
        if self.rect.top < self.top_bound or self.rect.bottom > self.bottom_bound:
            self.vertical_speed *= -1

class TurretEnemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = load_image(TURRET_COLOR, 40, 60)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.last_shot_time = pygame.time.get_ticks()
        self.shot_interval = 2000  # Shoot every 2 seconds

    def update(self, projectiles_group):
        now = pygame.time.get_ticks()
        if now - self.last_shot_time > self.shot_interval:
            self.shoot(projectiles_group)
            self.last_shot_time = now

    def shoot(self, projectiles_group):
        projectile = EnemyProjectile(self.rect.centerx, self.rect.bottom)
        projectiles_group.add(projectile)

class EnemyProjectile(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = load_image(PROJECTILE_COLOR, 10, 10)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.y = y
        self.speed = 5

    def update(self):
        self.rect.y += self.speed
        if self.rect.top > SCREEN_HEIGHT:
            self.kill()

class Hazard(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = load_image(HAZARD_COLOR, width, height)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Camera:
    def __init__(self, player):
        self.offset_x = 0
        self.player = player
        self.camera_speed = 0.1  # Adjust for smoothness

    def update(self):
        target_x = self.player.rect.centerx - SCREEN_WIDTH // 2
        # Smoothly interpolate towards the target offset
        self.offset_x += (target_x - self.offset_x) * self.camera_speed
        # Limit scrolling to world boundaries
        self.offset_x = max(0, min(self.offset_x, WORLD_WIDTH - SCREEN_WIDTH))

class Flagpole(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = load_image((255, 255, 255), 40, 200)  # White flagpole
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Level:
    def __init__(self, level_num):
        self.platforms = pygame.sprite.Group()
        self.coins = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.breakable_blocks = pygame.sprite.Group()
        self.powerups = pygame.sprite.Group()
        self.hazards = pygame.sprite.Group()
        self.moving_platforms = pygame.sprite.Group()
        self.turrets = pygame.sprite.Group()
        self.enemy_projectiles = pygame.sprite.Group()
        self.flagpoles = pygame.sprite.Group()
        self.load_level(level_num)

    def load_level(self, level_num):
        # Clear existing sprites
        self.platforms.empty()
        self.coins.empty()
        self.enemies.empty()
        self.breakable_blocks.empty()
        self.powerups.empty()
        self.hazards.empty()
        self.moving_platforms.empty()
        self.turrets.empty()
        self.enemy_projectiles.empty()

        # Load level data
        if level_num == 1:
            self.load_level_one()
        elif level_num == 2:
            self.load_level_two()
        else:
            # No more levels
            victory_screen()

    def load_level_one(self):
        # Platforms
        platform_list = [
            Platform(0, SCREEN_HEIGHT - 40, WORLD_WIDTH, 40),  # Ground
            Platform(300, 500, 200, 20),
            Platform(600, 400, 200, 20),
            Platform(900, 300, 200, 20),
            Platform(1200, 200, 200, 20),
            Platform(1500, 500, 200, 20),
            Platform(1800, 400, 200, 20),
            Platform(2100, 300, 200, 20),
            Platform(2400, 200, 200, 20),
            Platform(2700, 500, 200, 20),
        ]
        self.platforms.add(platform_list)

        # Coins
        coin_positions = [
            (350, 450), (650, 350), (950, 250), (1250, 150),
            (1550, 450), (1850, 350), (2150, 250), (2450, 150),
            (2750, 450)
        ]
        for pos in coin_positions:
            coin = Coin(*pos)
            self.coins.add(coin)

        # Enemies
        enemy_list = [
            Enemy(500, SCREEN_HEIGHT - 80, 500, 700),
            Enemy(1600, SCREEN_HEIGHT - 80, 1600, 1800),
            Enemy(2300, SCREEN_HEIGHT - 80, 2300, 2500),
        ]
        self.enemies.add(enemy_list)

        # Breakable Blocks
        block_positions = [
            (500, 300), (800, 250), (1100, 200),
            (1400, 150, 'grow'),  # Contains power-up
            (1700, 300), (2000, 250, 'invincibility'),  # Contains power-up
        ]
        for pos in block_positions:
            if len(pos) == 3:
                block = BreakableBlock(pos[0], pos[1], pos[2])
            else:
                block = BreakableBlock(pos[0], pos[1])
            self.breakable_blocks.add(block)

        # Hazards
        hazard = Hazard(1300, SCREEN_HEIGHT - 40 - 40, 100, 40)
        self.hazards.add(hazard)

        # Turrets
        turret = TurretEnemy(1900, SCREEN_HEIGHT - 40 - 60)
        self.turrets.add(turret)

        flagpole = Flagpole(WORLD_WIDTH - 100, SCREEN_HEIGHT - 240)
        self.flagpoles.add(flagpole)

    def load_level_two(self):
        # Similar setup for level two with different positions and elements
        # ...

        # For simplicity, let's just reload level one
        self.load_level_one()

def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

def update(player, level, camera):
    player.update(
        level.platforms,
        level.coins,
        level.enemies,
        level.breakable_blocks,
        level.powerups,
        level.hazards,
        level.enemy_projectiles
    )
    level.enemies.update()
    level.powerups.update(level.platforms)
    level.hazards.update()
    level.moving_platforms.update()
    for turret in level.turrets:
        turret.update(level.enemy_projectiles)
    level.enemy_projectiles.update()
    camera.update()
    check_level_completion(player, level)

def render(screen, player, level, camera):
    screen.fill((135, 206, 235))  # Sky blue background
    offset_x = camera.offset_x

    # Draw platforms
    for platform in level.platforms:
        screen.blit(platform.image, (platform.rect.x - offset_x, platform.rect.y))

    # Draw moving platforms
    for platform in level.moving_platforms:
        screen.blit(platform.image, (platform.rect.x - offset_x, platform.rect.y))

    # Draw breakable blocks
    for block in level.breakable_blocks:
        screen.blit(block.image, (block.rect.x - offset_x, block.rect.y))

    # Draw coins
    for coin in level.coins:
        screen.blit(coin.image, (coin.rect.x - offset_x, coin.rect.y))

    # Draw enemies
    for enemy in level.enemies:
        screen.blit(enemy.image, (enemy.rect.x - offset_x, enemy.rect.y))

    # Draw turrets
    for turret in level.turrets:
        screen.blit(turret.image, (turret.rect.x - offset_x, turret.rect.y))

    # Draw enemy projectiles
    for projectile in level.enemy_projectiles:
        screen.blit(projectile.image, (projectile.rect.x - offset_x, projectile.rect.y))

    # Draw power-ups
    for powerup in level.powerups:
        screen.blit(powerup.image, (powerup.rect.x - offset_x, powerup.rect.y))

    # Draw hazards
    for hazard in level.hazards:
        screen.blit(hazard.image, (hazard.rect.x - offset_x, hazard.rect.y))

    # Draw fireballs
    for fireball in player.fireballs:
        screen.blit(fireball.image, (fireball.rect.x - offset_x, fireball.rect.y))

    # Draw flagpoles
    for flagpole in level.flagpoles:
        screen.blit(flagpole.image, (flagpole.rect.x - offset_x, flagpole.rect.y))

    # Draw player
    screen.blit(player.image, (player.rect.x - offset_x, player.rect.y))

    # Draw HUD
    render_hud(screen, player)

    pygame.display.flip()

def render_hud(screen, player):
    score_text = font.render(f"Score: {player.score}", True, (0, 0, 0))
    lives_text = font.render(f"Lives: {player.lives}", True, (0, 0, 0))
    level_text = font.render(f"Level: {player.level}", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))
    screen.blit(lives_text, (10, 40))
    screen.blit(level_text, (10, 70))

def check_level_completion(player, level):
    # Check if the player has touched the flagpole
    if pygame.sprite.spritecollide(player, level.flagpoles, False):
        player.level += 1
        load_new_level(player, level)


def load_new_level(player, level):
    # Load next level data
    level.load_level(player.level)
    player.rect.x = 100
    player.rect.y = SCREEN_HEIGHT - PLAYER_HEIGHT - 40

def save_game(player):
    save_data = player.save_state()
    with open('savegame.dat', 'wb') as f:
        pickle.dump(save_data, f)

def load_game(player):
    try:
        with open('savegame.dat', 'rb') as f:
            save_data = pickle.load(f)
            player.load_state(save_data)
    except FileNotFoundError:
        pass

def main_menu():
    while True:
        screen.fill((0, 0, 0))
        title_text = font.render("Super Mario Clone", True, (255, 255, 255))
        start_text = font.render("1. Start Game", True, (255, 255, 255))
        load_text = font.render("2. Load Game", True, (255, 255, 255))
        quit_text = font.render("3. Quit", True, (255, 255, 255))
        screen.blit(title_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 100))
        screen.blit(start_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 - 50))
        screen.blit(load_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2))
        screen.blit(quit_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 + 50))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return 'new'  # Start new game
                if event.key == pygame.K_2:
                    return 'load'  # Load game
                if event.key == pygame.K_3:
                    pygame.quit()
                    sys.exit()

def pause_menu():
    paused = True
    while paused:
        screen.fill((0, 0, 0))
        pause_text = font.render("Game Paused", True, (255, 255, 255))
        resume_text = font.render("Press P to Resume", True, (255, 255, 255))
        screen.blit(pause_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 - 50))
        screen.blit(resume_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2))
        pygame.display.flip()
        clock.tick(15)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    wait_for_key_release(pygame.K_p)  # Wait for P key to be released
                    paused = False

def wait_for_key_release(key):
    """Waits until the specified key is released before proceeding."""
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYUP and event.key == key:
                waiting = False

def game_over_screen():
    screen.fill((0, 0, 0))  # Black background
    game_over_text = font.render("GAME OVER", True, (255, 0, 0))
    restart_text = font.render("Press R to Restart", True, (255, 255, 255))
    screen.blit(game_over_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 50))
    screen.blit(restart_text, (SCREEN_WIDTH//2 - 120, SCREEN_HEIGHT//2))
    pygame.display.flip()
    wait_for_restart()

def victory_screen():
    screen.fill((0, 0, 0))
    victory_text = font.render("You Win!", True, (255, 255, 0))
    restart_text = font.render("Press R to Play Again", True, (255, 255, 255))
    screen.blit(victory_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 - 50))
    screen.blit(restart_text, (SCREEN_WIDTH//2 - 120, SCREEN_HEIGHT//2))
    pygame.display.flip()
    wait_for_restart()

def wait_for_restart():
    waiting = True
    while waiting:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main()  # Restart the game

def main():
    choice = main_menu()
    player = Player()
    camera = Camera(player)
    level = Level(player.level)

    if choice == 'load':
        load_game(player)
        level.load_level(player.level)

    while True:
        handle_events()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_p]:
            pause_menu()
        if keys[pygame.K_s]:
            save_game(player)
        update(player, level, camera)
        render(screen, player, level, camera)
        clock.tick(60)

if __name__ == "__main__":
    main()
