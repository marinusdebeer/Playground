import pygame
import sys
import os
import pickle
import time
import random

# Configuration Flag
USE_IMAGES = True  # Set to False to use colored rectangles instead of images

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GROUND_HEIGHT = 80

# Configuration for block platforms
BLOCK_WIDTH = 40  # Block width in pixels
BLOCK_HEIGHT = 40  # Block height in pixels
POWERUP_CHANCE = 0.05  # 5% chance for a power-up or coin to be hidden in a block

# List of possible power-ups
powerup_types = ['grow', 'fireball_powerup', 'invincibility_powerup', 'speed', 'coin']


fullscreen = False

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

MUTE = False
TOTAL_TIME = 82

# Initialize screen and clock
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Super Mario Clone")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

def load_image(path, width, height):
    """
    Load an image from the specified path and scale it to the given width and height.
    The image will be stretched to exactly fill the area, ensuring no gaps in width or height.
    If the image fails to load or USE_IMAGES is False, return a colored rectangle.
    """
    if USE_IMAGES and path is not None:
        try:
            # Load the image
            image = pygame.image.load(path).convert_alpha()
            
            # Scale the image to fill the exact width and height
            image = pygame.transform.scale(image, (width, height))
            
            return image
        except pygame.error as e:
            print(f"Error loading image at {path}: {e}")
            # Fallback to a colored rectangle if image loading fails
            image = pygame.Surface((width, height))
            image.fill((255, 0, 255))  # Magenta indicates missing image
            return image
    else:
        # Return a colored rectangle
        image = pygame.Surface((width, height))
        image.fill(PLAYER_COLOR)  # Default color for platforms if no image
        return image

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        
        # Base directory for images
        base_dir = os.path.join('assets', 'images')
        self.fireball_sound = pygame.mixer.Sound(os.path.join('assets', 'sounds', 'Super Mario Bros.-Fireball Sound Effect.mp3'))
        self.powerup_sound_effect = pygame.mixer.Sound(os.path.join('assets', 'sounds', 'PowerUp Sound Effect.mp3'))
        self.coin_sound_effect = pygame.mixer.Sound(os.path.join('assets', 'sounds', 'Mario Coin Sound - Sound Effect.mp3'))
        self.fireball_sound.set_volume(0.8)
        self.powerup_sound_effect.set_volume(0.8)
        self.coin_sound_effect.set_volume(0.1)

        pygame.mixer.music.load('assets/sounds/Super Mario Bros. Theme Song.mp3')
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play(1)

        # Load player images or use rectangles
        if USE_IMAGES:
            # Load idle image
            self.image_idle_right = load_image(os.path.join(base_dir, 'mario_idle.png'), PLAYER_WIDTH, PLAYER_HEIGHT)
            self.image_idle_left = pygame.transform.flip(self.image_idle_right, True, False)

            # Load running animation frames
            self.run_frames_right = [
                load_image(os.path.join(base_dir, 'mario_running1.png'), PLAYER_WIDTH, PLAYER_HEIGHT),
                load_image(os.path.join(base_dir, 'mario_running2.png'), PLAYER_WIDTH, PLAYER_HEIGHT),
                load_image(os.path.join(base_dir, 'mario_running1.png'), PLAYER_WIDTH, PLAYER_HEIGHT),
                load_image(os.path.join(base_dir, 'mario_running2.png'), PLAYER_WIDTH, PLAYER_HEIGHT)
            ]
            # Create flipped frames for running left
            self.run_frames_left = [pygame.transform.flip(frame, True, False) for frame in self.run_frames_right]
            
            # Load jump images
            self.image_jump_right = load_image(os.path.join(base_dir, 'mario_jump.png'), PLAYER_WIDTH, PLAYER_HEIGHT)
            self.image_jump_left = pygame.transform.flip(self.image_jump_right, True, False)
            
            # Set initial image
            self.image = self.image_idle_right
        else:
            # Use colored rectangle
            self.image = load_image(None, PLAYER_WIDTH, PLAYER_HEIGHT)
        
        self.rect = self.image.get_rect()
        self.rect.x = 100
        self.rect.y = SCREEN_HEIGHT - PLAYER_HEIGHT - 80
        self.velocity_x = 0
        self.velocity_y = 0
        self.on_ground = False
        self.score = 0
        self.lives = 3
        self.level = 1
         # Crouch state
        self.is_crouched = False
        self.crouch_pressed = False
        self.direction = 1  # 1 for right, -1 for left
        self.fireballs = pygame.sprite.Group()
        self.last_shot = 0
        self.is_invincible = False
        self.invincibility_timer = 0
        self.invincibility_duration = 0
        self.speed_boost = False
        self.speed_boost_timer = 0
        self.speed_boost_duration = 0
        self.is_big = False
        self.scale_factor = 1.0
        self.height_scaling = 1.0
        self.fireball_ability = False  # Flag to enable fireball shooting
        
        # Flags for spacebar and jumping
        self.space_pressed = False  # Tracks if spacebar is held down

        # Movement attributes
        self.acceleration = 0.5
        self.deceleration = 0.2
        self.max_speed = 7

        # Jumping attributes
        self.jump_velocity = -15  # Velocity applied when jumping
        self.double_jump_velocity = -15  # Velocity applied when double jumping
        self.double_jump_used = False  # Tracks if double jump has been used
        self.jump_increase_factor = 1.0  # Factor to increase jump height

        self.prev_rect = self.rect.copy()  # To store previous position

        # Animation attributes
        if USE_IMAGES:
            self.current_run_frame = 0
            self.animation_timer = 0
            self.animation_speed = 100  # milliseconds between frames
    
    def set_image(self, new_image):
        """Set the player's image and adjust the rect while maintaining the center."""
        center = self.rect.center
        self.image = pygame.transform.scale(new_image, (int(PLAYER_WIDTH * self.scale_factor), int(PLAYER_HEIGHT * self.scale_factor * self.height_scaling)))
        self.rect = self.image.get_rect()
        self.rect.center = center
    
    def handle_event(self, event):
        """Handle individual events for the player."""
        if event.type == pygame.KEYDOWN and self.fireball_ability:
            if event.key == pygame.K_f:
                self.shoot_fireball()  # Fireball shooting logic when 'f' key is pressed

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.space_pressed = True
            elif event.key == pygame.K_DOWN:
                self.crouch_pressed = True  # Set crouch flag when down arrow is pressed

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                if self.on_ground:
                    self.jump()  # Perform jump when spacebar is released and on the ground
                elif not self.on_ground and not self.double_jump_used:
                    self.double_jump()  # Perform double jump
                self.space_pressed = False  # Stop braking or any spacebar related logic
            elif event.key == pygame.K_DOWN:
                self.crouch_pressed = False  # Unset crouch flag when down arrow is released

    def jump(self):
        """Handle jumping when spacebar is pressed."""
        self.velocity_y = self.jump_velocity * self.jump_increase_factor
        self.on_ground = False
        self.double_jump_used = False  # Reset double jump when jumping from ground

    def double_jump(self):
        """Handle double jumping when spacebar is pressed in the air."""
        self.velocity_y = self.double_jump_velocity * self.jump_increase_factor
        self.double_jump_used = True
    
    def update(self, platforms, coins, enemies, breakable_blocks, powerups, hazards, projectiles, moving_platforms):
        """Updates the player's state, handling movement, collisions, and interactions."""
        keys = pygame.key.get_pressed()

        self.prev_rect = self.rect.copy()

        # --- Horizontal Movement ---
        moving_left = keys[pygame.K_LEFT]
        moving_right = keys[pygame.K_RIGHT]

        if moving_left and not self.space_pressed:
            self.direction = -1
            self.velocity_x -= self.acceleration
            if USE_IMAGES:
                self.animate_running(left=True)
        elif moving_right and not self.space_pressed:
            self.direction = 1
            self.velocity_x += self.acceleration
            if USE_IMAGES:
                self.animate_running(left=False)
        else:
            # Apply deceleration when no movement key is pressed
            if self.velocity_x > 0:
                self.velocity_x -= self.deceleration
                if self.velocity_x < 0:
                    self.velocity_x = 0
            elif self.velocity_x < 0:
                self.velocity_x += self.deceleration
                if self.velocity_x > 0:
                    self.velocity_x = 0

        # Limit the player's horizontal speed
        if keys[pygame.K_LSHIFT]:
            self.max_speed = 14
        else:
            self.max_speed = 7
        self.velocity_x = max(-self.max_speed, min(self.velocity_x, self.max_speed))

        # Move horizontally
        self.rect.x += self.velocity_x

        # --- Horizontal Collision Detection ---
        resolve_collisions(self, platforms, 'horizontal')
        resolve_collisions(self, moving_platforms, 'horizontal')

        # --- Vertical Movement ---
        # Apply gravity
        self.velocity_y += GRAVITY
        if self.velocity_y > 10:  # Terminal velocity
            self.velocity_y = 10

        # Move vertically
        self.rect.y += self.velocity_y

        # --- Vertical Collision Detection ---
        self.on_ground = False  # Reset on_ground status before checking

        # Handle collision with breakable blocks
        resolve_block_collision(self, breakable_blocks, powerups)

        resolve_collisions(self, platforms, 'vertical')
        resolve_collisions(self, moving_platforms, 'vertical')

        # --- Moving with the Platform ---
        if self.on_ground:
            for platform in moving_platforms:
                # Check if the player is standing on this moving platform
                if self.rect.bottom == platform.rect.top and platform.rect.left <= self.rect.centerx <= platform.rect.right:
                    self.rect.x += platform.speed_x
                    self.rect.y += platform.speed_y

        # --- Additional Collision Handling ---
        # Break blocks when jumping from below
        for block in breakable_blocks:
            if self.rect.colliderect(block.rect):
                if self.velocity_y < 0:  # Moving upward
                    if self.rect.top <= block.rect.bottom:
                        block.break_block(powerups, self.is_big)
                        self.rect.top = block.rect.bottom
                        self.velocity_y = 0

        # Collect power-ups
        powerup_hits = pygame.sprite.spritecollide(self, powerups, True)
        for powerup in powerup_hits:
            self.collect_power_up(powerup)

        # Collect coins
        coin_hits = pygame.sprite.spritecollide(self, coins, True)
        for coin in coin_hits:
            self.coin_sound_effect.play()
            self.score += 10

        # Collision with hazards
        if not self.is_invincible:
            hazard_hits = pygame.sprite.spritecollide(self, hazards, False)
            if hazard_hits:
                if self.is_big:
                    self.is_big = False
                    self.jump_increase_factor = 1.0
                    self.scale_factor = 1.0
                    self.fireball_ability = False
                else:
                    self.lives -= 1
                    self.reset_position()

        # Collision with enemy projectiles
        if not self.is_invincible:
            projectile_hits = pygame.sprite.spritecollide(self, projectiles, True)
            if projectile_hits:
                if self.is_big:
                    self.is_big = False
                    self.jump_increase_factor = 1.0
                    self.scale_factor = 1.0
                    self.fireball_ability = False
                else:
                    self.lives -= 1
                    self.reset_position()

        # --- Enemy Collision (Defeat when jumping on top) ---
        for enemy in enemies:
            if self.rect.colliderect(enemy.rect):
                if self.velocity_y > 0 and self.rect.bottom <= enemy.rect.top + 10:  # Only when falling down
                    self.rect.bottom = enemy.rect.top  # Adjust the player's position to be on top of the enemy
                    self.velocity_y = -10  # Bounce up a little after defeating the enemy
                    enemy.kill()  # Remove the enemy from the game
                    self.score += 100  # Award points for defeating an enemy
                else:
                    if not self.is_invincible:
                        # Handle collision if player hits enemy from the side or bottom
                        if self.is_big:
                            self.is_big = False
                            self.jump_increase_factor = 1.0
                            self.scale_factor = 1.0
                            self.fireball_ability = False
                        else:
                            self.lives -= 1
                            self.reset_position()


        # --- Keep Player Within World Bounds ---
        if self.rect.left < 0:
            self.rect.left = 0
            self.velocity_x = 0
        if self.rect.right > WORLD_WIDTH:
            self.rect.right = WORLD_WIDTH
            self.velocity_x = 0

        # --- Update Fireballs ---
        self.fireballs.update(platforms)

        # Check fireball collisions with enemies
        for fireball in self.fireballs:
            enemy_hits = pygame.sprite.spritecollide(fireball, enemies, True)
            if enemy_hits:
                fireball.kill()
                self.score += 50  # Award points for defeating an enemy

        # --- Handle Power-Up Timers ---
        self.handle_power_up_timers()

        # --- Check if Player Fell Off the World ---
        if self.rect.top > SCREEN_HEIGHT:
            self.lives -= 1
            self.reset_position()

        # --- Update Player Image Based on State ---
        if USE_IMAGES:
            if self.space_pressed and self.on_ground:
                # Show jump image only when on the ground and spacebar is pressed
                if self.direction == 1:
                    self.set_image(self.image_jump_right)
                else:
                    self.set_image(self.image_jump_left)
            elif not self.on_ground:
                # Show in-air image when the player is in the air
                if self.direction == 1:
                    self.set_image(self.run_frames_right[1])
                else:
                    self.set_image(self.run_frames_left[1])
            elif moving_left or moving_right:
                # Running animation already handled in animate_running
                pass
            else:
                # Use idle image when standing still
                if self.direction == 1:
                    self.set_image(self.image_idle_right)
                else:
                    self.set_image(self.image_idle_left)

        # Handle crouch mechanic
        if self.space_pressed and not self.is_crouched and self.on_ground:
            self.crouch()
        elif not self.space_pressed and self.is_crouched:
            self.stand_up()

    def crouch(self):
        """Crouch by reducing the player's height by half."""
        self.is_crouched = True
        self.height_scaling = 0.5  # Half the size
        old_bottom = self.rect.bottom  # Preserve the bottom position

        # Update the image based on direction
        if self.direction == 1:
            self.set_image(self.image_idle_right)
        else:
            self.set_image(self.image_idle_left)

        # Adjust the rect to maintain the bottom position
        self.rect.bottom = old_bottom

    def stand_up(self):
        """Stand up by restoring the player's original height."""
        self.is_crouched = False
        self.height_scaling = 1.0  # Original size
        old_bottom = self.rect.bottom  # Preserve the bottom position

        # Update the image based on direction
        if self.direction == 1:
            self.set_image(self.image_idle_right)
        else:
            self.set_image(self.image_idle_left)

        # Adjust the rect to maintain the bottom position
        self.rect.bottom = old_bottom

    def animate_running(self, left=False):
        """Animate the running sequence."""
        if not USE_IMAGES:
            return  # No animation if not using images

        now = pygame.time.get_ticks()
        if now - self.animation_timer > self.animation_speed:
            self.animation_timer = now
            self.current_run_frame = (self.current_run_frame + 1) % len(self.run_frames_right)
            if left:
                self.set_image(self.run_frames_left[self.current_run_frame])
            else:
                self.set_image(self.run_frames_right[self.current_run_frame])
    
    def handle_power_up_timers(self):
        now = pygame.time.get_ticks()
        if self.is_invincible and now - self.invincibility_timer > self.invincibility_duration:
            self.is_invincible = False
        if self.speed_boost and now - self.speed_boost_timer > self.speed_boost_duration:
            self.speed_boost = False
            self.max_speed = 7  # Reset speed to normal
    
    def collect_power_up(self, powerup):
        self.powerup_sound_effect.play()
        if isinstance(powerup, InvincibilityPowerUp):
            self.is_invincible = True
            self.invincibility_timer = pygame.time.get_ticks()
            self.invincibility_duration = powerup.duration
        elif isinstance(powerup, SpeedBoostPowerUp):
            self.speed_boost = True
            self.max_speed = 14  # Increase speed
            self.speed_boost_timer = pygame.time.get_ticks()
            self.speed_boost_duration = powerup.duration
        elif isinstance(powerup, FireballPowerUp):
            self.fireball_ability = True
        elif powerup.type == 'grow':
            self.grow()
    
    def grow(self):
        if not self.is_big:
            self.jump_increase_factor = 1.5
            self.scale_factor = 1.5
            current_center = self.rect.center
            if USE_IMAGES:
                # Assuming 'idle_big.png' is the image for the grown player
                self.image = load_image(os.path.join('assets', 'images', 'mario_idle.png'), PLAYER_WIDTH, int(PLAYER_HEIGHT))
            else:
                self.image = load_image(None, PLAYER_WIDTH, int(PLAYER_HEIGHT))
                self.image.fill((255, 0, 0))  # Maintain red color or change as desired
            self.rect = self.image.get_rect()
            self.rect.center = current_center
            self.is_big = True
        
    def reset_position(self):
        self.rect.x = 100
        self.rect.y = SCREEN_HEIGHT - PLAYER_HEIGHT - GROUND_HEIGHT
        self.velocity_x = 0
        self.velocity_y = 0
        self.is_invincible = False
        self.speed_boost = False
        self.max_speed = 7
        self.is_big = False
        self.double_jump_used = False
        self.direction = 1
        self.jump_velocity = -15
        self.scale_factor = 1.0
        self.fireball_ability = False
        self.jump_increase_factor = 1.0

        if USE_IMAGES:
            self.set_image(self.image_idle_right)
        else:
            self.image = load_image(None, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.fireballs.empty()
        if self.lives <= 0:
            game_over_screen()  # Ensure this function is defined elsewhere

        if USE_IMAGES:
            self.set_image(self.image_idle_right)
        else:
            self.image = load_image(None, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.fireballs.empty()
        if self.lives <= 0:
            game_over_screen()  # Ensure this function is defined elsewhere
    
    def shoot_fireball(self):
        # Limit fireball shooting rate
        now = pygame.time.get_ticks()
        if now - self.last_shot < 500:  # 500 ms cooldown
            return
        self.last_shot = now
        fireball = Fireball(self.rect.centerx, self.rect.centery, self.direction)
        self.fireballs.add(fireball)
        #play sound effect
        if not MUTE:
            self.fireball_sound.play()
    
    def save_state(self):
        return {
            'score': self.score,
            'lives': self.lives,
            'level': self.level,
            'position': (self.rect.x, self.rect.y),
            'is_big': self.is_big,
            'fireball_ability': self.fireball_ability,
            'is_invincible': self.is_invincible,
            'speed_boost': self.speed_boost
        }
    
    def load_state(self, state):
        self.score = state['score']
        self.lives = state['lives']
        self.level = state['level']
        self.rect.x, self.rect.y = state['position']
        self.is_big = state['is_big']
        self.fireball_ability = state.get('fireball_ability', False)
        self.is_invincible = state.get('is_invincible', False)
        self.speed_boost = state.get('speed_boost', False)
        if self.is_big:
            self.jump_increase_factor = 1.5
            self.scale_factor = 1.5
            self.grow()
        else:
            if USE_IMAGES:
                self.set_image(self.image_idle_right)
            else:
                self.image = load_image(None, PLAYER_WIDTH, PLAYER_HEIGHT)

    def draw(self, screen):
        screen.blit(self.image, self.rect)
        self.fireballs.draw(screen)

class Fireball(pygame.sprite.Sprite):
    def __init__(self, x, y, direction):
        super().__init__()
        if USE_IMAGES:
            # Load fireball image
            base_dir = os.path.join('assets', 'images')
            self.image = load_image(os.path.join(base_dir, 'fireball.png'), 20, 20)

            # If direction is right (1), flip the image horizontally
            if direction == 1:
                self.image = pygame.transform.flip(self.image, True, False)
        else:
            # Use colored rectangle
            self.image = load_image(None, 20, 20)
            self.image.fill((255, 165, 0))  # Orange color for fireball
        
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
                        self.kill()

        # Remove fireball if off-screen
        if self.rect.right < 0 or self.rect.left > WORLD_WIDTH:
            self.kill()

def create_tiled_ground(level, block_image='unbreakable_block.png', block_size=40, rows=2):
    """
    Creates a tiled ground made up of multiple unbreakable blocks.

    Args:
        level (Level): The current level instance to which the ground blocks will be added.
        block_image (str): The image filename for the ground blocks.
        block_size (int): The size (width and height) of each block in pixels.
        rows (int): Number of rows to stack vertically.
    """
    ground_y_positions = [SCREEN_HEIGHT - block_size * (row + 1) for row in range(rows)]
    for y in ground_y_positions:
        for x in range(0, WORLD_WIDTH, block_size):
            block = Platform(x, y, block_size, block_size, image_path=block_image)
            level.platforms.add(block)

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, image_path='platform.png', border_color=(0, 0, 0), border_thickness=2):
        super().__init__()
        # Load the specified image or default to 'platform.png'
        self.image = load_image(os.path.join('assets', 'images', image_path), width, height)
        
        # If not using images, fill with PLATFORM_COLOR
        if not USE_IMAGES:
            self.image.fill(PLATFORM_COLOR)
        
        self.rect = self.image.get_rect(topleft=(x, y))


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

def resolve_collisions(player, sprite_group, direction):
    """
    Resolves collisions between the player and sprites in the provided sprite_group.

    Args:
        player (Player): The player object.
        sprite_group (pygame.sprite.Group): Group of sprites to check collisions against.
        direction (str): Direction of movement ('horizontal' or 'vertical').

    Returns:
        None
    """
    for sprite in sprite_group:
        if player.rect.colliderect(sprite.rect):
            if direction == 'horizontal':
                if player.velocity_x > 0:  # Moving right; hit left side of sprite
                    player.rect.right = sprite.rect.left
                elif player.velocity_x < 0:  # Moving left; hit right side of sprite
                    player.rect.left = sprite.rect.right
                player.velocity_x = 0
            elif direction == 'vertical':
                if player.velocity_y > 0:  # Falling down; hit top of sprite
                    player.rect.bottom = sprite.rect.top
                    player.on_ground = True
                    player.velocity_y = 0
                    player.double_jump_used = False
                elif player.velocity_y < 0:  # Jumping up; hit bottom of sprite
                    player.rect.top = sprite.rect.bottom
                    player.velocity_y = 0
def resolve_block_collision(player, block_group, powerups_group):
    """Handle collision detection for breakable blocks."""
    for block in block_group:
        if player.rect.colliderect(block.rect):
            # Determine the direction of collision based on previous position
            if player.prev_rect.bottom <= block.rect.top and player.velocity_y > 0:
                # Player was above the block and is now colliding from the top
                player.rect.bottom = block.rect.top
                player.velocity_y = 0
                player.on_ground = True
            elif player.prev_rect.top >= block.rect.bottom and player.velocity_y < 0:
                # Player was below the block and is now colliding from below
                block.break_block(powerups_group, player.is_big)
                player.rect.top = block.rect.bottom
                player.velocity_y = 0
            elif player.prev_rect.right <= block.rect.left and player.velocity_x > 0:
                # Player was to the left and is now colliding from the left
                player.rect.right = block.rect.left
                player.velocity_x = 0
            elif player.prev_rect.left >= block.rect.right and player.velocity_x < 0:
                # Player was to the right and is now colliding from the right
                player.rect.left = block.rect.right
                player.velocity_x = 0
                    

class BreakableBlock(pygame.sprite.Sprite):
    def __init__(self, x, y, contains_powerup=None):
        super().__init__()
        if USE_IMAGES and contains_powerup:
            base_dir = os.path.join('assets', 'images')
            # Example: Use different images based on the power-up type
            if contains_powerup == 'grow':
                self.image = load_image(os.path.join(base_dir, 'block.png'), 40, 40)
            elif contains_powerup == 'invincibility_powerup':
                self.image = load_image(os.path.join(base_dir, 'block.png'), 40, 40)
            elif contains_powerup == 'speed':
                self.image = load_image(os.path.join(base_dir, 'block.png'), 40, 40)
            else:
                self.image = load_image(os.path.join(base_dir, 'block.png'), 40, 40)
        elif USE_IMAGES:
            self.image = load_image(os.path.join('assets', 'images', 'block.png'), 40, 40)
        else:
            self.image = load_image(None, 40, 40)
            self.image.fill((139, 69, 19))  # Brown color for block
        
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.original_y = y  # Store the original y position for animation
        self.contains_powerup = contains_powerup
        self.breaking = False  # Flag to indicate if the block is breaking
        self.break_timer = 0  # Timer to control the breaking animation
        self.breakable = False

    def break_block(self, powerups_group, breakable=True):
        """Trigger the block breaking animation and release power-up."""
        self.breakable = breakable
        if not self.breaking:
            self.breaking = True
            self.break_timer = 0  # Start the break animation timer

        if self.contains_powerup:
            # Release a power-up
            if self.contains_powerup == 'grow':
                powerup = PowerUp(self.rect.centerx, self.rect.top, 'grow')
            elif self.contains_powerup == 'invincibility_powerup':
                powerup = InvincibilityPowerUp(self.rect.centerx, self.rect.top)
            elif self.contains_powerup == 'speed':
                powerup = SpeedBoostPowerUp(self.rect.centerx, self.rect.top)
            elif self.contains_powerup == 'fireball_powerup':
                powerup = FireballPowerUp(self.rect.centerx, self.rect.top)
            else:
                powerup = PowerUp(self.rect.centerx, self.rect.top, 'generic')
            powerups_group.add(powerup)
            self.contains_powerup = None

    def update(self):
        """Animate the block moving up and breaking."""
        if self.breaking:
            self.break_timer += 1

            # Animate the block moving up and down before breaking
            if self.break_timer <= 10:
                self.rect.y -= 2  # Move up for the first few frames
            elif self.break_timer <= 20:
                self.rect.y += 2  # Move back down to the original position

            if self.break_timer > 20 and self.breakable:
                self.kill()  # Remove the block after the animation completes
            elif self.break_timer > 20:
                self.breaking = False

class PowerUp(pygame.sprite.Sprite):
    def __init__(self, x, y, type='generic'):
        super().__init__()
        self.type = type
        if USE_IMAGES:
            base_dir = os.path.join('assets', 'images')
            if type == 'grow':
                self.image = load_image(os.path.join(base_dir, 'grow_bigger_powerup.png'), 30, 30)
            elif type == 'fireball_powerup':
                self.image = load_image(os.path.join(base_dir, 'fireball_powerup.png'), 30, 30)  # Fireball power-up image
            elif type == 'generic':
                self.image = load_image(os.path.join(base_dir, 'grow_bigger_powerup.png'), 30, 30)
            elif type == 'invincibility_powerup':
                self.image = load_image(os.path.join(base_dir, 'invincibility_powerup.png'), 30, 30)
            else:
                self.image = load_image(os.path.join(base_dir, 'grow_bigger_powerup.png'), 30, 30)
        else:
            # Use colored rectangle
            self.image = pygame.Surface((30, 30))
            self.image.fill(POWERUP_COLOR)
        
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
class FireballPowerUp(PowerUp):
    def __init__(self, x, y):
        super().__init__(x, y, 'fireball_powerup')
        if not USE_IMAGES:
            self.image.fill((255, 69, 0))  # Color for fireball power-up

class InvincibilityPowerUp(PowerUp):
    def __init__(self, x, y):
        super().__init__(x, y, 'invincibility_powerup')
        if not USE_IMAGES:
            self.image.fill(INVINCIBILITY_COLOR)
        self.duration = 5000  # Duration in milliseconds

class SpeedBoostPowerUp(PowerUp):
    def __init__(self, x, y):
        super().__init__(x, y, 'speed')
        if not USE_IMAGES:
            self.image.fill(SPEED_BOOST_COLOR)
        self.duration = 5000  # Duration in milliseconds

class Coin(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        if USE_IMAGES:
            base_dir = os.path.join('assets', 'images')
            self.image = load_image(os.path.join(base_dir, 'coin.png'), 20, 20)
        else:
            self.image = load_image(None, 20, 20)
            self.image.fill(COIN_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, left_bound, right_bound):
        super().__init__()
        if USE_IMAGES:
            base_dir = os.path.join('assets', 'images')
            self.image = load_image(os.path.join(base_dir, 'enemy1.png'), 40, 40)
        else:
            self.image = load_image(None, 40, 40)
            self.image.fill(ENEMY_COLOR)  # Blue color for enemy
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
        if USE_IMAGES:
            base_dir = os.path.join('assets', 'images')
            self.image = load_image(os.path.join(base_dir, 'flying_enemy.png'), 40, 40)
        else:
            self.image.fill((0, 255, 255))  # Cyan color for flying enemy
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
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
        if USE_IMAGES:
            base_dir = os.path.join('assets', 'images')
            self.image = load_image(os.path.join(base_dir, 'mario.png'), 40, 60)
        else:
            self.image = load_image(None, 40, 60)
            self.image.fill(TURRET_COLOR)  # Purple color for turret
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
        if USE_IMAGES:
            base_dir = os.path.join('assets', 'images')
            self.image = load_image(os.path.join(base_dir, 'mario.png'), 10, 10)
        else:
            self.image = load_image(None, 10, 10)
            self.image.fill(PROJECTILE_COLOR)
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
        if USE_IMAGES:
            base_dir = os.path.join('assets', 'images')
            self.image = load_image(os.path.join(base_dir, 'mario.png'), width, height)
        else:
            self.image = load_image(None, width, height)
            self.image.fill(HAZARD_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, destination_level=None):
        super().__init__()
        # Load the pipe image
        self.image = load_image(os.path.join('assets', 'images', 'pipe.png'), width, height)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.destination_level = destination_level  # Level to transport player to
    
    def transport_player(self, player):
        # Check if player is standing on top of the pipe and pressing "Down"
        if player.rect.bottom == self.rect.top and pygame.key.get_pressed()[pygame.K_DOWN]:
            return self.destination_level
        return None

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
        if USE_IMAGES:
            base_dir = os.path.join('assets', 'images')
            self.image = load_image(os.path.join(base_dir, 'flagpole.png'), 40, 200)  # White flagpole
        else:
            self.image = load_image(None, 40, 200)
            self.image.fill((255, 255, 255))  # White color for flagpole
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
        self.pipes = pygame.sprite.Group()
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
        self.flagpoles.empty()
        self.pipes.empty()

        # Load level data
        if level_num == 1:
            self.load_level_one()
        elif level_num == 2:
            self.load_level_two()
        else:
            # No more levels
            victory_screen()

    # Function to randomly assign power-ups to blocks
    def generate_block_platform(self, start_x, start_y, rows, cols):
        block_positions = []
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * BLOCK_WIDTH
                y = start_y + row * BLOCK_HEIGHT

                # Randomly assign a power-up with 5% chance
                if random.random() < POWERUP_CHANCE:
                    powerup = random.choice(powerup_types)  # Randomly choose a power-up type
                    block_positions.append((x, y, powerup))
                else:
                    block_positions.append((x, y))  # Regular block with no power-up

        return block_positions

    def load_level_one(self):
        # Create Tiled Ground - 2 blocks high
        create_tiled_ground(self, block_image='unbreakable_block.png', block_size=40, rows=2)

        # Platforms
        platform_list = [
            Platform(200, 400, 200, 40),
            Platform(600, 400, 200, 40),
            Platform(900, 300, 200, 40),
            Platform(1200, 300, 200, 40),
            Platform(1500, 500, 200, 40),
            Platform(1800, 400, 200, 40),
            Platform(2100, 300, 200, 40),
            Platform(2400, 400, 200, 40),
            Platform(2700, 500, 200, 40),
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
            Enemy(500, SCREEN_HEIGHT - 120, 500, 700),
            Enemy(1600, SCREEN_HEIGHT - 120, 1600, 1800),
            Enemy(2300, SCREEN_HEIGHT - 120, 2300, 2500),
        ]
        self.enemies.add(enemy_list)

        # Breakable Blocks
        block_positions = [
            (500, 300, 'grow'), (800, 250, 'fireball_powerup'), (1100, 200, 'fireball_powerup'),
            (1400, 150),  # Contains power-up
            (1700, 300), (2000, 250, 'invincibility_powerup'),  # Contains power-up
        ]
        # Generate "platforms" made of breakable blocks with some containing power-ups
        block_positions += self.generate_block_platform(500, 200, 2, 20)  # Two rows, 20 blocks wide
        block_positions += self.generate_block_platform(1500, 400, 2, 20)  # Another platform
        block_positions += self.generate_block_platform(2500, 500, 2, 20)  # Another platform
        for pos in block_positions:
            if len(pos) == 3:
                block = BreakableBlock(pos[0], pos[1], pos[2])
            else:
                block = BreakableBlock(pos[0], pos[1])
            self.breakable_blocks.add(block)

        # Hazards
        # hazard = Hazard(1300, SCREEN_HEIGHT - 40 - 40, 100, 40)
        # self.hazards.add(hazard)

        # Turrets
        # turret = TurretEnemy(1900, SCREEN_HEIGHT - 40 - 60)
        # self.turrets.add(turret)

        pipe = Pipe(800, SCREEN_HEIGHT - 80, 80, 80, destination_level='secret_area')
        self.platforms.add(pipe)  # Adding the pipe to the platforms group so it's rendered
        self.pipes.add(pipe)  # Add to a new pipes group if you want to manage pipes separately

        # Flagpole
        flagpole = Flagpole(WORLD_WIDTH - 100, SCREEN_HEIGHT - 240)
        self.flagpoles.add(flagpole)

    def load_level_two(self):
        
        create_tiled_ground(self, block_image='unbreakable_block.png', block_size=40, rows=2)
        
        platform_list = [
            # Platform(0, SCREEN_HEIGHT - 40, WORLD_WIDTH, 40),  # Ground
            Platform(400, 400, 200, 40),
            Platform(800, 400, 200, 40),
            Platform(1200, 300, 200, 40),
            Platform(1600, 200, 200, 40),
            Platform(2000, 400, 200, 40),
            Platform(2400, 400, 200, 40),
            Platform(2800, 300, 200, 40),
        ]
        self.platforms.add(platform_list)

        # Coins
        coin_positions = [
            (450, 450), (850, 350), (1250, 250), (1650, 150),
            (2050, 450), (2450, 350), (2850, 250)
        ]
        for pos in coin_positions:
            coin = Coin(*pos)
            self.coins.add(coin)

        # Enemies
        enemy_list = [
            FlyingEnemy(700, 300, 700, 900, 200, 400),
            FlyingEnemy(1500, 200, 1500, 1700, 100, 300),
        ]
        self.enemies.add(enemy_list)

        # Breakable Blocks
        block_positions = [
            (700, 350), (1100, 250), (1500, 150, 'speed'),  # Contains power-up
            (1900, 350), (2300, 250, 'invincibility_powerup'),  # Contains power-up
        ]
        for pos in block_positions:
            if len(pos) == 3:
                block = BreakableBlock(pos[0], pos[1], pos[2])
            else:
                block = BreakableBlock(pos[0], pos[1])
            self.breakable_blocks.add(block)

        # Hazards
        hazard = Hazard(1700, SCREEN_HEIGHT - 40 - 40, 100, 40)
        self.hazards.add(hazard)

        # Turrets
        turret = TurretEnemy(2100, SCREEN_HEIGHT - 40 - 60)
        self.turrets.add(turret)

        # Flagpole
        flagpole = Flagpole(WORLD_WIDTH - 100, SCREEN_HEIGHT - 240)
        self.flagpoles.add(flagpole)
    
def load_secret_area(player, level):
    # Clear the current level and load a new secret area level
    level.platforms.empty()
    level.coins.empty()
    level.enemies.empty()
    level.breakable_blocks.empty()
    level.powerups.empty()
    level.hazards.empty()
    level.moving_platforms.empty()
    level.turrets.empty()
    level.enemy_projectiles.empty()
    level.pipes.empty()

    # Define platforms in the secret area
    platforms = [
        Platform(0, SCREEN_HEIGHT - 40, WORLD_WIDTH, 40),  # Ground
        Platform(300, 450, 200, 40),  # Platform 1
        Platform(600, 350, 200, 40),  # Platform 2
        Platform(900, 250, 200, 40),  # Platform 3
        Platform(1200, 150, 200, 40),  # Platform 4
    ]
    level.platforms.add(platforms)

    # Add a ton of coins to make it exciting
    secret_coins = [
        Coin(350, 400), Coin(400, 400), Coin(450, 400), Coin(500, 400),  # Platform 1
        Coin(650, 300), Coin(700, 300), Coin(750, 300), Coin(800, 300),  # Platform 2
        Coin(950, 200), Coin(1000, 200), Coin(1050, 200), Coin(1100, 200),  # Platform 3
        Coin(1250, 100), Coin(1300, 100), Coin(1350, 100), Coin(1400, 100),  # Platform 4
        # Ground coins
        Coin(100, SCREEN_HEIGHT - 80), Coin(150, SCREEN_HEIGHT - 80), Coin(200, SCREEN_HEIGHT - 80),
        Coin(250, SCREEN_HEIGHT - 80), Coin(300, SCREEN_HEIGHT - 80), Coin(350, SCREEN_HEIGHT - 80),
    ]
    level.coins.add(secret_coins)

    # Add a power-up
    powerup = PowerUp(1000, SCREEN_HEIGHT - 120, type='invincibility_powerup')
    level.powerups.add(powerup)

    # Add hazards like spikes or enemies (optional)
    hazard = Hazard(600, SCREEN_HEIGHT - 40 - 40, 100, 40)  # Hazard near the ground
    level.hazards.add(hazard)

    # Add moving platforms for extra excitement
    moving_platforms = [
        MovingPlatform(800, 300, 150, 30, [(800, 300), (1000, 200)], 2),
        MovingPlatform(400, 400, 150, 30, [(400, 400), (600, 300)], 2),
    ]
    level.moving_platforms.add(moving_platforms)

    # Add a pipe that brings the player back to the main level
    return_pipe = Pipe(1500, SCREEN_HEIGHT - 80, 80, 80, destination_level='main_level')
    level.platforms.add(return_pipe)
    level.pipes.add(return_pipe)

    # Set player's position when entering the secret area
    player.rect.x = 100
    player.rect.y = SCREEN_HEIGHT - PLAYER_HEIGHT - 40

def load_main_level(player, level):
    # Clear current level and load the original main level
    level.platforms.empty()
    level.coins.empty()
    level.enemies.empty()
    level.breakable_blocks.empty()
    level.powerups.empty()
    level.hazards.empty()
    level.moving_platforms.empty()
    level.turrets.empty()
    level.enemy_projectiles.empty()
    level.pipes.empty()  # Clear any pipes in the secret area
    
    # Load the main level (assuming it's level 1 here)
    level.load_level_one()

    # Set player's position when they return from the secret area
    player.rect.x = 1500  # Set a new position after returning from the secret area
    player.rect.y = SCREEN_HEIGHT - PLAYER_HEIGHT - 40

def handle_events(player):
    global fullscreen
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        else:
            player.handle_event(event)

def update(player, level, camera):
    
    player.update(
        level.platforms,
        level.coins,
        level.enemies,
        level.breakable_blocks,
        level.powerups,
        level.hazards,
        level.enemy_projectiles,
        level.moving_platforms
    )
    level.breakable_blocks.update()

    # Check if player interacts with pipes
    for pipe in level.pipes:
        destination_level = pipe.transport_player(player)
        if destination_level == 'secret_area':
            load_secret_area(player, level)
        elif destination_level == 'main_level':
            load_main_level(player, level)

    level.enemies.update()
    level.powerups.update(level.platforms)
    level.hazards.update()
    level.moving_platforms.update()
    for turret in level.turrets:
        turret.update(level.enemy_projectiles)
    level.enemy_projectiles.update()
    camera.update()
    check_level_completion(player, level)

def render(screen, player, level, camera, remaining_time):
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
    render_hud(screen, player, remaining_time)

    pygame.display.flip()

def render_hud(screen, player, remaining_time):
    time_text = font.render(f"Time: {remaining_time}", True, (0, 0, 0))
    score_text = font.render(f"Score: {player.score}", True, (0, 0, 0))
    lives_text = font.render(f"Lives: {player.lives}", True, (0, 0, 0))
    level_text = font.render(f"Level: {player.level}", True, (0, 0, 0))
    screen.blit(time_text, (10, 10))
    screen.blit(score_text, (10, 40))
    screen.blit(lives_text, (10, 70))
    screen.blit(level_text, (10, 110))

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
        print("No save game found.")

def instructions_screen():
    showing_instructions = True
    while showing_instructions:
        screen.fill((0, 0, 0))
        title_text = font.render("Instructions", True, (255, 255, 255))
        instructions = [
            "Left Arrow: Move Left",
            "Right Arrow: Move Right",
            "Spacebar: Jump (Double Jump available)",
            "F: Shoot Fireball (when power-up collected)",
            "P: Pause Game",
            "S: Save Game",
            "Press M to return to Main Menu"
        ]
        screen.blit(title_text, (SCREEN_WIDTH // 2 - 80, 50))
        for idx, line in enumerate(instructions):
            instr_text = font.render(line, True, (255, 255, 255))
            screen.blit(instr_text, (50, 100 + idx * 30))
        pygame.display.flip()
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    showing_instructions = False

def main_menu(paused=False):
    while True:
        screen.fill((0, 0, 0))
        title_text = font.render("Super Mario Clone", True, (255, 255, 255))
        
        start_text = font.render("1. Start Game", True, (255, 255, 255))
        load_text = font.render("2. Load Game", True, (255, 255, 255))
        instructions_text = font.render("3. Instructions", True, (255, 255, 255))
        fullscreen_text = font.render("4. Full screen", True, (255, 255, 255))
        quit_text = font.render("5. Quit", True, (255, 255, 255))
        resume_text = font.render("6. Resume", True, (255, 255, 255))
        screen.blit(title_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 150))
        screen.blit(start_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 - 50))
        screen.blit(load_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2))
        screen.blit(instructions_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 + 50))
        screen.blit(fullscreen_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 + 100))
        screen.blit(quit_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 + 150))
        if paused:
            screen.blit(resume_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 + 200))
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
                    instructions_screen()  # Show instructions
                if event.key == pygame.K_4:
                    set_fullscreen()
                if event.key == pygame.K_5:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_6 and paused:
                    return 'resume'

def set_fullscreen():
    global fullscreen
    fullscreen = not fullscreen
    if fullscreen:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

def game_over_screen():
    screen.fill((0, 0, 0))  # Black background
    game_over_text = font.render("GAME OVER", True, (255, 0, 0))
    restart_text = font.render("Press M for Main Menu or Q to Quit", True, (255, 255, 255))
    screen.blit(game_over_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 50))
    screen.blit(restart_text, (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2))
    pygame.display.flip()
    wait_for_restart()

def victory_screen():
    screen.fill((0, 0, 0))
    victory_text = font.render("You Win!", True, (255, 255, 0))
    restart_text = font.render("Press M for Main Menu or Q to Quit", True, (255, 255, 255))
    screen.blit(victory_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 - 50))
    screen.blit(restart_text, (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2))
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
                if event.key == pygame.K_m:
                    waiting = False
                    main()  # Return to main menu
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

def wait_for_key_release(key):
    """Waits until the specified key is released before proceeding."""
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYUP and event.key == key:
                waiting = False

def main():
    start_time = time.time()  # Start time of the game
    while True:
        choice = main_menu()
        player = Player()
        camera = Camera(player)
        level = Level(player.level)

        if choice == 'load':
            load_game(player)
            level.load_level(player.level)
            
        game_running = True
        while game_running:
            handle_events(player)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_p]:
                choice = main_menu(paused=True)
            if keys[pygame.K_s]:
                save_game(player)
            # Calculate elapsed time
            current_time = time.time()
            elapsed_time = current_time - start_time
            remaining_time = max(0, TOTAL_TIME - int(elapsed_time))  # Calculate remaining time
            #convert to string with minutes and seconds
            minutes = remaining_time // 60
            seconds = remaining_time % 60
            time_str = f"{minutes:02}:{seconds:02}"

            update(player, level, camera)
            render(screen, player, level, camera, time_str)
            clock.tick(60)
            if remaining_time <= 0 or player.lives <= 0:
                pygame.mixer.music.rewind()
                game_over_screen()
                game_running = False
                
            if player.lives <= 0:
                game_running = False  # Exit to main menu after game over

if __name__ == "__main__":
    main()