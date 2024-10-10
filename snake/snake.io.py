import pygame
import random
import string
from collections import deque

pygame.init()
pygame.mixer.init()  # Initialize the mixer module

# Define the light and dark mode color schemes
LIGHT_MODE_COLORS = {
    "bg": (173, 216, 230),  # Light blue background
    "snake_body": (0, 0, 255),  # Blue snake body
    "snake_head": (0, 0, 128),  # Darker blue snake head
    "food": (255, 255, 0),  # Yellow food
    "text": (0, 0, 0),  # Black text
    "minimap_bg": (200, 200, 200),  # Light minimap background
    "minimap_border": (0, 0, 0)  # Black minimap border
}

DARK_MODE_COLORS = {
    "bg": (25, 25, 25),  # Dark gray background
    "snake_body": (0, 255, 0),
    "snake_head": (0, 128, 0),
    "food": (255, 69, 0),  # Orange-red food
    "text": (255, 255, 255),  # White text
    "minimap_bg": (50, 50, 50),  # Dark minimap background
    "minimap_border": (255, 255, 255)  # White minimap border
}
class Game:
    # Class-level constants for better readability
    FPS = 10
    FOOD_AMOUNT = 200  # Adjusted for performance
    HIGH_SCORE_FILE = "highscore.txt"
    MAP_MULTIPLIER = 4  # Map size is 4 times the game surface size
    WINDOWED_SCREEN_WIDTH = 1200
    WINDOWED_SCREEN_HEIGHT = 800
    SCROLL_MARGIN = 100  # Pixels near the edges to trigger camera movement

    # Colors
    COLOR_PLAYER = (0, 255, 0)
    COLOR_PLAYER_HEAD = (0, 128, 0)
    COLOR_AI_IMMORTAL = (255, 0, 0)
    COLOR_AI_HEAD = (128, 0, 0)
    COLOR_FOOD = (255, 255, 0)
    COLOR_BG = (173, 216, 230)
    COLOR_TEXT = (0, 0, 0)
    COLOR_MINIMAP_BG = (200, 200, 200)
    COLOR_MINIMAP_BORDER = (0, 0, 0)
    COLOR_MINIMAP_PLAYER = (0, 0, 255)
    COLOR_MINIMAP_AI = (255, 0, 0)


    # Directions
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    # Fixed block size
    BLOCK_SIZE = 20  # Fixed size for consistency

    def __init__(self, fullscreen=False):
        # Game settings
        self.fps = self.FPS
        self.food_amount = self.FOOD_AMOUNT
        self.high_score_file = self.HIGH_SCORE_FILE
        self.muted = False
        self.paused = False
        self.background_music_playing = True
        self.selected_option = 0
        self.fullscreen = fullscreen  # Track fullscreen state

        self.dark_mode = True  # Default to light mode

        # Set initial color scheme to light mode
        self.colors = DARK_MODE_COLORS.copy()

        # Retrieve display information
        info = pygame.display.Info()
        self.screen_width = info.current_w if self.fullscreen else self.WINDOWED_SCREEN_WIDTH
        self.screen_height = info.current_h if self.fullscreen else self.WINDOWED_SCREEN_HEIGHT

        # Calculate number of visible columns and rows based on screen size
        self.visible_cols = self.screen_width // self.BLOCK_SIZE
        self.visible_rows = self.screen_height // self.BLOCK_SIZE

        # Define the game surface size based on visible blocks
        self.game_surface_width = self.visible_cols * self.BLOCK_SIZE
        self.game_surface_height = self.visible_rows * self.BLOCK_SIZE

        # Define the map size (fixed size based on windowed screen size)
        self.map_width = self.MAP_MULTIPLIER * self.WINDOWED_SCREEN_WIDTH
        self.map_height = self.MAP_MULTIPLIER * self.WINDOWED_SCREEN_HEIGHT

        # Initialize camera position at (0, 0)
        self.camera_x = 0
        self.camera_y = 0

        # Initialize minimap size
        self.minimap_width = 200  # Fixed size
        self.minimap_height = 150  # Fixed size

        # Load the sound files with exception handling
        self.load_sounds()

        # Create screen with or without fullscreen
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags)
        pygame.display.set_caption('Advanced Snake Game')
        self.clock = pygame.time.Clock()

        # Initialize snakes and foods
        self.initialize_game_objects()

        self.pause_menu_options = [
          "Resume Game",
          "Mute/Unmute Sounds",
          "Toggle Fullscreen",
          "Toggle Dark Mode",  # Add the toggle option for dark mode
          "Respawn Player",
          "Quit Game"
      ]


    def load_sounds(self):
        """Load game sounds with exception handling."""
        try:
            self.food_sound = pygame.mixer.Sound("231769__copyc4t__ding.flac")
        except pygame.error:
            print("Error loading food sound.")
            self.food_sound = None

        try:
            self.boost_sound = pygame.mixer.Sound("538377__kostas17__steam-train-processed.wav")
        except pygame.error:
            print("Error loading boost sound.")
            self.boost_sound = None

        try:
            pygame.mixer.music.load("629147__holizna__chill-lofi-melody-loop-95-bpm.wav")
            pygame.mixer.music.set_volume(0.3)
            pygame.mixer.music.play(-1)
        except pygame.error:
            print("Error loading background music.")
            self.background_music_playing = False

    def initialize_game_objects(self):
        """Initialize player and AI snakes, and food."""
        # Initialize player snake at the center of the map
        self.player_snake = Snake(
            x=self.map_width // 2,
            y=self.map_height // 2,
            color=self.COLOR_PLAYER,
            head_color=self.COLOR_PLAYER_HEAD,
            is_player=True,
            game=self
        )

        # Define AI snake colors
        self.ai_snake_colors = [
            ((0, 128, 128), (0, 64, 64)),
            ((128, 0, 128), (64, 0, 64)),
            ((255, 165, 0), (200, 100, 0)),
            ((0, 255, 255), (0, 128, 128)),
            ((255, 20, 147), (128, 10, 73))
        ]

        # Initialize AI snakes at random positions
        self.ai_snakes = [
            Snake(
                x=random.randint(0, self.map_width // self.BLOCK_SIZE - 1) * self.BLOCK_SIZE,
                y=random.randint(0, self.map_height // self.BLOCK_SIZE - 1) * self.BLOCK_SIZE,
                color=color[0],
                head_color=color[1],
                name=self.generate_random_name(),
                game=self
            )
            for color in self.ai_snake_colors
        ]

        # Initialize food lists
        self.foods = []
        self.snake_foods = []

        # Combine all snakes into a single list
        self.snakes = [self.player_snake] + self.ai_snakes

        # Spawn initial food
        self.spawn_initial_food()

    def spawn_initial_food(self):
        """Spawn initial food items."""
        occupied_positions = self.get_occupied_positions()
        while len(self.foods) < self.food_amount:
            position = Food.spawn(occupied_positions, self)
            if position:
                self.foods.append(Food(position=position, game=self))
                occupied_positions.add(position)
            else:
                break  # No available positions to spawn food

    def generate_random_name(self):
        """Generate a more creative and realistic snake name."""
        # List of creative snake names
        SNAKE_NAMES = [
            "Slithers", "Fang", "Serpentina", "Cobra Kai", "Slinky", 
            "Nagini", "Venom", "Hissandra", "Rattler", "Aspen",
            "Viperion", "Boa", "Sidewinder", "Slyther", "Anaconda",
            "Pythonia", "Copperhead", "Scalez", "Rex", "Kaa"
        ]
        return random.choice(SNAKE_NAMES)

    def load_high_score(self):
        """Load the high score from a file."""
        try:
            with open(self.high_score_file, 'r') as file:
                return int(file.read().strip())
        except Exception:
            return 0

    def save_high_score(self, score):
        """Save the high score to a file."""
        try:
            with open(self.high_score_file, 'w') as file:
                file.write(str(score))
        except Exception as e:
            print(f"Error saving high score: {e}")

    def run(self):
        """Main game loop."""
        while True:
            self.screen.fill(self.COLOR_BG)
            keys = pygame.key.get_pressed()

            for event in pygame.event.get():
                self.handle_events(event)

            if not self.paused:
                self.update_game_logic(keys)
                self.draw_elements()
                self.draw_minimap()
                self.draw_stamina_bar()
            else:
                self.draw_pause_menu()

            pygame.display.flip()
            self.clock.tick(self.fps)

    def handle_events(self, event):
        """Handle user input events."""
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()
            elif event.key == pygame.K_p:
                self.paused = not self.paused
            elif self.paused:
                # Pause menu navigation
                if event.key == pygame.K_UP:
                    self.selected_option = (self.selected_option - 1) % len(self.pause_menu_options)
                elif event.key == pygame.K_DOWN:
                    self.selected_option = (self.selected_option + 1) % len(self.pause_menu_options)
                elif event.key == pygame.K_RETURN:
                    self.handle_pause_menu_selection()
            else:
                # Player movement
                if event.key == pygame.K_UP:
                    self.player_snake.set_direction(self.UP)
                elif event.key == pygame.K_DOWN:
                    self.player_snake.set_direction(self.DOWN)
                elif event.key == pygame.K_LEFT:
                    self.player_snake.set_direction(self.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.player_snake.set_direction(self.RIGHT)

    def handle_pause_menu_selection(self):
        """Handle the selection in the pause menu."""
        option = self.pause_menu_options[self.selected_option]
        if option == "Resume Game":
            self.paused = False
        elif option == "Mute/Unmute Sounds":
            self.muted = not self.muted
            if self.muted:
                pygame.mixer.music.pause()
                if self.boost_sound:
                    pygame.mixer.Channel(1).pause()
            else:
                pygame.mixer.music.unpause()
                if self.boost_sound:
                    pygame.mixer.Channel(1).unpause()
            self.background_music_playing = not self.background_music_playing
        elif option == "Toggle Fullscreen":
            self.toggle_fullscreen()
        elif option == "Respawn Player":
            self.player_snake = Snake(
                x=self.map_width // 2,
                y=self.map_height // 2,
                color=self.COLOR_PLAYER,
                head_color=self.COLOR_PLAYER_HEAD,
                is_player=True,
                game=self
            )
            self.snakes[0] = self.player_snake
            self.paused = False
        elif option == "Toggle Dark Mode":  # Handle the dark mode toggle
            self.toggle_dark_mode()
        elif option == "Quit Game":
            pygame.quit()
            exit()
    def toggle_dark_mode(self):
        """Toggle between dark and light mode."""
        self.dark_mode = not self.dark_mode  # Toggle the mode
        
        # Apply the correct color scheme
        if self.dark_mode:
            self.colors = DARK_MODE_COLORS.copy()
        else:
            self.colors = LIGHT_MODE_COLORS.copy()

        # Feedback to the player
        font = pygame.font.SysFont(None, 36)
        mode_text = "Dark Mode Enabled" if self.dark_mode else "Light Mode Enabled"
        toggle_text = font.render(mode_text, True, self.colors["text"])
        self.screen.blit(toggle_text, (
            self.screen_width // 2 - toggle_text.get_width() // 2,
            self.screen_height // 2 - toggle_text.get_height() // 2
        ))
        pygame.display.flip()
        pygame.time.delay(1000)  # Display for 1 second


    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            # Get current display resolution
            info = pygame.display.Info()
            self.screen_width, self.screen_height = info.current_w, info.current_h
            flags = pygame.FULLSCREEN
        else:
            self.screen_width, self.screen_height = self.WINDOWED_SCREEN_WIDTH, self.WINDOWED_SCREEN_HEIGHT
            flags = 0  # Windowed mode

        # Recalculate visible blocks based on new screen size
        self.visible_cols = self.screen_width // self.BLOCK_SIZE
        self.visible_rows = self.screen_height // self.BLOCK_SIZE

        # Update game surface size based on visible blocks
        self.game_surface_width = self.visible_cols * self.BLOCK_SIZE
        self.game_surface_height = self.visible_rows * self.BLOCK_SIZE

        # Adjust minimap size (optional)
        self.minimap_width = 200  # Fixed size
        self.minimap_height = 150  # Fixed size

        # Recalculate camera position to ensure it's within new boundaries
        self.update_camera()

        # Resize the display
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags)

        # Provide visual feedback for fullscreen toggle (optional)
        font = pygame.font.SysFont(None, 36)
        toggle_text = font.render("Fullscreen Toggled", True, (255, 255, 255))
        self.screen.blit(toggle_text, (
            self.screen_width // 2 - toggle_text.get_width() // 2,
            self.screen_height // 2 - toggle_text.get_height() // 2
        ))
        pygame.display.flip()
        pygame.time.delay(1000)  # Display for 1 second

    def update_game_logic(self, keys):
        """Update game logic, including snake movement and collisions."""
        # Handle dead snakes before updating occupied positions
        self.handle_dead_snakes()
        occupied_positions = self.get_occupied_positions()

        for snake in self.snakes:
            if snake.alive:
                if snake == self.player_snake:
                    # Check if the player is boosting
                    boosting = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
                    if boosting and self.player_snake.stamina > 0:
                        move_times = 2
                        if self.boost_sound and not self.muted:
                            if not pygame.mixer.Channel(1).get_busy():
                                pygame.mixer.Channel(1).play(self.boost_sound)
                    else:
                        move_times = 1
                        if self.boost_sound:
                            pygame.mixer.Channel(1).stop()

                    # Update stamina
                    self.player_snake.update_stamina(boosting)

                    for _ in range(move_times):
                        snake.move()
                        snake.check_collision(self.snakes)
                        self.consume_food(snake)  # Check for food after each move
                else:
                    if self.foods:
                        nearest_food = snake.find_nearest_food(self.foods)
                        snake.move_towards_food(nearest_food)
                    else:
                        snake.move_randomly()
                    snake.move()
                    snake.check_collision(self.snakes)
                    self.consume_food(snake)  # Check for food after each AI move

        # Update high score
        if self.player_snake.score > self.load_high_score():
            self.save_high_score(self.player_snake.score)

        # Spawn new food
        self.spawn_food(occupied_positions)

        # Update camera position based on player position
        self.update_camera()

    def handle_dead_snakes(self):
        """Convert dead snakes into food and respawn AI snakes."""
        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                # Convert each body segment into food
                for segment in snake.body:
                    self.snake_foods.append(Food(position=segment, game=self))
                # Respawn AI snakes or handle game over for the player
                if not snake.is_player:
                    color_index = (i - 1) % len(self.ai_snake_colors)
                    new_snake = Snake(
                        x=random.randint(0, self.map_width // self.BLOCK_SIZE - 1) * self.BLOCK_SIZE,
                        y=random.randint(0, self.map_height // self.BLOCK_SIZE - 1) * self.BLOCK_SIZE,
                        color=self.ai_snake_colors[color_index][0],
                        head_color=self.ai_snake_colors[color_index][1],
                        name=self.generate_random_name(),
                        game=self
                    )
                    self.snakes[i] = new_snake
                else:
                    self.game_over()

    def get_occupied_positions(self):
        """Get all positions occupied by snakes and food."""
        occupied = set()
        for snake in self.snakes:
            if snake.alive:
                occupied.update(snake.body)
        for food in self.foods + self.snake_foods:
            occupied.add(food.position)
        return occupied

    def consume_food(self, snake):
        """Check and handle if the snake has consumed any food."""
        if not snake.alive:
            return

        head_pos = snake.body[0]

        for food_list in [self.foods, self.snake_foods]:
            for food in food_list[:]:
                if food.position == head_pos:
                    snake.grow()
                    food_list.remove(food)
                    # Spawn a new regular food to replace the consumed one
                    new_pos = Food.spawn(self.get_occupied_positions(), self)
                    if new_pos:
                        self.foods.append(Food(position=new_pos, game=self))
                    break  # Stop checking after consuming one food

    def spawn_food(self, occupied_positions):
        """Spawn new food to maintain the desired amount."""
        while len(self.foods) < self.food_amount:
            position = Food.spawn(occupied_positions, self)
            if position:
                self.foods.append(Food(position=position, game=self))
                occupied_positions.add(position)
            else:
                break  # No available positions to spawn food

    def update_camera(self):
        """Update the camera position based on the player's position."""
        head_x, head_y = self.player_snake.body[0]

        # Define the margin within which the camera should start moving
        left_margin = self.camera_x + self.SCROLL_MARGIN
        right_margin = self.camera_x + self.visible_cols * self.BLOCK_SIZE - self.SCROLL_MARGIN
        top_margin = self.camera_y + self.SCROLL_MARGIN
        bottom_margin = self.camera_y + self.visible_rows * self.BLOCK_SIZE - self.SCROLL_MARGIN

        # Move camera left
        if head_x < left_margin and self.camera_x > 0:
            self.camera_x -= self.BLOCK_SIZE
            if self.camera_x < 0:
                self.camera_x = 0

        # Move camera right
        if head_x > right_margin and self.camera_x < self.map_width - self.game_surface_width:
            self.camera_x += self.BLOCK_SIZE
            if self.camera_x > self.map_width - self.game_surface_width:
                self.camera_x = self.map_width - self.game_surface_width

        # Move camera up
        if head_y < top_margin and self.camera_y > 0:
            self.camera_y -= self.BLOCK_SIZE
            if self.camera_y < 0:
                self.camera_y = 0

        # Move camera down
        if head_y > bottom_margin and self.camera_y < self.map_height - self.game_surface_height:
            self.camera_y += self.BLOCK_SIZE
            if self.camera_y > self.map_height - self.game_surface_height:
                self.camera_y = self.map_height - self.game_surface_height

    def game_over(self):
        """Display the game over screen and handle user input."""
        if self.boost_sound:
            pygame.mixer.Channel(1).stop()

        # Update high score before displaying
        current_score = self.player_snake.score
        high_score = self.load_high_score()

        # If current score exceeds high score, update it
        if current_score > high_score:
            self.save_high_score(current_score)
            high_score = current_score  # Update high_score variable

        # Render texts
        font_large = pygame.font.SysFont(None, 72)
        font_medium = pygame.font.SysFont(None, 48)
        font_small = pygame.font.SysFont(None, 36)

        game_over_text = font_large.render("Game Over", True, (255, 0, 0))
        current_score_text = font_medium.render(f"Your Score: {current_score}", True, (255, 255, 255))
        high_score_text = font_medium.render(f"High Score: {high_score}", True, (255, 255, 255))
        restart_text = font_small.render("Press R to Respawn or Q to Quit", True, (255, 255, 255))

        # Blit texts to the screen
        self.screen.blit(game_over_text, (
            self.screen_width // 2 - game_over_text.get_width() // 2,
            self.screen_height // 2 - 150
        ))
        self.screen.blit(current_score_text, (
            self.screen_width // 2 - current_score_text.get_width() // 2,
            self.screen_height // 2 - 50
        ))
        self.screen.blit(high_score_text, (
            self.screen_width // 2 - high_score_text.get_width() // 2,
            self.screen_height // 2
        ))
        self.screen.blit(restart_text, (
            self.screen_width // 2 - restart_text.get_width() // 2,
            self.screen_height // 2 + 100
        ))

        pygame.display.flip()

        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.player_snake = Snake(
                        x=self.map_width // 2,
                        y=self.map_height // 2,
                        color=self.COLOR_PLAYER,
                        head_color=self.COLOR_PLAYER_HEAD,
                        is_player=True,
                        game=self
                    )
                    self.snakes[0] = self.player_snake
                    self.paused = False
                    return
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()

    def draw_elements(self):
        """Draw all game elements on the screen."""
        # Draw background based on current mode
        self.screen.fill(self.colors["bg"])

        # Draw food
        for food in self.foods + self.snake_foods:
            food.draw(self.screen, self.camera_x, self.camera_y, self.BLOCK_SIZE)

        # Draw snakes
        for snake in self.snakes:
            if snake.alive:
                snake.draw(self.screen, self.camera_x, self.camera_y, self.BLOCK_SIZE)

        # Draw scores
        self.draw_scores()

    def draw_scores(self):
        """Draw the scores of the player and AI snakes."""
        font = pygame.font.SysFont(None, 24)
        max_scores_displayed = min(len(self.snakes), 10)
        
        # Use the correct text color based on the mode
        text_color = self.colors["text"]

        for i in range(max_scores_displayed):
            snake = self.snakes[i]
            if snake.is_player:
                label = f"Player Score: {snake.score}"
            else:
                label = f"{snake.name} Score: {snake.score}"
            score_text = font.render(label, True, text_color)
            self.screen.blit(score_text, (10, 10 + i * 20))

        # Display high score
        high_score = self.load_high_score()
        high_score_text = font.render(f"High Score: {high_score}", True, text_color)
        self.screen.blit(high_score_text, (self.screen_width - high_score_text.get_width() - 10, 10))


    def draw_pause_menu(self):
        """Draw the pause menu on the screen with the game board visible in the background."""
        # First, draw all the game elements as they are
        self.draw_elements()
        
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(150)  # Set transparency level (0 is fully transparent, 255 is fully opaque)
        overlay.fill((0, 0, 0))  # Fill with black to create the overlay
        self.screen.blit(overlay, (0, 0))  # Blit the overlay on top of the game elements

        # Draw the menu options on top of the semi-transparent overlay
        font = pygame.font.SysFont(None, 48)
        for index, option in enumerate(self.pause_menu_options):
            color = (255, 255, 255) if index == self.selected_option else (200, 200, 200)
            option_text = font.render(option, True, color)
            self.screen.blit(
                option_text,
                (
                    self.screen_width // 2 - option_text.get_width() // 2,
                    self.screen_height // 2 - 100 + index * 50
                )
            )

    def draw_minimap(self):
        """Draw the minimap showing the overview of the map."""
        minimap_surface = pygame.Surface((self.minimap_width, self.minimap_height))
        minimap_surface.fill(self.COLOR_MINIMAP_BG)
        pygame.draw.rect(minimap_surface, self.COLOR_MINIMAP_BORDER, minimap_surface.get_rect(), 2)

        minimap_scale_x = self.minimap_width / self.map_width
        minimap_scale_y = self.minimap_height / self.map_height

        # Draw snakes on the minimap
        for snake in self.snakes:
            for segment in snake.body:
                if snake.is_player:
                    color = self.COLOR_MINIMAP_PLAYER
                else:
                    color = self.COLOR_MINIMAP_AI if snake.alive else (100, 100, 100)
                x = int(segment[0] * minimap_scale_x)
                y = int(segment[1] * minimap_scale_y)
                if 0 <= x < self.minimap_width and 0 <= y < self.minimap_height:
                    minimap_surface.fill(color, (x, y, 2, 2))

        # Draw food on the minimap
        for food in self.foods + self.snake_foods:
            x = int(food.position[0] * minimap_scale_x)
            y = int(food.position[1] * minimap_scale_y)
            if 0 <= x < self.minimap_width and 0 <= y < self.minimap_height:
                minimap_surface.fill(self.COLOR_FOOD, (x, y, 2, 2))

        # Blit the minimap to the screen
        self.screen.blit(
            minimap_surface,
            (self.screen_width - self.minimap_width - 10, self.screen_height - self.minimap_height - 10)
        )
    def draw_stamina_bar(self):
        """Draw the stamina bar for the player."""
        bar_width = 200
        bar_height = 20
        x_position = 20
        y_position = self.screen_height - 40

        # Calculate the current stamina width
        current_stamina_width = (self.player_snake.stamina / self.player_snake.max_stamina) * bar_width

        # Draw the background of the stamina bar (gray)
        pygame.draw.rect(self.screen, (128, 128, 128), (x_position, y_position, bar_width, bar_height))

        # Draw the current stamina (green)
        pygame.draw.rect(self.screen, (0, 255, 0), (x_position, y_position, current_stamina_width, bar_height))

        # Draw the border of the stamina bar (black)
        pygame.draw.rect(self.screen, (0, 0, 0), (x_position, y_position, bar_width, bar_height), 2)

class Snake:
    def __init__(self, x, y, color, head_color, is_player=False, name="Snake", game=None):
        self.body = deque([(x, y)])
        self.direction = random.choice([game.UP, game.DOWN, game.LEFT, game.RIGHT])
        self.color = color
        self.head_color = head_color
        self.is_player = is_player
        self.alive = True
        self.immortal_ticks = 10 if not is_player else 0
        self.score = 0
        self.name = name
        self.game = game

        # Stamina attributes (for player)
        if self.is_player:
            self.stamina = 100  # Start with full stamina
            self.max_stamina = 100
            self.stamina_depletion_rate = 5  # How much stamina is used per boost
            self.stamina_regeneration_rate = 2  # How much stamina regenerates per frame when not boosting

    def move(self):
        """Move the snake in the current direction."""
        if not self.alive:
            return
        head_x, head_y = self.body[0]
        new_head = (
            (head_x + self.direction[0] * self.game.BLOCK_SIZE) % self.game.map_width,
            (head_y + self.direction[1] * self.game.BLOCK_SIZE) % self.game.map_height
        )
        self.body.appendleft(new_head)
        self.body.pop()

    def grow(self):
        """Grow the snake by adding a segment."""
        tail = self.body[-1]
        self.body.append(tail)
        self.score += 1
        if self.is_player and not self.game.muted and self.game.food_sound:
            self.game.food_sound.play()

    def set_direction(self, new_direction):
        """Set the new direction of the snake, preventing 180-degree turns."""
        if (self.direction[0] * -1, self.direction[1] * -1) != new_direction and new_direction != self.direction:
            self.direction = new_direction

    def update_stamina(self, boosting):
        """Update stamina based on whether the player is boosting or not."""
        if self.is_player:
            if boosting:
                if self.stamina > 0:
                    self.stamina -= self.stamina_depletion_rate
                    if self.stamina < 0:
                        self.stamina = 0  # Ensure stamina doesn't go negative
            else:
                self.stamina += self.stamina_regeneration_rate
                if self.stamina > self.max_stamina:
                    self.stamina = self.max_stamina  # Ensure stamina doesn't exceed max limit

    def check_collision(self, snakes):
        """Check for collisions with other snakes."""
        if self.immortal_ticks > 0:
            self.immortal_ticks -= 1
            return
        head_x, head_y = self.body[0]
        for snake in snakes:
            if not snake.alive:
                continue
            if snake != self and (head_x, head_y) in snake.body:
                self.alive = False
                return

    def find_nearest_food(self, foods):
        """Find the nearest food item."""
        if not foods:
            return None
        head_x, head_y = self.body[0]
        nearest_food = min(
            foods,
            key=lambda food: abs(food.position[0] - head_x) + abs(food.position[1] - head_y)
        )
        return nearest_food

    def move_towards_food(self, food):
        """Move the snake towards the given food item."""
        if not food:
            self.move_randomly()
            return
        possible_directions = [self.game.UP, self.game.DOWN, self.game.LEFT, self.game.RIGHT]
        reverse_direction = (-self.direction[0], -self.direction[1])
        possible_directions = [d for d in possible_directions if d != reverse_direction]

        head_x, head_y = self.body[0]
        dx = food.position[0] - head_x
        dy = food.position[1] - head_y

        preferred_direction = None
        if abs(dx) >= abs(dy):
            if dx > 0:
                preferred_direction = self.game.RIGHT
            elif dx < 0:
                preferred_direction = self.game.LEFT
        else:
            if dy > 0:
                preferred_direction = self.game.DOWN
            elif dy < 0:
                preferred_direction = self.game.UP

        if preferred_direction and preferred_direction in possible_directions:
            self.set_direction(preferred_direction)
        else:
            if possible_directions:
                self.set_direction(random.choice(possible_directions))

    def move_randomly(self):
        """Move the snake in a random valid direction."""
        possible_directions = [self.game.UP, self.game.DOWN, self.game.LEFT, self.game.RIGHT]
        reverse_direction = (-self.direction[0], -self.direction[1])
        possible_directions = [d for d in possible_directions if d != reverse_direction]
        self.set_direction(random.choice(possible_directions))

    def draw(self, screen, camera_x, camera_y, block_size):
        """Draw the snake on the screen."""
        for segment in self.body:
            screen_x = segment[0] - camera_x
            screen_y = segment[1] - camera_y
            if 0 <= screen_x < self.game.game_surface_width and 0 <= screen_y < self.game.game_surface_height:
                # Draw body segments
                pygame.draw.rect(
                    screen,
                    self.color,
                    pygame.Rect(screen_x, screen_y, block_size, block_size)
                )
        # Draw head with a different color
        head = self.body[0]
        head_x = head[0] - camera_x
        head_y = head[1] - camera_y
        if 0 <= head_x < self.game.game_surface_width and 0 <= head_y < self.game.game_surface_height:
            pygame.draw.rect(
                screen,
                self.head_color,
                pygame.Rect(head_x, head_y, block_size, block_size)
            )

class Food:
    def __init__(self, position=None, game=None):
        self.game = game
        self.position = (
            (position[0] // self.game.BLOCK_SIZE) * self.game.BLOCK_SIZE,
            (position[1] // self.game.BLOCK_SIZE) * self.game.BLOCK_SIZE
        )

    @staticmethod
    def spawn(occupied_positions, game):
        """Spawn a food item at a random unoccupied position."""
        all_positions = set(
            (x * game.BLOCK_SIZE, y * game.BLOCK_SIZE)
            for x in range(game.map_width // game.BLOCK_SIZE)
            for y in range(game.map_height // game.BLOCK_SIZE)
        )
        available_positions = list(all_positions - occupied_positions)
        if not available_positions:
            return None
        return random.choice(available_positions)

    def draw(self, screen, camera_x, camera_y, block_size):
        """Draw the food on the screen."""
        screen_x = self.position[0] - camera_x
        screen_y = self.position[1] - camera_y
        if 0 <= screen_x < self.game.game_surface_width and 0 <= screen_y < self.game.game_surface_height:
            pygame.draw.rect(
                screen,
                self.game.COLOR_FOOD,
                pygame.Rect(screen_x, screen_y, block_size, block_size)
            )

if __name__ == "__main__":
    # Start the game in windowed mode
    # To start in fullscreen, set fullscreen=True
    game = Game(fullscreen=False)
    game.run()
