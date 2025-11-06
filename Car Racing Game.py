import pygame
import cv2
import numpy as np
import random
import math
import time
from pygame import gfxdraw

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
ROAD_WIDTH = 400
LANE_WIDTH = ROAD_WIDTH // 3
CAR_WIDTH = 60
CAR_HEIGHT = 100
FINISH_LINE_DISTANCE = 5000

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
GREEN = (0, 255, 0)
BROWN = (139, 69, 19)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
DARK_GREEN = (0, 100, 0)
TREE_GREEN = (34, 139, 34)
HOUSE_COLOR = (160, 82, 45)

class HeadTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_face_center = None
        self.baseline_face_size = None
        self.baseline_face_y = None
        self.calibration_frames = 50
        self.frame_count = 0
        self.face_positions = []
        self.movement_smoothing = []
        self.smoothing_frames = 5
        
    def get_head_movement(self):
        ret, frame = self.cap.read()
        if not ret:
            return 0, 0  # No movement if camera fails
        
        frame = cv2.flip(frame, 1)  # Mirror the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) > 0:
            # Get the largest face (closest to camera)
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            face_size = w * h
            
            # Calibration phase
            if self.frame_count < self.calibration_frames:
                self.face_positions.append((face_center_x, face_center_y, face_size))
                self.frame_count += 1
                if self.frame_count == self.calibration_frames:
                    # Calculate baseline
                    avg_x = sum(pos[0] for pos in self.face_positions) / len(self.face_positions)
                    avg_y = sum(pos[1] for pos in self.face_positions) / len(self.face_positions)
                    avg_size = sum(pos[2] for pos in self.face_positions) / len(self.face_positions)
                    self.last_face_center = (avg_x, avg_y)
                    self.baseline_face_size = avg_size
                    self.baseline_face_y = avg_y
                return 0, 0
            
            if self.last_face_center and self.baseline_face_size and self.baseline_face_y:
                # Calculate movement with enhanced sensitivity
                dx = face_center_x - self.last_face_center[0]
                dy = face_center_y - self.baseline_face_y
                
                # Size change for forward/backward movement (more sensitive)
                size_ratio = face_size / self.baseline_face_size
                size_change = size_ratio - 1.0
                
                # Enhanced sensitivity for horizontal movement
                horizontal_movement = max(-1, min(1, dx / 30))  # More sensitive steering
                
                # Combine size change and y-movement for vertical control
                y_movement = max(-1, min(1, -dy / 25))  # Head up/down (inverted)
                size_movement = max(-1, min(1, size_change * 8))  # Face closer/farther
                
                # Use the stronger signal for vertical movement
                if abs(size_movement) > abs(y_movement):
                    vertical_movement = size_movement
                else:
                    vertical_movement = y_movement
                
                # Apply smoothing
                current_movement = (horizontal_movement, vertical_movement)
                self.movement_smoothing.append(current_movement)
                
                if len(self.movement_smoothing) > self.smoothing_frames:
                    self.movement_smoothing.pop(0)
                
                # Calculate smoothed movement
                if len(self.movement_smoothing) >= 3:
                    avg_h = sum(m[0] for m in self.movement_smoothing) / len(self.movement_smoothing)
                    avg_v = sum(m[1] for m in self.movement_smoothing) / len(self.movement_smoothing)
                    return avg_h, avg_v
                else:
                    return horizontal_movement, vertical_movement
        
        return 0, 0
    
    def release(self):
        self.cap.release()

class GameObject:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.rect = pygame.Rect(x, y, width, height)
    
    def update_position(self, dx, dy):
        self.x += dx
        self.y += dy
        self.rect.x = self.x
        self.rect.y = self.y
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

class Tree(GameObject):
    def __init__(self, x, y):
        super().__init__(x, y, 30, 80, BROWN)
        self.leaves_rect = pygame.Rect(x - 20, y - 40, 70, 60)
    
    def draw(self, screen):
        # Draw trunk
        pygame.draw.rect(screen, self.color, self.rect)
        # Draw leaves
        pygame.draw.ellipse(screen, TREE_GREEN, self.leaves_rect)
    
    def update_position(self, dx, dy):
        super().update_position(dx, dy)
        self.leaves_rect.x += dx
        self.leaves_rect.y += dy

class House(GameObject):
    def __init__(self, x, y):
        super().__init__(x, y, 80, 60, HOUSE_COLOR)
        self.roof_points = [(x + 40, y - 20), (x, y), (x + 80, y)]
    
    def draw(self, screen):
        # Draw house
        pygame.draw.rect(screen, self.color, self.rect)
        # Draw roof
        pygame.draw.polygon(screen, RED, self.roof_points)
        # Draw door
        door_rect = pygame.Rect(self.x + 30, self.y + 20, 20, 40)
        pygame.draw.rect(screen, DARK_GRAY, door_rect)
    
    def update_position(self, dx, dy):
        super().update_position(dx, dy)
        self.roof_points = [(self.x + 40, self.y - 20), (self.x, self.y), (self.x + 80, self.y)]

class Hurdle(GameObject):
    def __init__(self, x, y):
        super().__init__(x, y, 40, 30, RED)
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 3)

class Balloon:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.speed = random.uniform(1, 3)
        self.sway = random.uniform(-1, 1)
        self.string_length = random.randint(50, 100)
    
    def update(self):
        self.y -= self.speed
        self.x += self.sway * 0.5
    
    def draw(self, screen):
        # Draw balloon
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 20)
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), 20, 2)
        # Draw string
        pygame.draw.line(screen, BLACK, (self.x, self.y + 20), (self.x, self.y + self.string_length), 2)

class CarRacingGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Head-Controlled Car Racing")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)
        
        # Game state
        self.game_state = "start"  # "start", "playing", "finished", "game_over"
        self.player_name = ""
        self.input_active = False
        
        # Game objects
        self.car_x = SCREEN_WIDTH // 2
        self.car_y = SCREEN_HEIGHT - 150
        self.car_speed = 0
        self.max_speed = 8
        self.min_speed = 1
        self.road_offset = 0
        self.distance_traveled = 0
        
        # Head tracking
        self.head_tracker = HeadTracker()
        
        # Environment objects
        self.trees = []
        self.houses = []
        self.hurdles = []
        self.balloons = []
        
        # Timing
        self.start_time = 0
        self.finish_time = 0
        
        # Generate initial environment
        self.generate_environment()
        
    def generate_environment(self):
        # Generate trees on both sides
        for i in range(100):
            y_pos = -i * 200 - random.randint(0, 100)
            # Left side trees
            if random.random() < 0.7:
                self.trees.append(Tree(50 + random.randint(0, 100), y_pos))
            # Right side trees
            if random.random() < 0.7:
                self.trees.append(Tree(SCREEN_WIDTH - 150 + random.randint(0, 100), y_pos))
        
        # Generate houses
        for i in range(30):
            y_pos = -i * 300 - random.randint(0, 200)
            side = random.choice(["left", "right"])
            if side == "left":
                self.houses.append(House(100 + random.randint(0, 50), y_pos))
            else:
                self.houses.append(House(SCREEN_WIDTH - 200 + random.randint(0, 50), y_pos))
        
        # Generate hurdles on the road
        for i in range(20):
            y_pos = -i * 400 - random.randint(200, 400)
            lane = random.randint(0, 2)  # 3 lanes
            x_pos = SCREEN_WIDTH // 2 - ROAD_WIDTH // 2 + lane * LANE_WIDTH + LANE_WIDTH // 2 - 20
            self.hurdles.append(Hurdle(x_pos, y_pos))
    
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if self.game_state == "start":
                if event.type == pygame.KEYDOWN:
                    if self.input_active:
                        if event.key == pygame.K_RETURN:
                            if self.player_name.strip():
                                self.start_game()
                        elif event.key == pygame.K_BACKSPACE:
                            self.player_name = self.player_name[:-1]
                        else:
                            if len(self.player_name) < 20:
                                self.player_name += event.unicode
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    # Check if clicked on input field
                    input_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, 300, 300, 50)
                    if input_rect.collidepoint(mouse_x, mouse_y):
                        self.input_active = True
                    # Check if clicked on play button
                    elif (SCREEN_WIDTH // 2 - 100 <= mouse_x <= SCREEN_WIDTH // 2 + 100 and
                          400 <= mouse_y <= 450):
                        if self.player_name.strip():
                            self.start_game()
            
            elif self.game_state in ["finished", "game_over"]:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.restart_game()
        
        return True
    
    def start_game(self):
        self.game_state = "playing"
        self.start_time = time.time()
        self.car_speed = self.min_speed
    
    def restart_game(self):
        self.game_state = "start"
        self.player_name = ""
        self.input_active = False
        self.car_x = SCREEN_WIDTH // 2
        self.car_speed = 0
        self.road_offset = 0
        self.distance_traveled = 0
        self.balloons = []
        # Regenerate environment
        self.trees = []
        self.houses = []
        self.hurdles = []
        self.generate_environment()
    
    def update_game(self):
        if self.game_state != "playing":
            return
        
        # Get head movement
        horizontal_movement, vertical_movement = self.head_tracker.get_head_movement()
        
        # Update car horizontal position with enhanced sensitivity
        car_move_x = horizontal_movement * 7  # Increased sensitivity
        new_car_x = self.car_x + car_move_x
        
        # Keep car within road boundaries
        road_left = SCREEN_WIDTH // 2 - ROAD_WIDTH // 2
        road_right = SCREEN_WIDTH // 2 + ROAD_WIDTH // 2
        
        if road_left + CAR_WIDTH // 2 <= new_car_x <= road_right - CAR_WIDTH // 2:
            self.car_x = new_car_x
        
        # Update car speed based on vertical movement with better responsiveness
        speed_change = vertical_movement * 0.5  # Increased sensitivity
        self.car_speed = max(self.min_speed, min(self.max_speed, self.car_speed + speed_change))
        
        # Update road offset and distance (synchronized with car speed)
        movement_speed = self.car_speed
        self.road_offset += movement_speed
        self.distance_traveled += movement_speed
        
        # Move all environment objects at the same speed as the road
        for tree in self.trees:
            tree.update_position(0, movement_speed)
        
        for house in self.houses:
            house.update_position(0, movement_speed)
        
        for hurdle in self.hurdles:
            hurdle.update_position(0, movement_speed)
        
        # Check collisions with hurdles
        car_rect = pygame.Rect(self.car_x - CAR_WIDTH // 2, self.car_y - CAR_HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT)
        for hurdle in self.hurdles[:]:  # Create copy for safe removal
            if car_rect.colliderect(hurdle.rect):
                self.game_state = "game_over"
                return
        
        # Remove objects that are too far behind
        self.trees = [tree for tree in self.trees if tree.y < SCREEN_HEIGHT + 100]
        self.houses = [house for house in self.houses if house.y < SCREEN_HEIGHT + 100]
        self.hurdles = [hurdle for hurdle in self.hurdles if hurdle.y < SCREEN_HEIGHT + 100]
        
        # Check if finished
        if self.distance_traveled >= FINISH_LINE_DISTANCE:
            self.game_state = "finished"
            self.finish_time = time.time()
            # Create celebration balloons
            for _ in range(50):
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = SCREEN_HEIGHT + random.randint(0, 100)
                color = random.choice([RED, BLUE, GREEN, YELLOW, (255, 0, 255), (255, 165, 0)])
                self.balloons.append(Balloon(x, y, color))
    
    def draw_start_screen(self):
        self.screen.fill(WHITE)
        
        # Title
        title_text = self.big_font.render("Head-Controlled Racing", True, BLACK)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 150))
        self.screen.blit(title_text, title_rect)
        
        # Instructions
        instructions = [
            "Tilt head LEFT/RIGHT to steer",
            "Tilt head UP to accelerate",
            "Tilt head DOWN to slow down",
            "Stay in lanes and avoid red hurdles!"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.font.render(instruction, True, BLACK)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, 200 + i * 30))
            self.screen.blit(text, text_rect)
        
        # Input field
        pygame.draw.rect(self.screen, WHITE, (SCREEN_WIDTH // 2 - 150, 300, 300, 50))
        pygame.draw.rect(self.screen, BLACK, (SCREEN_WIDTH // 2 - 150, 300, 300, 50), 2)
        
        # Input text
        input_text = self.font.render(self.player_name, True, BLACK)
        self.screen.blit(input_text, (SCREEN_WIDTH // 2 - 140, 315))
        
        # Placeholder text
        if not self.player_name:
            placeholder = self.font.render("Enter your name", True, GRAY)
            self.screen.blit(placeholder, (SCREEN_WIDTH // 2 - 140, 315))
        
        # Play button
        button_color = GREEN if self.player_name.strip() else GRAY
        pygame.draw.rect(self.screen, button_color, (SCREEN_WIDTH // 2 - 100, 400, 200, 50))
        pygame.draw.rect(self.screen, BLACK, (SCREEN_WIDTH // 2 - 100, 400, 200, 50), 2)
        
        play_text = self.font.render("PLAY", True, BLACK)
        play_rect = play_text.get_rect(center=(SCREEN_WIDTH // 2, 425))
        self.screen.blit(play_text, play_rect)
    
    def draw_road(self):
        # Draw grass background
        self.screen.fill(GREEN)
        
        # Draw road
        road_left = SCREEN_WIDTH // 2 - ROAD_WIDTH // 2
        road_right = SCREEN_WIDTH // 2 + ROAD_WIDTH // 2
        pygame.draw.rect(self.screen, DARK_GRAY, (road_left, 0, ROAD_WIDTH, SCREEN_HEIGHT))
        
        # Draw lane dividers
        lane_divider_width = 4
        for i in range(1, 3):
            x = road_left + i * LANE_WIDTH
            # Dashed lines
            dash_length = 40
            gap_length = 20
            total_length = dash_length + gap_length
            
            # Calculate offset for moving dashes
            dash_offset = int(self.road_offset) % total_length
            
            for y in range(-dash_offset, SCREEN_HEIGHT + total_length, total_length):
                if y + dash_length > 0:
                    dash_rect = pygame.Rect(x - lane_divider_width // 2, y, lane_divider_width, dash_length)
                    pygame.draw.rect(self.screen, WHITE, dash_rect)
        
        # Draw finish line if close
        finish_line_y = SCREEN_HEIGHT - (FINISH_LINE_DISTANCE - self.distance_traveled)
        if finish_line_y > -50 and finish_line_y < SCREEN_HEIGHT + 50:
            # Checkered pattern
            square_size = 20
            for i in range(ROAD_WIDTH // square_size + 1):
                for j in range(3):
                    x = road_left + i * square_size
                    y = finish_line_y + j * square_size - square_size
                    color = BLACK if (i + j) % 2 == 0 else WHITE
                    pygame.draw.rect(self.screen, color, (x, y, square_size, square_size))
    
    def draw_environment(self):
        # Draw all environment objects
        for tree in self.trees:
            if -100 <= tree.y <= SCREEN_HEIGHT + 100:
                tree.draw(self.screen)
        
        for house in self.houses:
            if -100 <= house.y <= SCREEN_HEIGHT + 100:
                house.draw(self.screen)
        
        for hurdle in self.hurdles:
            if -100 <= hurdle.y <= SCREEN_HEIGHT + 100:
                hurdle.draw(self.screen)
    
    def draw_car(self):
        # Draw car body
        car_rect = pygame.Rect(self.car_x - CAR_WIDTH // 2, self.car_y - CAR_HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT)
        pygame.draw.rect(self.screen, BLUE, car_rect)
        pygame.draw.rect(self.screen, BLACK, car_rect, 3)
        
        # Draw car windows
        window_rect = pygame.Rect(self.car_x - CAR_WIDTH // 2 + 10, self.car_y - CAR_HEIGHT // 2 + 10, CAR_WIDTH - 20, 30)
        pygame.draw.rect(self.screen, (135, 206, 235), window_rect)
        
        # Draw wheels
        wheel_size = 12
        pygame.draw.circle(self.screen, BLACK, (int(self.car_x - CAR_WIDTH // 2 + 15), int(self.car_y + CAR_HEIGHT // 2 - 15)), wheel_size)
        pygame.draw.circle(self.screen, BLACK, (int(self.car_x + CAR_WIDTH // 2 - 15), int(self.car_y + CAR_HEIGHT // 2 - 15)), wheel_size)
        pygame.draw.circle(self.screen, BLACK, (int(self.car_x - CAR_WIDTH // 2 + 15), int(self.car_y - CAR_HEIGHT // 2 + 15)), wheel_size)
        pygame.draw.circle(self.screen, BLACK, (int(self.car_x + CAR_WIDTH // 2 - 15), int(self.car_y - CAR_HEIGHT // 2 + 15)), wheel_size)
    
    def draw_ui(self):
        if self.game_state == "playing":
            # Create semi-transparent background for UI elements
            ui_surface = pygame.Surface((300, 200))
            ui_surface.set_alpha(180)
            ui_surface.fill(BLACK)
            self.screen.blit(ui_surface, (5, 5))
            
            # Speed indicator
            speed_text = self.font.render(f"Speed: {self.car_speed:.1f}", True, WHITE)
            self.screen.blit(speed_text, (15, 15))
            
            # Time
            elapsed_time = time.time() - self.start_time
            time_text = self.font.render(f"Time: {elapsed_time:.1f}s", True, WHITE)
            self.screen.blit(time_text, (15, 50))
            
            # Distance progress
            progress = (self.distance_traveled / FINISH_LINE_DISTANCE) * 100
            progress_text = self.font.render(f"Progress: {progress:.1f}%", True, WHITE)
            self.screen.blit(progress_text, (15, 85))
            
            # Player name
            name_text = self.font.render(f"Player: {self.player_name}", True, WHITE)
            self.screen.blit(name_text, (15, 120))
            
            # Head tracking status
            if self.head_tracker.frame_count < self.head_tracker.calibration_frames:
                calib_progress = (self.head_tracker.frame_count / self.head_tracker.calibration_frames) * 100
                calib_text = self.font.render(f"Calibrating: {calib_progress:.0f}%", True, YELLOW)
                self.screen.blit(calib_text, (15, 155))
            else:
                ready_text = self.font.render("Head Tracking: READY", True, GREEN)
                self.screen.blit(ready_text, (15, 155))
    
    def draw_finished_screen(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Update and draw balloons
        for balloon in self.balloons[:]:
            balloon.update()
            if balloon.y > -50:
                balloon.draw(self.screen)
            else:
                self.balloons.remove(balloon)
        
        # Victory message
        victory_text = self.big_font.render("CONGRATULATIONS!", True, WHITE)
        victory_rect = victory_text.get_rect(center=(SCREEN_WIDTH // 2, 200))
        self.screen.blit(victory_text, victory_rect)
        
        # Player name
        name_text = self.font.render(f"{self.player_name} finished the race!", True, WHITE)
        name_rect = name_text.get_rect(center=(SCREEN_WIDTH // 2, 280))
        self.screen.blit(name_text, name_rect)
        
        # Time
        total_time = self.finish_time - self.start_time
        time_text = self.font.render(f"Your time: {total_time:.2f} seconds", True, WHITE)
        time_rect = time_text.get_rect(center=(SCREEN_WIDTH // 2, 320))
        self.screen.blit(time_text, time_rect)
        
        # Restart instruction
        restart_text = self.font.render("Press SPACE to play again", True, WHITE)
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, 400))
        self.screen.blit(restart_text, restart_rect)
    
    def draw_game_over_screen(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Game over message
        game_over_text = self.big_font.render("GAME OVER!", True, RED)
        game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, 300))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Restart instruction
        restart_text = self.font.render("Press SPACE to try again", True, WHITE)
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, 400))
        self.screen.blit(restart_text, restart_rect)
    
    def run(self):
        running = True
        
        while running:
            running = self.handle_input()
            
            if self.game_state == "start":
                self.draw_start_screen()
            elif self.game_state == "playing":
                self.update_game()
                self.draw_road()
                self.draw_environment()
                self.draw_car()
                self.draw_ui()
            elif self.game_state == "finished":
                self.draw_road()
                self.draw_environment()
                self.draw_car()
                self.draw_finished_screen()
            elif self.game_state == "game_over":
                self.draw_road()
                self.draw_environment()
                self.draw_car()
                self.draw_game_over_screen()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        self.head_tracker.release()
        pygame.quit()

if __name__ == "__main__":
    try:
        game = CarRacingGame()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required libraries installed:")
        print("pip install pygame opencv-python numpy")