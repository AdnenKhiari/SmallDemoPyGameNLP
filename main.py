import pygame
import sys
import random  # Import random for generating random colors
from processor import Command, TinyBERTProcessor

# Pygame setup
pygame.init()

# Constants for screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Create the display surface
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Command Processor Game")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Character settings
character_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
character_size = 50
character_color = BLUE
jumping = False
jump_height = 10
jump_count = jump_height

# Function to generate a random non-black color
def get_random_color():
    while True:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if color != BLACK:  # Ensure the color is not black
            return color

# Initialize the command processors
tinybert_processor = TinyBERTProcessor('./tinybert-trained-model', 'huawei-noah/TinyBERT_General_4L_312D')
command_processor = tinybert_processor  # Choose which processor to use

# Game loop
clock = pygame.time.Clock()
input_command = ""
action = None  # To store the action determined by the command processor
command_processed = False  # Flag to indicate if command was processed

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:  # Process command on Enter key
                print(f"Processing command: {input_command}")
                action = command_processor.process_command(input_command)  # Use the command processor
                
                # Check if the action is to change color
                if action == Command.CHANGE_COLOR:
                    character_color = get_random_color()  # Change to a random color
                
                input_command = ""  # Clear the input command after processing
                command_processed = True  # Set flag to indicate command has been processed
            elif event.key == pygame.K_BACKSPACE:
                input_command = input_command[:-1]  # Remove the last character
            else:
                input_command += event.unicode  # Add character to input command

    # Update character position based on action
    if action == Command.MOVE_LEFT:
        character_pos[0] -= 5
    elif action == Command.MOVE_RIGHT:
        character_pos[0] += 5
    elif action == Command.JUMP and not jumping:
        jumping = True

    # Jumping logic
    if jumping:
        if jump_count >= -jump_height:
            neg = 1 if jump_count >= 0 else -1
            character_pos[1] -= (jump_count ** 2) * 0.1 * neg  # Parabolic jump
            jump_count -= 1
        else:
            jump_count = jump_height
            jumping = False

    # Clip character position to keep it within the screen borders
    character_pos[0] = max(0, min(character_pos[0], SCREEN_WIDTH - character_size))
    character_pos[1] = max(0, min(character_pos[1], SCREEN_HEIGHT - character_size))

    # Fill the screen with black
    screen.fill(BLACK)

    # Draw the character
    pygame.draw.rect(screen, character_color, (*character_pos, character_size, character_size))

    # Draw input box
    input_box = pygame.Rect(50, SCREEN_HEIGHT - 50, 700, 40)
    pygame.draw.rect(screen, WHITE, input_box, 2)

    # Render the input text
    font = pygame.font.Font(None, 36)
    text_surface = font.render(input_command, True, WHITE)
    screen.blit(text_surface, (input_box.x + 5, input_box.y + 5))

    # Display the action taken (for debugging and user feedback)
    if command_processed:
        action_surface = font.render(f"Action: {action}" if action else "Unknown Command", True, RED)
        screen.blit(action_surface, (50, SCREEN_HEIGHT - 100))
        command_processed = False  # Reset the command processed flag

    # Update the display
    pygame.display.flip()
    clock.tick(FPS)
