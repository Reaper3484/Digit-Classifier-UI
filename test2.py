import pygame

# Initialize pygame
pygame.init()

# Screen settings
width, height = 800, 600
input_screen_height = 100  # For example, this could be a separate area for input
output_screen_width = 300
padding = 10
output_rect_padding = 5
output_rect_color = (0, 128, 255)
background_color = (30, 30, 30)

# Sample probabilities (these would come from your model)
probabilities = [0.1, 0.3, 0.8, 0.6, 0.9, 0.2, 0.4, 0.7, 0.5, 0.05]

# Screen creation
screen = pygame.display.set_mode((width, height))

# Define rect height for each bar
rect_height = (height - input_screen_height) // 10

def draw_horizontal_rectangles(probabilities):
    """
    Draw horizontal rectangles based on the given probabilities.
    Each probability determines the width of the rectangle.
    """
    for i, prob in enumerate(probabilities):
        # Calculate the width of the rectangle proportional to the probability
        rect_width = int(prob * (output_screen_width - padding))  # Scaled width based on probability
        
        # Calculate the position and height of the rectangle
        rect_x = width - output_screen_width + padding  # Start from the right side of the screen
        rect_y = i * rect_height + output_rect_padding  # Vertical position for each bar

        # Draw the rectangle
        pygame.draw.rect(screen, output_rect_color, (rect_x, rect_y, rect_width, rect_height - output_rect_padding))

# Main loop
running = True
while running:
    screen.fill(background_color)  # Fill the screen with a background color

    # Draw horizontal rectangles based on probabilities
    draw_horizontal_rectangles(probabilities)

    # Update the display
    pygame.display.flip()

    # Handle events (such as window close)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit pygame
pygame.quit()
