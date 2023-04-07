import pygame
import random
import math
import imageio

def upsample_trajectory(xs, ys):
    upsampled_xs = []
    upsampled_ys = []

    # Iterate over each pair of consecutive points in the original trajectory
    for i in range(len(xs) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]

        # Compute the distance and angle between the two points
        dx = x1 - x0
        dy = y1 - y0
        dist = ((dx ** 2) + (dy ** 2)) ** 0.5
        angle = math.atan2(dy, dx)

        # Compute the step size for this segment
        step = dist / 10.0

        # Add the upsampled points for this segment to the output lists
        for j in range(10):
            t = j / 10.0
            x = x0 + (t * dx)
            y = y0 + (t * dy)
            upsampled_xs.append(x)
            upsampled_ys.append(y)

    # Add the final point to the output lists
    upsampled_xs.append(xs[-1])
    upsampled_ys.append(ys[-1])

    return upsampled_xs, upsampled_ys


frames = []

for ep in range(100):
    with open(f"Testing/moves_{ep}.txt", "r") as file:
        # Read the first line and convert it to a list of integers
        line1 = file.readline().split()
        xs = [int(item) for item in line1]
        
        # Read the second line and convert it to a list of integers
        line2 = file.readline().split()
        ys = [int(item) for item in line2]


    xs, ys = upsample_trajectory(xs, ys)
    # Initialize Pygame
    pygame.init()

    # Load the background image
    background = pygame.image.load("world_cropped.jpg")

    # Set the dimensions of the window
    WIDTH = background.get_width()
    HEIGHT = background.get_height()

    # Create the window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Set the title of the window
    pygame.display.set_caption("Moving Circle")

    # Set the initial position and velocity of the circle
    x = xs[0]
    y = ys[0]
    vx = random.randint(-5, 5)
    vy = random.randint(-5, 5)

    # Set the radius and color of the circle
    radius = 154
    color = (0, 255, 0)

    # Set the thickness of the circle
    thickness = 5

    # Set the clock for the game loop
    clock = pygame.time.Clock()

    # Set the starting index of the trajectory
    index = 0

    # Game loop
    for idx in range(len(xs)):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        # Move the circle
        x = xs[index]
        y = ys[index]
        index += 1

        color = (255*(len(xs)-idx)/len(xs), 255*idx/len(xs), 0)
        
        # Bounce the circle off the walls
        if x - radius < 0 or x + radius > WIDTH:
            vx = -vx
        if y - radius < 0 or y + radius > HEIGHT:
            vy = -vy
        
        # Clear the screen
        screen.blit(background, (0, 0))
        
        # Draw the circle
        pygame.draw.circle(screen, color, (x, y), radius, thickness)

        pygame.draw.circle(screen, (255, 255, 255), (800, 667), radius, thickness)
        
        # Update the screen
        pygame.display.update()
        
        # Wait for half a second
        pygame.time.wait(80)

        frames.append(pygame.surfarray.array3d(screen))
        # Set the frame rate
        clock.tick(120)
# Save the frames as a GIF
imageio.mimsave(f"./Gifs/animation1_{ep}.gif", frames, fps=30)


