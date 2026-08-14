# 0. Import
import pygame
import random


# 1. Initialize
pygame.init()


# 2. Constants
WIDTH = 600
HEIGHT = 600
CELL_SIZE = 20

# 3. Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SNAKE GAME")

# 4. Create a clock
clock = pygame.time.Clock()

# Variables
snake = [(5, 10), (6, 10), (7, 10)]
direction = (1,0)
food = (random.randint(0, 29), random.randint(0, 29))
score = 0
font = pygame.font.Font(None, 36)
game_state = "playing"

# 5. Game loop
running = True
while running:
    # a. Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if game_state == "playing":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    new_dir = (1, 0)
                    if (direction[0] + new_dir[0], direction[1] + new_dir[1]) != (0,0):
                        direction = new_dir
                if event.key == pygame.K_LEFT:
                    new_dir = (-1,0)
                    if (direction[0] + new_dir[0], direction[1] + new_dir[1]) != (0,0):
                        direction = new_dir
                if event.key == pygame.K_UP:
                    new_dir = (0,-1)
                    if (direction[0] + new_dir[0], direction[1] + new_dir[1]) != (0,0):
                        direction = new_dir
                if event.key == pygame.K_DOWN:
                    new_dir = (0,1)
                    if (direction[0] + new_dir[0], direction[1] + new_dir[1]) != (0,0):
                        direction = new_dir

        if game_state == "game_over":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    snake = [(5, 10), (6, 10), (7, 10)]
                    direction = (1, 0)
                    food = (random.randint(0, 29), random.randint(0, 29))
                    score = 0
                    game_state = "playing"
                if event.key == pygame.K_q:
                    running = False   

    if game_state == "playing":
        # Snake config
        # Check the head
        head = snake[-1]

        # Calculate new head
        new_head = (head[0] + direction[0], head[1] + direction[1])

        if new_head[0] < 0 or new_head[0] >= 30 or new_head[1] < 0 or new_head[1] >= 30:
            game_state = "game_over"
        elif new_head in snake:
            game_state = "game_over"
        else:
            snake.append(new_head)
            if new_head == food:
                food = (random.randint(0, 29), random.randint(0, 29))
                score += 1
            else:
                snake.pop(0)

    # b. Draw
    screen.fill((0, 0, 0))

    # c. Draw grid lines
    # vertical
    for col in range(30):
        x = col * CELL_SIZE
        pygame.draw.line(screen, (40, 40, 40), (x, 0), (x, HEIGHT))

    # horizontal
    for row in range(30):
        y = row * CELL_SIZE
        pygame.draw.line(screen, (40, 40, 40), (0, y), (WIDTH, y))

    # Draw snake
    for segment in snake:
        x = segment[0] * CELL_SIZE
        y = segment[1] * CELL_SIZE
        pygame.draw.rect(screen, (0, 200, 0), (x, y, CELL_SIZE, CELL_SIZE))

    # Draw food
    fx = food[0] * CELL_SIZE
    fy = food[1] * CELL_SIZE
    pygame.draw.rect(screen, (200, 0, 0), (fx, fy, CELL_SIZE, CELL_SIZE))

    # Draw score
    text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    # Game over screen
    if game_state == "game_over":
        go_text = font.render("GAME OVER", True, (255, 0, 0))
        go_rect = go_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(go_text, go_rect)

        restart_text = font.render("Press R to restart / Q to quit", True, (255, 255, 255))
        restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40))
        screen.blit(restart_text, restart_rect)

    # d. Push
    pygame.display.flip()

    # e. Cap the framerate
    clock.tick(10)

pygame.quit()