import pygame
from pygame.surfarray import array3d
import sys, time, random

# New imports
from pygame import display
import numpy as np
import gym
from gym import error,spaces,utils
from gym.utils import seeding

# Colors
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0) 

# Our game class
# We create a game in OOP style using pygame
# Note that the game logic is not in side the class, but outside in a loop
# The game logic is actually responsible for updating/drawing the snake body
class SnakeEnv(gym.Env):
    
    # New structure: human playing enabled
    metadata = {'render.modes':['human']}
    
    # We/The do not let the user pass/es the window size
    # to have standard training images
    def __init__(self):
        '''
        Defines the initial game window size.
        '''
        #self.frame_size_x = frame_size_x
        #self.frame_size_y = frame_size_y
        self.frame_size_x = 200
        self.frame_size_y = 200
        self.action_space = spaces.Discrete(4)
        # We create a pygame window with the window size
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
        # How long Gym trains, to avoid looping forever - it depends on the game
        self.STEP_LIMIT = 1000
        # Initialize sleep variable to use it when human plays
        self.sleep = 0
        # We need to define reset()
        self.reset()
        
    def reset(self):
        '''
        Resets the game, along with the default snake size and spawning food.
        '''
        # Reset initializes the state
        self.game_window.fill(BLACK)
        # The position needs to be feasible wrt the window size passed by the user!
        # So it should be bigger than 100x50
        self.snake_pos = [100, 50]
        # The initial snake (head) pos is set to be 3 squares of 10 pixels long
        # starting from its snake_pos
        self.snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]
        # Random positionining function, defined later
        self.food_pos = self.spawn_food()
        # Flag to whether food was alread spawn - yes
        self.food_spawn = True
        # Movement direction, applied continuously
        self.direction = "RIGHT"
        # Action = Change of movement direction
        self.action = self.direction
        self.score = 0
        self.steps = 0
        print("Game Reset.")
        
    def change_direction(self, action, direction):
        '''
        Changes direction based on action input. 
        Checkes to make sure snake can't go back on itself.
        '''
        # Check that we don't go back against the snake
        if action == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if action == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if action == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if action == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'
        # Current direction is changed iff we don't go back against the snake
        return direction
    
    def move(self, direction, snake_pos):
        '''
        Updates snake_pos list to reflect direction change.
        '''
        # One square (10 pixels) updated in the direction
        # Note that we don't update self.snake_pos yet
        if direction == 'UP':
            snake_pos[1] -= 10
        if direction == 'DOWN':
            snake_pos[1] += 10
        if direction == 'LEFT':
            snake_pos[0] -= 10
        if direction == 'RIGHT':
            snake_pos[0] += 10
        return snake_pos

    def spawn_food(self):
        '''
        Spawns food in a random location on window size.
        '''
        # We discretize the window in 10x10 squares and select a random square
        # Note we don't change self.food_pos yet
        return [random.randrange(1, (self.frame_size_x//10)) * 10,
                random.randrange(1, (self.frame_size_y//10)) * 10]

    def eat(self):
        '''
        Returns Boolean indicating if Snake has "eaten" the white food square.
        '''
        # We can eat the food if the snake (head) position
        # is in the same position as the food
        return self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]
    
    def human_step(self, event):
        '''
        Takes human keyboard event and then returns it as an action string.
        '''
        # Here, we let the user interact with the keyboard
        # However, note that the RL environment won't use this function
        # Also note that the pygame event here is another event than the RL one
        # Basically, here we create the action string out from keyboard inputs
        # Default action
        action = None
        # If the event is Quit, the quit
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # If the event is a pressed key
        # convert keybord key to action strings
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 'UP'
            if event.key == pygame.K_DOWN:
                action = 'DOWN'
            if event.key == pygame.K_LEFT:
                action = 'LEFT'
            if event.key == pygame.K_RIGHT:
                action = 'RIGHT'
            # Esc -> Create event to quit the game
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
        return action
    
    def display_score(self, color, font, size):
        '''
        Updates the score in top left.
        '''
        # A system font selected
        score_font = pygame.font.SysFont(font, size)
        # We create a surface where the string is rendered
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        # Take the bounding box element of the text surface
        score_rect = score_surface.get_rect()
        # Set the midtop of the bbox in the desired coordinates
        # It depends on the window size...
        score_rect.midtop = (self.frame_size_x/10, 15)
        # Put it on the snake window
        self.game_window.blit(score_surface, score_rect)
    
    def game_over(self):
        '''
        Checks if the snake has touched the bounding box or itself.
        '''
        # We define the conditions to end the game
        # TOUCH BOX / WINDOW EDGE
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10:
            self.end_game()
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10:
            self.end_game()
        # TOUCH OWN BODY: Compare head to the rest of blocks
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                self.end_game()

    def end_game(self):
        '''
        Clean up and end the game.
        '''
        # Here we end the game;
        # the conditions for that are check in game_over()
        message = pygame.font.SysFont('arial', 45)
        message_surface = message.render('Game has Ended.', True, RED)
        message_rect = message_surface.get_rect()
        message_rect.midtop = (self.frame_size_x/2, self.frame_size_y/4)
        self.game_window.fill(BLACK)
        self.game_window.blit(message_surface, message_rect)
        self.display_score(RED, 'times', 20)
        pygame.display.flip()
        time.sleep(3)
        pygame.quit()
        sys.exit()