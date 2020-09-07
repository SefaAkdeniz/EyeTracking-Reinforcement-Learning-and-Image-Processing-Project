# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:38:08 2020

@author: sefa
"""

import pygame
import random

pygame.init()
clock = pygame.time.Clock()
timer = 10  # Decrease this to count down.
dt = 0  # Delta time (time since last tick).

black = (0,0,0)
red = (255,0,0)

gameDisplay = pygame.display.set_mode((1920,1080),pygame.FULLSCREEN)
gameDisplay.fill(black)

while True:
    timer -= dt
    if timer <= 0:
        timer = 10 
    dt = clock.tick(30) / 1000 
    if timer<5:
        gameDisplay.fill(black)
        x=random.randint(0,1920)
        y=random.randint(0, 1080)
        print(x)
        print(y)
        pygame.draw.circle(gameDisplay, red, (x,y), 25)
        timer=10
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    pygame.display.update()