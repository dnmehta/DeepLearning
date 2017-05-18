import pygame 
import random

red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
WHITE = (255, 255, 255)


pygame.init();

size=(250,250)
screen=pygame.display.set_mode(size)
clock = pygame.time.Clock()

screen.fill(WHITE)

for k in range(0,1):
	y_off=0
	x_off=0;
	r=random.randint(0,100)
	xrand=random.randint(0,3)
	yrand=random.randint(0,3)
	for i in range(0,3):
		x_off=0
		for j in range(0,3):

			color=random.randint(0,2)
			if color == 0:
				color=red
			elif color == 1:
				color = green
			else:
				color=blue
			
			if r==0 and i == xrand and j==yrand:
				color=WHITE

			dim=(60+x_off-5, 60+y_off-5,40,40)
			pygame.draw.rect(screen, color, dim)


			x_off=x_off+60

		y_off=y_off+60


	pygame.display.flip()
	clock.tick(60)
	name="stimuliii"+".png" #enter name of stimulli
	pygame.image.save(screen,name)

#Quitting the program
flag=1
while flag:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			flag = 0


pygame.quit()
