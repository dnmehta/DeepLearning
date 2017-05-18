import cv2

for i in range(5001,75001):
	name="stimuliiii"+str(i)+".png"
	image = cv2.imread(name)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(name,gray_image)
	cv2.imshow('color_image',image)
	cv2.imshow('gray_image',gray_image) 
		             # Waits forever for user to press any key
	cv2.destroyAllWindows()        # Closes displayed windows
	 
	#End of Code