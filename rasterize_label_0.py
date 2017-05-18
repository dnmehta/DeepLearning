import numpy as np 
import pickle 


import cv
name="stimuliiii5001"+".png"
im=cv.LoadImageM(name)
x=np.asarray(im) 
ans=np.reshape(x,(1,-1))
for i in range(5002,7501):
	name="stimuliiii"+str(i)+".png"
	im=cv.LoadImageM(name)
	x=np.asarray(im) 
	c=np.reshape(x,(1,-1))
	ans=np.row_stack((ans,c))
	print (i)

f=open('label0_stimuli_1.pkl','wb')
pickle.dump(ans,f)






