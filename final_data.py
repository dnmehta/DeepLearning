import gzip
import six.moves.cPickle as pickle

f=gzip.open('test_data.pkl.gz','rb')
print "...loading test_data"
l=pickle.load(f)

a=l[0][0:1000]
b=l[1][0:1000]

test_data=a,b

a=l[0][1000:2001]
b=l[1][1000:2001]

validate_data=a,b

f1=gzip.open('train_data.pkl.gz','rb')
print "..loading train_data"
train_data=pickle.load(f1)

z=train_data, validate_data, test_data


f2=open('final_data.pkl','wb')
print "...dumping"
pickle.dump(z,f2)