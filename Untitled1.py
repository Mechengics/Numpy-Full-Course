#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[14]:


data=np.array([[1,2,3,4],[6,7,8,9]])


# In[15]:


data


# In[16]:


data*10


# In[10]:


data+data


# In[1]:


#If you want to figure out the dimension of the data you can use ndim ( n is for numpy and dim is the short form for dimension)
data.ndim
#The output will be 2( because the new formed data is two dimesnional.)


# In[18]:


#Those of you who have read matrix they now what a shape of a matrix is
#but for those who haven't understand rows and columns( 2 min task), after you do that, remember roasted cat. row column 
data.shape


# In[20]:


np.zeros(10)


# In[21]:


np.zeros((3,2))#If you have to specify the dimesnion you can just use this method.


# In[22]:


np.empty((3,3,3))# If you operate this, this gives garbage non zero values. this function should be only used if you intend to populate the array eith data.


# In[26]:


np.arange(10)#This is the equivalent of range in numpy.


# In[28]:


#Generally the data types for these kinds of array will be float 64


# In[29]:


#We will discuss some important numpy array functions. 


# In[30]:


#array: this is used to create arrays. You can the specify the data type. If you don't do so it will be automatically specified.


# In[6]:


#asarray: this operates to create array as the command array but if the input is alread an array it doesnot copy it. 
#for example . 
import numpy as np


# In[9]:


rangeinnumpy=np.arange(9)
rangeinnumpu


# In[22]:


#ones: produces an array which contains "1" and we should specify the shape of this array
#ones_like: when we say like here we mean that the array that we are creating that are filled with ones are just like another array.
onesinnumpy=np.ones((2,3))
onesinnumpy


# In[23]:


data=np.array([[1,2,3,4],[6,7,8,9]])
onesinnumpylike=np.ones_like(data)#this takes an example from the given data and creates an array just like the shape mentioned there.
onesinnumpylike


# In[ ]:


#zeros: (just like what we had done with ones)
#zeros_like(just likt e the previous ones)
#empty( this is just like ones and zeros but it doesnot populate the given data with zero or zero_like)
#empty_like(same thing)
#np.full(it makes the give array full just like the previous one but we should give it a data)
#np.full_like( Example is given below)


# In[31]:


np.full_like?#if you want to know abou this we can use the question mark symbol to gain details about this thing
numpyfull_like=np.full_like(data, 0.1, dtype=np.double)
numpyfull_like


# In[4]:


# eye, identity	Create a square N × N identity matrix (1s on the diagonal and 0s elsewhere)


# In[ ]:


#Numpy uses data tye felxibility so that we can interact with lower level programming languages such as c++ and fortan. 
#This feature has helped numpy to become a widely used library.
#don't mind memorizing these data types you can refer to them when necessary. 
#search on internet


# In[7]:


#you can use as type to change the data type of the given data/
#for example the data type of data is int64 you can convert it into float 64 using the command
newtype=data.astype((np.float64))


# In[20]:


#let's define the data first in order to perform the task.
import numpy as np
data=np.ones((2,3))
data


# In[21]:


#Arithmetic with numpy arrays:
#you can make different arithmetic operations using the arrays you have defined
multiplymatrix=data*data
multiplymatrix


# In[15]:


subtractmatrix=data-data
subtractmatrix


# In[22]:


inversematrix=1/data
inversematrix


# In[23]:


arr=data**2
arr


# In[24]:


data>arr


# In[ ]:


#These are simple ways from which you can do some simple arithmetic operations using numpy. 


# In[25]:


#Basic Slicing and Indexing


# In[26]:


data


# In[30]:


#Indexing: 
data[(1,0)]#remember that while indexing the index should be started with zero( This is repeated again and again so mind it.)


# In[31]:


data[1,0]=3#Changing the value of the specified location can be done this using method.
data


# In[42]:


#Lets Deal with three dimesional Array.
threedimensionalarray=np.array([[[1,2,3],[3,4,5],[5,6,7]],[[8,9,10],[11,12,13],[13,14,15]]])
#Try to write the arrangement of the given matrix. 
threedimensionalarray[0]#Gives the first array in the above three dimensional array


# In[43]:


threedimensionalarray[0,0,1]


# In[ ]:


#to copy the given matrix to the previous value we can use the copy command which functions as follows:
old_value=threedimensionalarray[0].copy()


# In[ ]:


#This way of treating numbers in 3d and 2d are not present in lists and other things of python. So we have introduced num py library in this case. 


# In[7]:


#Indexing with slices:
# say we have an array that is 
import numpy as np
slicingarrays=np.array([[1,2,3],[3,4,5],[6,7,8]])
slicingarrays[:2]#Select the first two rows of the above created matrix.
#Slice just like you index the elements. 
slicingarrays[:2,1:]#This states select first two rays and then leave first the first 2 columns and slect the other


# In[ ]:


#Some times wee need to specific columns and slice the rows
slicingarrays[1,:2]
#Refer to the book return by wemickinney if you are not clear to this point
#I hope upto this point you understand, but since the concept is little hard to grab if you don't have hands on practice on matrix, so you should refer to the original book by ws mickinney.


# In[17]:


import numpy as np
#it is not necessary that the data we calculate is always made up of 
names=np.array(["Bob","Joe","Will","Bob","Will","Joe"],dtype='<U4')
data=np.array([[4,7],[0,2],[-5,6],[0,0],[1,2],[-12,-4]])
print(names)
# suppose that for the given names we have a data such that nth row of names represent data
#when we write as 
data[names=="Bob"]
# this gives us the rows of the number of matrix data in which the bob is present in the matrix names


# In[18]:


#If you want to negate the condition you can use != or ~ (this symbol is present in the series of your number keys just infront of 1 press shift and type this)
names != "Bob"
#See the output


# In[20]:


data[~(names=="Bob")]#This gives those rows whose index is not present in the row.


# In[21]:


#When we want to combine two lists having different names we conduct it as follows:
combined1=(names=="Bob")|(names=="Will")
combined1


# In[23]:


data[combined1]#gives those lists where both are present. 


# In[30]:


#if you want to select a particular column in this list you can use the code as 
data[combined1, 1:]# This slects all the rows except first


# In[32]:


#replace the destination with the numbers, for example we want to replace all the places of bob except. the Bob field
data[data<0]=0
data


# In[33]:


#you are going to change the value of certan name than you can use the command.
data[names !="Bob"]=7
data
#The result is going to be on the changed data alright.


# In[35]:


#Let's Learn fancy indexing 
#This is the process to fill the matrix in sequence. 
arr4=np.zeros((8,4))
for i in range(8):
    arr4[i]=i
#run it baby: # It will run good.
arr4


# In[36]:


#now you want to select a particular rows and create a matrix from them what do you do?
#here is the code
arr4[[4,3,0,6]]
#that worked!


# In[37]:


#what if we use negative signs instead of the positive signs that are present here in this format
arr4[[-3,-5,-7]]
#this selects from backward. 


# In[41]:


# if you want to reshape the matrix and the desired numbers in range you can use the techinque called as reshape
#the code here follows as
arr5=np.arange(32).reshape((8,4))#8x4 = 32 so therefore range upto 3
arr5# If you are someone comming from a mathematics background who has dealt with just numbers that is less than one than the original one. 
#for example total number of spaces in the matrix using reshape command is 32 and the number there is 32,( here is the bit chanlleging analysis. bcause it is 32 it starts from 0 so total elements should be 33 but this never the case because the specified 33 is actually 32)


# In[43]:


arr5[[1,5,7,2],[0,3,1,2]]#there may be two different interpertation of what this code does
#1 is that first of all it selects the rows and then slect the columns and evaluates it to create a square matrix
#2 It collects elements by intersection principle


# In[50]:


#to operate as 1 we should conduct the process called as slicing in this
arr5[[1,5,7,2]][:,[0,3,1,2]]


# In[52]:


#THE DIFFERENCE BETWEEN FANCY INDEXING AND SLICING IS THAT IN SLICING ONLY RESULTS ARE DISPLAYED WITHOUT INTERFERING THE ORIGINAL FUNCTION
#BUT IN FANCY INDEXING THE ORIGINAL FUNCTION IS MODIFIED. 


# In[53]:


arr6=np.arange(15).reshape((3,5))
arr6.T


# In[54]:


#when multplying different forms of a given primary matrix using the np.dot
np.dot(arr6.T,arr6)


# In[57]:


arr6.T@arr6(#)


# In[58]:


arr6.swapaxes(0,1)#swaps the axis this returns the value without making a copy. 


# In[ ]:


#you can use the random number generator in python that actually generates the random numbers
#this when integrated with the numpy library can give us large datas, in very short period of time. )
#for example
samples=np.random_standard_normal(size=(4,4))


# In[63]:


rng=np.random.default_rng(seed=12345)
data=rng.standard_normal((2,3))#states the shpe of the matrix
rng
data


# In[64]:


#Numpy random number generator 
#Permutation and Shuffle has the same action but permutation changes the original data while shuffle doesnot affect the original data. 


# In[65]:


#uniform: This helps in drawing dara from the original data distribution. 
#integers: Draw random integers from low to high ranges. 
#standard_normal: draws data randomly from data whose standard deviation is 1 and mean is 0
#binomial: draw data from a binomial distribution
#normal: Draw data from the normal(Gaussian Distribution)
#beta: Draw Samples from the beta distribution( Beta distribution is such a data distribution that choses a value between 0 and 1 based on a specific model of selecting called as probability distribution function. 
#This is adjusted by two values alpha and beta
#chisquare: This is some kind of statistical calculation method. It takes time and a bunches of text to learn which sadly cannot fit in here. 
#gamma: draws samples from gamma distributin
#unifrom: Draw samples from a uniform [0, 1) distribution


# In[66]:


#Universal Functions:  Fast Element-Wise Array Functions
#These are like those python functions that can target each element and that can operate on them.


# In[68]:


#first of all lets create an array, like we have previously done. 
array1=np.arange(10)
array1
np.sqrt(array1)


# In[69]:


np.exp(array1)


# In[113]:


#Suppose we have two different values and they are x and y. 
x11=rng.standard_normal(8)
x


# In[111]:


y11=rng.standard_normal(8)
y


# In[8]:


np.maximum(x11,y11)#this will select maximum from each x and y and present them in a single array.#yowill get different result everytume 


# In[9]:


import numpy as np
rng=np.random.default_rng(seed=12345)
#Now we are going witht he modf command and this is important
arr7=rng.standard_normal(7)*5 #This command generates random numbers which are from 0 to 1 so in order to actually make them grater than 1 we use this command.)
#now we are going to seperate the decimals from the given value and in order to do that we proceed as folloes
remainder1,integer1=np.modf(arr7)
remainder1


# In[10]:


integer1


# In[17]:


#This unfunc is also used to generate results without changing the original form of the functions. 
back=np.zeros_like(arr7)
np.add(arr7, 1, out=back)#this is little bit confusing beacuse out=back or back =out is very difficult to translate but you can remeber this as, we are getting something out and that thing is back.
back


# In[ ]:


#Here are some of the important unfuncs, and getting this is very very important alright.
#abs or fabs: They are primiraly used for getting the absolute value element-wise for integer, floating-point or complex values
#sqrt: Compute the square root of each element. 
#square: Does the square of each element. 
#exp: computes the exponential of each element. 
#log, log10,log2,log1p: these are log bases and log1p means log(1+x)
#sign: Compute the sign of each element: 1 (positive),0(zero), or -1 (negative)
#ceil: Compute the ceiling of each element. (i.e the smallest integer greater than or equal to that number)
#floor: Compute the floor of each element(i.e the largest integer less than or equal to each element)
#rint: round off
#modf: Helps for returning fractional or integral part of the arrays.
#isnan: return boolean array indicating wether each value is NaN
#isfinite, isinf: return boolean values indicating each element is finite or not. 
#cos, cosh, sin, sinh, tan, tanh: Trignometric and hyperbolic functions
#arccos, arccosh arcsin.......: gives the inverse values of trigonometric functions. 
#logical_not: compute truth value of not x element wise equivalent to ~ array


# In[18]:


#Some important binary universal functions:
#add: adds corresponding elements in the array.
#subtract subtracts elements in second array from first. 
#multipy: multiply array elements. 
#divide, floor_divide: divide is to divide and floor devide removes the remainder. 
#power raise elements in first array ti powers indicated in second array
#maximum, f_max element-wise maximum: fmax ignores Nan
#minimum, fmin: element wise minimum
#mod: element wise modulus
#copysign: copy sign of values in second argument to values in first argument. 
#greater than, greater than or equal to, less than, less than or equal to, equal, not equal
#logical_and (&)
#logical_or(|)
#logical_xor(^)
#this logical is what and or gate and xor gate does if you have read semiconductors. 


# In[ ]:


#Array Oriented Programminf with Arrays: 
#If you use normal process for caluclation instead of this than you would do for loop
#but since the problem is more simplifed here some people like to call it as vectorization
#vectorized process are much more faster than those that are using the for loops


# In[22]:


#Want to create a grid in python. 
points=np.arange(-5,5,0.01)#Range is already there. from -5 to 5 and increases 0.01 
xs, ys=np.meshgrid(points,points)
z=np.sqrt(xs**2+ys**2)
z


# In[27]:


#Want to see this how this looks life. Remember that graphing and plotting and visualization will be discussed later.
import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray, extent=[-5, 5, -5, 5])
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
plt.close("all")#Remove this in order to view the graph


# In[36]:


#Expressing Conditional Logic as Array Operations:
#Suppose you have created two arrays like this 
arr8=np.arange(6)
arr9_1=np.arange(6)
arr9=arr9_1 *5
arr9


# In[39]:


#Now Specificy the conditions:
cond=np.array([True, True, False,True,False,True])
result=[(x if c else y)
        for x,y,c in zip(arr8,arr9,cond)]
result


# In[ ]:


#The above given method is ineffective due to  following 2 reasons:
#1. They cannot be used in multidimensional arrays
#2. The are ineffective when large data since it is time consuming.


# In[40]:


#We can use where insteade of this to solve the above issue. 
arr10=rng.standard_normal((4,4))
arr10
arr10>0


# In[41]:


#where can be used as:
np.where(arr10>0,2,-2)#this condition states that if arr>0 than use 2 and if not the case it uses -2
#operate the expression to see the results. 


# In[42]:


#Mathematical and Stastical Method:
arr11=rng.standard_normal((5,4))
arr11
arr11.mean()


# In[43]:


arr11.sum()


# In[44]:


arr11.sum(axis=1)


# In[45]:


arr11.sum(axis=0)


# In[46]:


#cumsum and cumprod
arr=np.array([0,1,2,4,5,6,7])
arr.cumsum()


# In[ ]:


# um	Sum of all the elements in the array or along an axis; zero-length arrays have sum 0
# mean	Arithmetic mean; invalid (returns NaN) on zero-length arrays
# std, var	Standard deviation and variance, respectively
# min, max	Minimum and maximum
# argmin, argmax	Indices of minimum and maximum elements, respectively
# cumsum	Cumulative sum of elements starting from 0
# cumprod	Cumulative product of elements starting from 1


# In[47]:


arr11=rng.standard_normal(100)


# In[50]:


(arr11>0).sum()


# In[55]:


(arr11<=0).sum()


# In[57]:


#any and all functions are the additional functions that are used for testing wether the given value is present 
#any means any is present or not
#all means either all of them is true. 
bools=np.array([False,False,True,False])
bools.any()


# In[58]:


bools.all()


# In[61]:


#Similar to the python list type we can use sort to select elements in the ascending order. 
arr12=rng.standard_normal((5,3))
arr12


# In[62]:


arr12.sort()
arr12


# In[63]:


# you can use arr12.(axis=0) or arr12.(axis=1) to sort specific column and rows respectively. 


# In[64]:


#Unique and Other Set Logic
names=np.array(["Bob","Will","Joe","Bob","Will","Joe","Joe"])
np.unique(names)#This returns the sorted unique values in the array.


# In[65]:


#Similar can be don with an array of numbers. 
#Alternative use the set type:
sorted(set(names))


# In[66]:


#But this method is not good because it returns list and we have to convert it into array again.
#numpy.in1d tests wether a set of value of one array is present in the other array.
np.in1d(names,["Bob","Joe"])


# In[68]:


#Here are the some of the important set operations:
# #unique(x)---> compute the sorted, unique elements in x
# intersect1d(x,y)---->Compute the sorted common elements in x and y
#union1d(x,y): Compute the sorted union of elements. 
#in1d(x,y): Checks wether an element of x is contained in y. And the result is checked on basis of x, that means true false .....
#Conventionally you can tell it anything. You can say checking the elements of y to be in x.
#setdiff1d(x,y) Set difference, elements in x  that are not in y
#setxorld(x,y) Set symmetric differences; elements that are in either of the arrays but not both.


# In[69]:


#We can use np.save to load and np.load to load the data
#you can use as: 
np.save("name of the file", arr12)


# In[71]:


np.load("name of the file.npy")


# In[83]:


#you can also save multiple arrays in a single file. 
np.savez("savedfile2", a=arr11,b=arr12)


# In[87]:


#you can acess the saved arrays as dictionaries in python for this you can use the method
arch=np.load("savedfile2.npz")
arch["a"]


# In[ ]:


#Linear Algebra
#multiply the matrix we can use np.dot(x,y)(multiplies matrix x and y)
#you can also use @ (I think we have already disscussed about this.)


# In[ ]:


#numpy.linalg has a standard se of matrix decomposition and things like inverse and determinant. 
from numpy.linalg import inv, qr
#Here are some important linlag functions that you can use in order to solve this issue. 
#diag: Returns the diagonal
#dot: Matrix Multiplication
#trace: Compute the sum of diagonal elements.
#det: Compute the matrix determinant
#eig: Compute eigen values and eigenvectors of a square matrix
#inv: Compute the inverse of square matrix
#pinv: Compute the moore penrose pseudoinverse of matrix
#qr: compute the qr decomposition
#svd: Compute the singular value decomposition
#solve: Solve the linear system Ax=b for x where A is square matrix
#lstsq: Compute the least-squares solution to Ax=b


# In[ ]:


#This is enough for learning numpy in todays world for the purpose of data analysis. You can move into pandas after this.
#Regards
#Ankit Sangrouls
#Mechengics

