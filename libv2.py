import numpy as np
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def Relu(Z):
    return np.maximum(Z, 0.33)
def ReLU_deriv(Z):
    return Z > 0

def forward_prop(W1, b1, X):
    A=X.T.dot(W1)
    return A

def out_arr1(Y):
    arr_Y = np.zeros((Y.size, 10))  # Now all the elements in the matrix are zeroes
    arr_Y[np.arange(Y.size), Y] = 1 # parameters: (row,col) so it iterates thru the 
    arr_Y = arr_Y.T                 # rows as a range for row and the respective column
    return arr_Y                    # and reassigns it as 1
    
def draw(x, W1):
    arr = np.zeros(10,)
    arr[x] = 1
    A = forward_prop(W1, 2, arr)  # You need to define forward_prop function
    A = Relu(A)  # You need to define ReLU function
    df = pd.DataFrame(A.reshape((28, 28)))
    plt.gray()
    
    # Create the plot without axis
    fig, ax = plt.subplots()
    ax.imshow(df)
    ax.axis('off')  # Turn off the axis
    
    # Save the image without the axis
    plt.savefig('./static/drawings/drawing.png', bbox_inches='tight', pad_inches=0)

def drawmorethan1digit(x,W1):
    df=pd.DataFrame()
    temp=x
    while temp>0:
        x=temp%10
        temp=temp//10
        arr=np.zeros(10,)
        arr[x]=1
        A = forward_prop(W1, 2, arr)
        A=Relu(A)
        df1 = pd.DataFrame(A.reshape((28, 28)))
        df=pd.concat([df1,df],axis=1)
    plt.gray()
        # Create the plot without axis
    fig, ax = plt.subplots()
    ax.imshow(df)
    ax.axis('off')  # Turn off the axis
    
    # Save the image without the axis
    plt.savefig('./static/drawings/drawing.png', bbox_inches='tight', pad_inches=0)
    



