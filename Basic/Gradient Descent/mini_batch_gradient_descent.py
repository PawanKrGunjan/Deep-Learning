import numpy as np
import matplotlib.pyplot as plt

def mini_batch_gradient_descent(X, y, epochs = 100, batch_size = 5, learning_rate = 0.01):
    
    n_features = x.shape[1]
    n_samples = x.shape[0]   # number of rows in X

    # numpy array with 1 row and columns equal to number of features. 
    # In our case number_of_features = 3 (area, bedroom and age)
    w = np.ones(shape=(n_features)) 
    b = 0
    
    if batch_size > n_samples: # In this case mini batch becomes same as batch gradient descent
        batch_size = n_samples
        
    Cost = []
    Epoch = []
    
    num_batches = int(n_samples/batch_size)
    
    for i in range(epochs):    
        random_indices = np.random.permutation(n_samples)
        X_tmp = x[random_indices]
        y_tmp = y[random_indices]
        
        for j in range(0,n_samples, batch_size):
            Xj = X_tmp[j:j+batch_size]
            yj = y_tmp[j:j+batch_size]
            y_predicted = np.dot(w, Xj.T) + b
            
            w_grad = -(2/len(Xj))*(Xj.T.dot(yj-y_predicted))
            b_grad = -(2/len(Xj))*np.sum(yj-y_predicted)
            
            w = w - learning_rate * w_grad
            b = b - learning_rate * b_grad
                
            cost = np.mean(np.square(yj-y_predicted)) # MSE (Mean Squared Error)
        
        if i%10==0:
            Cost.append(cost)
            Epoch.append(i)
            
    result = {'w1': w[0],'w2':w[1], 'b':b, 'Cost':Cost, 'Epoch':Epoch}

    return result


x =  np.array([[2,4,6,8,10,15,25,75],[25,45,65,85,105,155,255,755]]).T

y =  np.array([52,94,136,178,220,325, 535, 1585]).T

result = mini_batch_gradient_descent(x/755,y/1585, epochs = 5000, batch_size = 2, learning_rate = 0.00001)

print(result['w1'])
print(result['w2'])
print(result['b'])

# Graph
plt.figure(dpi = 153)

plt.plot(result['Epoch'], result['Cost'], 'g')

plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('Mini Batch Gradient Descent')
plt.show()