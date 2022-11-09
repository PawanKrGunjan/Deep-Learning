import numpy as np
import matplotlib.pyplot as plt
def stochastic_gradient_descent(x, y, epoch, learning_rate):
    n_features = x.shape[1]
    n_sample   = x.shape[0]
    weight     = np.ones(shape = (n_features))
    bias       = 0
    
    Cost  = []
    Epoch = []
    
    for i in range(epoch):
        random_index = np.random.randint(0,n_sample-1)
        sample_x     = x[random_index]
        sample_y     = y[random_index]
        y_pred       = np.dot(weight, sample_x.T) + bias
        
        w_Grad = -(2/n_sample) * (sample_x.T.dot(sample_y-y_pred))
        b_Grad = -(2/n_sample) * np.sum(sample_y - y_pred)

        weight = weight - learning_rate * w_Grad
        bias   = bias   - learning_rate * b_Grad
        
        
        cost = np.square(sample_y - y_pred)

        Cost.append(cost)
        Epoch.append(i)

        #if min(Cost) < cost:
            #break      
        
    result = {'w1': weight[0],'w2':weight[1], 'b':bias, 'Cost':Cost, 'Epoch':Epoch}

    return result



x =  np.array([[2,4,6,8,10,15,25,75],[25,45,65,85,105,155,255,755]])

y =  np.array([52,94,136,178,220,325, 535, 1585])

result = stochastic_gradient_descent(x/755,y/1585, epoch = 10000, learning_rate = 0.001)

print(result['w1'])
print(result['w2'])
print(result['b'])

# Graph
plt.figure(dpi = 150)

plt.plot(result['Epoch'], result['Cost'], 'g')

plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('Stochastic Gradient Descent')
plt.show()