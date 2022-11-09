import numpy as np
import matplotlib.pyplot as plt

def batch_gradient_descent(x,y, epoch, learning_rate):
    n_features = x.shape[1]
    n_sample   = x.shape[0]
    weight     = np.ones(shape = (n_features))
    bias       = 0
    
    Cost  = []
    Epoch = []
    
    for i in range(epoch):
        y_pred = np.dot(weight, x.T) + bias
        
        w_Grad = -(2/n_sample) * (x.T.dot(y-y_pred))
        b_Grad = -(2/n_sample) * np.sum(y-y_pred)

        weight = weight - learning_rate * w_Grad
        bias   = bias   - learning_rate * b_Grad
        
        
        cost = np.mean(np.square(y - y_pred))
        Cost.append(cost)
        Epoch.append(i)
        
        #if min(Cost) < cost:
        #    break
    result = {'w1': weight[0],'w2':weight[1], 'b':bias, 'Cost':Cost, 'Epoch':Epoch}

    return result


x =  np.array([[2,4,6,8,10,15,25,75],[25,45,65,85,105,155,255,755]]).T

y =  np.array([52,94,136,178,220,325, 535, 1585]).T

result = batch_gradient_descent(x/755,y/1585, epoch = 10000, learning_rate = 0.01)

print(result['w1'])
print(result['w2'])
print(result['b'])

# Graph
plt.figure(dpi = 153)

plt.plot(result['Epoch'], result['Cost'], 'g')

plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('Bath Gradient Descent')
plt.show()