import matplotlib.pyplot as plt
import numpy as np

valid_loss=[]
train_loss=np.array([.9,.9,.8,.7,.6,.5,.4,.3,.2,.1])
valid_loss_=np.array([.9,.9,.88,.87,.76,.45,.44,.1,.1,.1])

train_metric=np.array([.1,.2,.2,.5,.6,.7,.8,.9,.9,.99])
valid_metric=np.array([.01,.21,.21,.54,.67,.71,.88,.98,.99,.99])

for i in valid_loss_:
    valid_loss.append(i)
    
# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

plt.plot(range(1,len(train_metric)+1),train_metric, label='Training DS')
plt.plot(range(1,len(valid_metric)+1),valid_metric,label='Validation DS')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')

font1 = {'size':20}

plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')