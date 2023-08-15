import pandas
import os
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

##imported 'torch_nockd_ckd.csv' dataset from workspace bucket

df= pandas.read_csv('torch_nockd_ckd.csv')
df.head()
train_data, test_data = train_test_split(df, test_size=0.33, random_state=42)

len(train_data) #1520
len(test_data) #750

X1 = train_data.iloc[:, 0:6]
y1 = train_data.iloc[:, 6]
X2 = test_data.iloc[:, 0:6]
y2 = test_data.iloc[:, 6]

model = Sequential()
model.add(Dense(6, input_shape=(6,), activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', AUC()])

model.fit(X1, y1, epochs=150, batch_size=100)

'''
.
.
.
16/16 [==============================] - 0s 2ms/step - loss: 0.4217 - accuracy: 0.8224 - auc: 0.8743
Epoch 146/150
16/16 [==============================] - 0s 2ms/step - loss: 0.4224 - accuracy: 0.8230 - auc: 0.8747
Epoch 147/150
16/16 [==============================] - 0s 2ms/step - loss: 0.4218 - accuracy: 0.8217 - auc: 0.8746
Epoch 148/150
16/16 [==============================] - 0s 2ms/step - loss: 0.4207 - accuracy: 0.8263 - auc: 0.8752
Epoch 149/150
16/16 [==============================] - 0s 2ms/step - loss: 0.4197 - accuracy: 0.8250 - auc: 0.8762
Epoch 150/150
16/16 [==============================] - 0s 2ms/step - loss: 0.4205 - accuracy: 0.8257 - auc: 0.8748
<keras.callbacks.History at 0x7f66b28e1f10>
'''

results = model.evaluate(X1, y1)
print('Loss is : %.2f' % (results[0] * 100), '%')
print('Accuracy: %.2f' % (results[1] * 100), '%') 
print('AUC: %.2f' % (results[2] * 100), '%') 
print(results)

'''
48/48 [==============================] - 0s 1ms/step - loss: 0.4195 - accuracy: 0.8263 - auc: 0.8761
Loss is : 41.95 %
Accuracy: 82.63 %
AUC: 87.61 %
[0.4195202887058258, 0.8263157606124878, 0.8760665059089661]
'''

predictions1 = model.predict(X2)
rounded1 = [round(a[0]) for a in predictions1]

print(type(y2)) 
#<class 'pandas.core.series.Series'>

jh2=[]
for j, h in y2.items():
    jh2.append(h)


print(type(jh2)) 
#<class 'list'>

j=0
for i in range(len(predictions1)):
    if rounded1[i]==jh2[i]:
       j+=1
print(f'{j} correct predictions out of {len(predictions1)} which is about {round(j*100/len(predictions1))} percent')
#626 correct predictions out of 750 which is about 83 percent

firstpre = np.array(rounded1) ## predicttions are converted to numpy array 
com1 = confusion_matrix( jh2, firstpre)
sns.heatmap(com1, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


