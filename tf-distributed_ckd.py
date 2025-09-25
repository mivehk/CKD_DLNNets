import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras import Input
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler #z score generated negative values which model will lose at relu
from sklearn.preprocessing import MinMaxScaler

# Select the columns to normalize
cols_to_scale = ['egfr', 'hba1c']

# Create a copy of the original dataframe to work with
df = nockd_ckd.copy()

# Initialize the scaler
scalerMM = MinMaxScaler()
scalerSS = StandardScaler()

# Split dataframe before scaling so model won't be exposed to test dataset.
train_data, test_data = train_test_split(df, test_size=0.33, random_state=42)

# Fit on the training set to have data_min_, data_max_, mean_ and scale_ objects
#then scale testing dataset using that scaling
train_data[cols_to_scale] = scalerMM.fit_transform(train_data[cols_to_scale])
test_data[cols_to_scale] = scalerMM.transform(test_data[cols_to_scale])

print(len(train_data)) #1520
print(len(test_data)) #750

X1 = train_data.iloc[:, 0:6]
y1 = train_data.iloc[:, 6]
X2 = test_data.iloc[:, 0:6]
y2 = test_data.iloc[:, 6]


#TensorFlow distributed training strategies
'''
# 1- Mirrored strategy
strategy = tf.distribute.MirroredStrategy()
#2- Multi-worker mirrored strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()
# On worker 0
export TF_CONFIG='{
  "cluster": {
    "worker": ["host1:port", "host2:port"]
  },
  "task": {"type": "worker", "index": 0}
}'

# On worker 1
export TF_CONFIG='{
  "cluster": {
    "worker": ["host1:port", "host2:port"]
  },
  "task": {"type": "worker", "index": 1}
}'
# 3- TPU strategy
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # Automatically detects TPU
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
# 4- Parameter server strategy
strategy = tf.distribute.experimental.ParameterServerStrategy()

#declarative is compact for declaring fixed architecture in one step.
with strategy.scope():
    model = Sequential([
        Input(shape=(6,)), 
        layers.Dense(6, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(2, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

#imperative is stepwise when your architecture depends on conditions
with strategy.scope():
    model = Sequential()
    model.add(Input(shape=(6,))) 
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

'''

model = Sequential()
model.add(Input(shape=(6,))) 
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', AUC()])

model.fit(X1, y1, epochs=32, batch_size=100)

results = model.evaluate(X2, y2) 
print('Loss is : %.2f' % (results[0] * 100), '%')
print('Accuracy: %.2f' % (results[1] * 100), '%') 
print('AUC: %.2f' % (results[2] * 100), '%') 
print(results)

#saving scaling object parameters such as min/max/range in a pickle file
import joblib
joblib.dump(scalerMM, 'egfr_hba1c_minmax_scaler.pkl')

model.save('ckd_model_keras.h5')
model.save('ckd_model_keras.keras')



# Input my personal data from the NYU Langone 1/17/2020 as one test result.
# Format: [egfr, hba1c, sab, race_black, race_asian, race_other]
input_data = [[4.7, 8.7, 0, 0, 0, 0]]


# Load the saved scaler
scaler = joblib.load('egfr_hba1c_minmax_scaler.pkl')

# Apply the same MinMax scaling to egfr and hba1c
# egfr = col0 and hba1c = col1, but transform expects 2d array, so adding [0] at the end.
input_data[0][0:2] = scaler.transform([input_data[0][0:2]])[0]



model = load_model('ckd_model_keras.keras')


# Convert 2D list to 2d NumPy array because keras expects numpy array (or tensors) not python list.
input_array = np.array(input_data)

# Predict CKD probability and extract floating point scaler from predicted 2d numpy array.
prediction = model.predict(input_array)[0][0]

# Interpret the result
if prediction >= 0.5:
    result = "Likely CKD (Positive)"
else:
    result = "Unlikely CKD (Negative)"

print(f"Predicted probability of CKD: {prediction:.2f}")
print("Interpretation:", result)

testds_pred = model.predict(X2)
print(type(testds_pred )) #<class 'numpy.ndarray'>
print(len(testds_pred )) #750
testds_pred[177] #array([0.3218806], dtype=float32)
print(round(testds_pred[177][0])) #0
predict_by_round = [round(a[0]) for a in testds_pred ]
print(type(y2)) #<class 'pandas.core.series.Series'>
y2 #real test_ds outcomes were rows randomly selected by sklearn train_test_split()

for i, (j,h) in enumerate(y2.items()):
    if i==10:
        break
    print(f'Index: {j},  Value: {h}')


'''
Index: 188,  Value: 0
Index: 2083,  Value: 1
Index: 1675,  Value: 1
Index: 1089,  Value: 0
Index: 1378,  Value: 1
Index: 680,  Value: 0
Index: 548,  Value: 0
Index: 1055,  Value: 0
Index: 1116,  Value: 0
Index: 582,  Value: 0
'''

real_outcomes=[]
for j, h in y2.items():
    real_outcomes.append(h)

j=0
for i in range(len(testds_pred)):
    if predict_by_round[i]==real_outcomes[i]:
       j+=1
print(f'{j} correct predictions out of {len(testds_pred)} which is about {round(j*100/len(testds_pred))} percent')
#630 correct predictions out of 750 which is about 84 percent




firstpre = np.array(predict_by_round) ## predicttions are converted to numpy array 
com1 = confusion_matrix( real_outcomes, firstpre)
sns.heatmap(com1, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
