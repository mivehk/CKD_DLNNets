*<h1> Deep Neural Networks for Chronic Kidney Disease Binary Classification</h1>*</br>

   Chronic Kidney disease often has no symptoms until your kidneys are badly damaged. More than one in seven American adults, or about 37 million, are estimated to have Chronic Kidney Disease (CKD), and 40% of people with severely reduced kidney function (not on dialysis) are not aware of having CKD (CDC, 2023). The underrepresented population of seniors 75 and older with Atrial Fibrillation and Type-II diabetes was selected from the NIH All-of-Us research portal due to the higher risk of vascular outcomes with changes in their renal excretion. American College of Cardiology and American Heart Association (ACC/AHA) scoring system, CHA2DS2-VASc is a potential predictor of stroke; therefore, we investigated the relation of glycemic control with renal outcomes in those scoring for oral anticoagulation medications by this system. We looked at the correlation between simultaneous reads of the Estimated Glomerular Filtration Rate (eGFR) with Glycated Hemoglobin (HbA1c) and the association of this bivariate with the CKD diagnosis.</br>

   The qualified CHA2DS2-VASc (F>=3, M>=2) population was stratified into three groups based on the date and time of CKD diagnosis reflected in All-of-Us periodical Collection Data Reports (CDR) and measurement dates. A cohort study was designed by the principle of CKD diagnosis and the exposure to eGFR and HbA1c concurrent reads to inspect retrospective measurements’ reads recorded before and after the diagnosis dates (1283 reads in 266 NoCKD patients, 564 reads in 82 preCKD patients and 987 reads in 144 CKD patients). We denoted mean of 58.4 for eGFR across the entire population, and observed decreased eGFR values at 60 in 46.22% of noCKD cohort for diagnosis of mild kidney damage. The mean of HbA1c across three cohorts with managed type-II diabetes was equal (CKD=6.72, noCKD= 6.67, preCKD=.6.72). We used Python libraries for deep learning neural networks to evaluate the numerical values of eGFR and HbA1c; in conjunction with nominal indicator codes of race and sex at birth to make predictive analytic models that could calculate the probability of CKD as a floating number (Keras AUC= 87.61%  & PyTorch AUC= 73.8%). </br>

   we trained two deep learning neural networks using PyTorch and Keras/Tensorflow libraries for the purpose of CKD status prediction. We instantiated the Multilayer Perceptron (MLP) model class for binary classification of CKD status based on eGFR and HbA1c as two numerical variables, but we also used indicator coding for inclusion of sex at birth (Female=1, Male=0) and developed a one-hot encoding system for binary indicators of Black, Asian or Other (More than one race or unanswered) with White as the baseline category. </br> 

   The combined dataset of 2270 records was split into 1521 training records (67%) for the network to learn weights and 749 test objects (33%) for model evaluation. A fully meshed network of nodes in three layers was designed with rectified linear unit (Relu) activation function in the first two layers with “Kaiming-He” weight initialization and sigmoid activation function in the output layer to generate probability predictions. The training dataset was built from epochs of shuffled 32x6 variable batches and fed into the network by clearing the gradients of the previous training epoch for each current of Stochastic Gradient Descent (SGD) optimization, then forward passing the new epoch into the model to calculate loss using Binary Cross Entropy Loss (BCELoss) in comparison to actual values. The backpropagation of the calculated loss through the model for coefficient(weight) modifications reduced the loss with step sizes of 0.01 and momentum of 0.9 within SGD configurations. </br>
   
   The Keras model is compatible for using TensorFlow `tf.distribute` strategies for scalability on multi-GPU, multi-node, and TPU environments. </br>

```python
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC
import os
import json

# Example: Choose your data parallelism strategy among Asynchronous and Synchronous architectures
# strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.MultiWorkerMirroredStrategy()
# strategy = tf.distribute.TPUStrategy(resolver)
# strategy = tf.distribute.experimental.ParameterServerStrategy()

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = Sequential([
        Input(shape=(6,)),
        Dense(6, activation='relu'),
        Dense(4, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', AUC()]
    )


model.fit(X1, y1, epochs=150, batch_size=100)

#For MultiWorkerMirroredStrategy, define TF_CONFIG per machine to specify its role in the cluster:
# Example of multiworker
TF_CONFIG={
  "cluster": {
    "chief": ["host1:port"],
    "worker": ["host2:port", "host3:port"]
  },
  "task": {"type": "worker", "index": 0}
}
os.environ['TF_CONFIG']= json.dump(TF_CONFIG)
#print(os.environ.get('TF_CONFIG'))

#For TPU support, use Google Colab or a GCP environment with a TPU runtime. Use:

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
```