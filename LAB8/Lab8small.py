import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
data = pd.DataFrame({
    'cgpa':[9.2,8.5,9.0,7.5,8.2,9.1,7.8,9.3,8.4,8.6],
    'interactiveness':['yes','no','no','no','yes','yes','yes','yes','no','yes'],
    'practical_knowledge':['verygood','good','average','average','good','good','verygood','good','good','average'],
    'communication':['good','moderate','poor','good','moderate','moderate','poor','good','good','good'],
    'job_offer':['yes','yes','no','no','yes','yes','no','yes','yes','yes']
})
encoders = {}
for col in ['interactiveness','practical_knowledge','communication','job_offer']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
X = data.drop('job_offer', axis=1)
y = data['job_offer']
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)
print("Model trained successfully.")
plt.figure(figsize=(12,6))
plot_tree(model,
          feature_names=X.columns,
          class_names=encoders['job_offer'].classes_,
          filled=True,
          rounded=True)
plt.title("Decision Tree for Job Offer Prediction")
plt.show()
test = pd.DataFrame([{
    'cgpa':6.5,
    'interactiveness':'yes',
    'practical_knowledge':'good',
    'communication':'good'
}])
for col in ['interactiveness','practical_knowledge','communication']:
    test[col] = encoders[col].transform(test[col])
pred = model.predict(test)
pred = encoders['job_offer'].inverse_transform(pred)
print("Prediction Job Offer for test sample:", pred[0])
