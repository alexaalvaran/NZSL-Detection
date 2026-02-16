import pickle
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

data_dict = pickle.load(open('data.pickle', 'rb'))

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42, stratify=labels)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

