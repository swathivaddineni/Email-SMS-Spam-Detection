import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
data.head()
data['label_num'] = data.label.map({'ham':0, 'spam':1})
data.head()
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label_num'], test_size=0.2, random_state=42)
cv = CountVectorizer(stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

model = MultinomialNB()
model.fit(X_train_cv, y_train)
y_pred = model.predict(X_test_cv)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
msgs = ["Congratulations! You've won a $1000 Walmart gift card. Go to link now!", 
        "Hey Swathi, can we meet tomorrow at college?"]
msgs_cv = cv.transform(msgs)
preds = model.predict(msgs_cv)
for msg, pred in zip(msgs, preds):
    print(f"{msg} --> {'SPAM' if pred==1 else 'HAM'}")
import joblib
joblib.dump(model, 'spam_model.pkl')
joblib.dump(cv, 'vectorizer.pkl')
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print("Model trained successfully with high accuracy.")
print("The model can now classify unseen messages as SPAM or HAM.")

