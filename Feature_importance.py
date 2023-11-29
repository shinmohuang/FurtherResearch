#%% 
"""# **Feature importance**"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc

rc('font', family='serif')
warnings.simplefilter(action='ignore', category=FutureWarning)

feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,df.columns)), columns=['Relative Importance','Feature'])

plt.figure(figsize=(7, 5))
sns.barplot(x="Relative Importance", y="Feature", data=feature_imp.sort_values(by="Relative Importance", ascending=False)[0:33])

plt.tight_layout()

#Create a PDF file
pdf_filename = "feature_importance.pdf"
with PdfPages(pdf_filename) as pdf:
    pdf.savefig()
# Five features
top_features5 = feature_imp[58:63]['Feature'].tolist()
top_features5 = list(map(int, top_features5))
top_features5 = [feature - 1 for feature in top_features5]
top_features5= matrix[:,top_features5]
X_train, X_test, y_train, y_test = train_test_split(top_features5, y, test_size=0.3, random_state=12)

top_features5 = feature_imp[58:63]['Feature'].tolist()
top_features5

# LGBM model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')

plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Actuals', fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.grid(False)
plt.savefig('CM_top_features5.pdf')
plt.show()
#%%
# Eight features
top_features8 = feature_imp[56:63]['Feature'].tolist()
top_features8 = list(map(int, top_features8))
top_features8 = [feature - 1 for feature in top_features8]
top_features8= matrix[:,top_features8]
X_train, X_test, y_train, y_test = train_test_split(top_features8, y, test_size=0.3, random_state=12)

top_features8 = feature_imp[56:63]['Feature'].tolist()
top_features8

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')

plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Actuals', fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.grid(False)
plt.savefig('CM_top_features7.pdf')
plt.show()

#%%
#Twelve features

top_features12 = feature_imp[52:63]['Feature'].tolist()
top_features12 = list(map(int, top_features12))
top_features12 = [feature - 1 for feature in top_features12]
top_features12= matrix[:,top_features12]
X_train, X_test, y_train, y_test = train_test_split(top_features12, y, test_size=0.3, random_state=12)

top_features12 = feature_imp[52:63]['Feature'].tolist()
top_features12

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')

plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Actuals', fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.grid(False)
plt.savefig('CM_top_features11.pdf')
plt.show()

#Sixteen features

top_features16 = feature_imp[47:63]['Feature'].tolist()
top_features16 = list(map(int, top_features16))
top_features16 = [feature - 1 for feature in top_features16]
top_features16= matrix[:,top_features16]
X_train, X_test, y_train, y_test = train_test_split(top_features16, y, test_size=0.3, random_state=12)

top_features16 = feature_imp[47:63]['Feature'].tolist()
top_features16

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')

plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Actuals', fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.grid(False)
plt.savefig('CM_top_features16.pdf')
plt.show()

#Twenty features

top_features20 = feature_imp[30:63]['Feature'].tolist()
top_features20 = list(map(int, top_features20))
top_features20 = [feature - 1 for feature in top_features20]
top_features20= matrix[:,top_features20]
X_train, X_test, y_train, y_test = train_test_split(top_features20, y, test_size=0.3, random_state=12)

top_features20 = feature_imp[30:63]['Feature'].tolist()
top_features20

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')

plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Actuals', fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.grid(False)
plt.savefig('CM_top_features33].pdf')
plt.show()

# EMG features

top_EMG=matrix[:,56:63]
X_train, X_test, y_train, y_test = train_test_split(top_EMG, y, test_size=0.3, random_state=12)

top_EMG.shape

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')

plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Actuals', fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.grid(False)
plt.savefig('CM_top_EMG.pdf')
plt.show()

# Fiber features

top_FIB=matrix[:,48:56]
X_train, X_test, y_train, y_test = train_test_split(top_FIB, y, test_size=0.3, random_state=12)

top_FIB.shape

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')

plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Actuals', fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.grid(False)
plt.savefig('CM_top_FIB.pdf')
plt.show()

# IMU (neck and wrist) features

top_IMUS=matrix[:,[12,13,16,21,36,43,44,0,4,15]]
X_train, X_test, y_train, y_test = train_test_split(top_IMUS, y, test_size=0.3, random_state=12)

top_IMUS.shape

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')

plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Actuals', fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.grid(False)
plt.savefig('CM_top_IMUS.pdf')
plt.show()

# IMU wrist features

top_IMU1=matrix[:,[13,16,21,0,4]]
X_train, X_test, y_train, y_test = train_test_split(top_IMU1, y, test_size=0.3, random_state=12)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')

plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Actuals', fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.grid(False)
plt.savefig('CM_top_IMU1.pdf')
plt.show()

# IMU neck features

top_IMU2=matrix[:,[24,36,43,44]]
X_train, X_test, y_train, y_test = train_test_split(top_IMU2, y, test_size=0.3, random_state=12)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')

plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Actuals', fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.grid(False)
plt.savefig('CM_top_IMU2.pdf')
plt.show()

# Both IMUS and Fiber features

top_IMUS_FIB=matrix[:,[12,13,16,21,36,43,44,0,4,15,49,52,55]]
X_train, X_test, y_train, y_test = train_test_split(top_IMUS_FIB, y, test_size=0.3, random_state=12)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')

plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Actuals', fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.grid(False)
plt.savefig('CM_top_IMUS_FIB.pdf')
plt.show()

#%% 