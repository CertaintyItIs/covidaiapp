#this is still in progress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('ML_ALgos/Case_Information.csv')

# Remove unnecessary columns
df = df.drop(['case_id', 'date_announced_as_removed', 'province', 'muni_city', 'region'], axis=1)

# Convert status and health_status columns to categorical variables
df['status'] = df['status'].astype('category')
df['health_status'] = df['health_status'].astype('category')

# Convert date_announced and date_recovered columns to datetime format
df['date_announced'] = pd.to_datetime(df['date_announced'])
df['date_recovered'] = pd.to_datetime(df['date_recovered'], errors='coerce')

# new column to indicate recovery 1 and 0
df['recovered'] = np.where(df['health_status'] == 'Recovered', 1,0 ) #0 for the died patient

# Split into training and testing sets
X = df[['age']]
y = df['recovered']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# HistGradientBoostingClassifier model
hgb_model = HistGradientBoostingClassifier()
hgb_model.fit(X_train, y_train)

# predictions on the test set
y_pred = hgb_model.predict(X_test)


# Calculate the count of recovered and deceased patients for each age
recovered_counts = df[df['recovered'] == 1]['age'].value_counts().sort_index()
deceased_counts = df[df['recovered'] == 0]['age'].value_counts().sort_index()

# figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the count of recovered patients for each age
ax.plot(recovered_counts.index, recovered_counts.values, label='Recovered', color='g')

# Plot the count of deceased patients for each age
ax.plot(deceased_counts.index, deceased_counts.values, label='Died', color='r')

#title and labels
ax.set_title('Age Distribution of Recovered and Died Patients')
ax.set_xlabel('Age')
ax.set_ylabel('Patient Count')

# legend
ax.legend()
#the plot
plt.show()
