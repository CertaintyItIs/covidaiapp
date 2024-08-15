#this is still in progress
import pandas as pd
import numpy as np
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

#accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#classification report
print(classification_report(y_test, y_pred))
