from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

# Load dataset
data = pd.read_csv(r'C:\Users\USER\Documents\TK 3\Database Data Mining\Last\sample.csv')

# Periksa kolom
print("Kolom dalam dataset:", data.columns)

# Ekstraksi fitur dari kolom 'URL'
data['url_length'] = data['URL'].apply(len)  # Panjang URL
data['num_dots'] = data['URL'].apply(lambda x: x.count('.'))  # Jumlah titik
data['num_slashes'] = data['URL'].apply(lambda x: x.count('/'))  # Jumlah slash

# Gunakan kolom 'Label' sebagai target dan fitur hasil ekstraksi
X = data[['url_length', 'num_dots', 'num_slashes']]  # Fitur numerik
y = data['Label']

# Konversi label ke numerik jika perlu (jika label berupa teks)
if y.dtype == 'object':
    y = y.factorize()[0]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, r'C:\Users\USER\Documents\TK 3\Database Data Mining\Last\model.h5')
print("Model saved as 'model.h5'")
