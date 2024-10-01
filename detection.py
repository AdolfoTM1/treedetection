import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
from lime import lime_tabular

# Crear un conjunto de datos inventado
np.random.seed(42)

# Número de muestras
n_samples = 1000

# Generar características inventadas (solo numéricas para este ejemplo)
data = {
    'transaction_amount': np.random.lognormal(mean=3, sigma=1, size=n_samples),
    'transaction_hour': np.random.randint(0, 24, size=n_samples),
    'is_weekend': np.random.choice([0, 1], size=n_samples),
    'inusualidad': np.random.choice([0, 1], size=n_samples),
    'location': np.random.choice(['urban', 'suburban', 'rural'], size=n_samples),
    'payment_method': np.random.choice(['credit_card', 'debit_card', 'paypal'], size=n_samples),
}

# Generar variable objetivo (ROS)
data['ROS'] = (
    (data['inusualidad'] == 1) &
    ((data['transaction_amount'] > np.percentile(data['transaction_amount'], 95)) |
     (data['transaction_hour'] > 20) | (data['transaction_hour'] < 6))
).astype(int)

df = pd.DataFrame(data)

# Convertir variables categóricas a dummies (variables ficticias)
df = pd.get_dummies(df, columns=['location', 'payment_method'], drop_first=True)

# Separar características y variable objetivo
X = df.drop('ROS', axis=1)
y = df['ROS']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Utilizar LIME para la interpretación del modelo
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=['No ROS', 'ROS'],
    mode='classification'
)

# Interfaz de Streamlit
st.set_page_config(layout="wide")
st.title("Predicción de ROS con LIME")

# Inputs del usuario
transaction_amount = st.number_input("Monto de la transacción:", min_value=0.0)
transaction_hour = st.number_input("Hora de la transacción (0-23):", min_value=0, max_value=23)
is_weekend = st.selectbox("¿Es fin de semana?", ["No", "Sí"])
inusualidad = st.selectbox("¿Es una transacción inusual?", ["No", "Sí"])

location = st.selectbox("Ubicación:", ["Urbana", "Suburbana", "Rural"])
payment_method = st.selectbox("Método de pago:", ["Tarjeta de crédito", "Tarjeta de débito", "PayPal"])

# Convertir inputs a valores numéricos
is_weekend = 1 if is_weekend == "Sí" else 0
inusualidad = 1 if inusualidad == "Sí" else 0

# Convertir variables categóricas a dummies
location_dummies = pd.get_dummies(pd.Series(location), prefix='location')
payment_method_dummies = pd.get_dummies(pd.Series(payment_method), prefix='payment_method')

# Crear DataFrame con los inputs
new_data = pd.DataFrame({
    'transaction_amount': [transaction_amount],
    'transaction_hour': [transaction_hour],
    'is_weekend': [is_weekend],
    'inusualidad': [inusualidad]
})

# Agregar dummies al DataFrame
new_data = pd.concat([new_data, location_dummies, payment_method_dummies], axis=1)

# Asegurarse de que las columnas coincidan con el modelo
new_data = new_data.reindex(columns=X_train.columns, fill_value=0)

# Explicar la predicción
if st.button("Generar predicción"):
    exp = explainer.explain_instance(
        data_row=new_data.iloc[0],
        predict_fn=model.predict_proba,
        num_features=len(X_train.columns)  # Incluir todas las características en la explicación
    )
    
    # Mostrar el gráfico de LIME (más grande)
    st.components.v1.html(exp.as_html(), height=1200)
