import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

# ─────────────────────────────────────────
# 1. GENERATION DES DONNEES SYNTHETIQUES
# ─────────────────────────────────────────
np.random.seed(42)

lengths = np.random.randint(4, 20, 500)          # longueurs de mots de passe (4 à 19)
times   = np.exp(1.5 * lengths - 5               # relation exponentielle réelle
                 + np.random.normal(0, 0.5, 500)) # + bruit gaussien (σ=0.5)

df = pd.DataFrame({'length': lengths, 'time_to_crack': times})
df.to_csv('password_data.csv', index=False)
print("✅ Données générées:", df.shape)

# ─────────────────────────────────────────
# 2. TRANSFORMATION LOGARITHMIQUE
# ─────────────────────────────────────────
# La relation est exponentielle → on prend log pour la linéariser
# y = exp(w*x + b)  →  log(y) = w*x + b  ← relation linéaire !
df['log_time'] = np.log(df['time_to_crack'])

# ─────────────────────────────────────────
# 3. PREPARATION DES DONNEES
# ─────────────────────────────────────────
X = df[['length']].values   # feature  : longueur du mot de passe
y = df['log_time'].values   # target   : log(temps de crack)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ─────────────────────────────────────────
# 4. ENTRAINEMENT DU MODELE
# ─────────────────────────────────────────
# LinearRegression utilise l'équation normale :
# θ = (X^T X)^{-1} X^T y  (closed-form, pas de gradient descent)
model = LinearRegression()
model.fit(X_train, y_train)

w = model.coef_[0]       # poids (slope)
b = model.intercept_     # biais (intercept)

print(f"\n📐 Paramètres appris :")
print(f"   w = {w:.6f}")
print(f"   b = {b:.6f}")
print(f"   Formule : log(crack_time) = {w:.4f} × length + ({b:.4f})")

# ─────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Métriques sur le set de test :")
print(f"   R²   = {r2:.4f}  (1.0 = parfait)")
print(f"   MAE  = {mae:.4f} log-secondes")
print(f"   RMSE = {rmse:.4f} log-secondes")

# ─────────────────────────────────────────
# 6. EXEMPLE DE PREDICTION
# ─────────────────────────────────────────
print(f"\n🔍 Exemples de prédictions :")
for length in [6, 8, 10, 12, 16]:
    log_sec = w * length + b
    seconds = np.exp(log_sec)
    if seconds < 60:
        t = f"{seconds:.2f} sec"
    elif seconds < 3600:
        t = f"{seconds/60:.1f} min"
    elif seconds < 86400:
        t = f"{seconds/3600:.1f} hrs"
    elif seconds < 31536000:
        t = f"{seconds/86400:.1f} jours"
    else:
        t = f"{seconds/31536000:.1f} années"
    print(f"   {length} caractères → {t}")

# ─────────────────────────────────────────
# 7. SAUVEGARDE DES PARAMETRES
# ─────────────────────────────────────────
results = {"w": w, "b": b, "r2": r2, "mae": mae, "rmse": rmse}
with open('model_params.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Paramètres sauvegardés dans model_params.json")
