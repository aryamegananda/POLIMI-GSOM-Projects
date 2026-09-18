"""
Neural Network Example: Travel Expense Fraud Detection
Predicts if a travel expense claim is fraudulent based on:
- meal_expenses, accommodation_expenses, transport_expenses
- days of travel, distance in km
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
df = pd.read_csv('travel_data.csv')
print("=== RAW DATA (first 5 rows) ===")
print(df.head())
print(f"\nTotal rows: {len(df)}")
print(f"Fraud cases: {df['is_fraud'].sum()}")
print(f"Normal cases: {(df['is_fraud'] == 0).sum()}")

# ============================================================
# STEP 2: SPLIT INTO FEATURES (X) AND LABEL (y)
# ============================================================
# X = what the model sees (the inputs)
# y = what the model predicts (fraud or not)
X = df[['meal_expenses', 'accommodation_expenses', 'transport_expenses', 'days', 'distance_km']]
y = df['is_fraud']

# ============================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================
# 80% to learn from, 20% to test on (data it has never seen)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================================
# STEP 4: SCALE THE DATA
# ============================================================
# Neural networks need scaled data (values around 0-1)
# Why? Because weights are initialized small, and large input
# values (like 900 for expenses) would dominate small ones (like 3 for days)
# Decision trees DON'T need this — they just find split points
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== BEFORE SCALING (first row) ===")
print(X_train.iloc[0].values)
print("\n=== AFTER SCALING (first row) ===")
print(X_train_scaled[0])

# ============================================================
# STEP 5: BUILD AND TRAIN THE NEURAL NETWORK
# ============================================================
nn = MLPClassifier(
    hidden_layer_sizes=(8, 4),  # 2 hidden layers: 8 neurons, then 4 neurons
    activation='relu',           # activation function (adds non-linearity)
    max_iter=1000,              # maximum training rounds
    random_state=42
)

# Architecture:
# Input layer:  5 neurons  (one per feature)
# Hidden layer 1: 8 neurons (learns combinations of features)
# Hidden layer 2: 4 neurons (learns higher-level patterns)
# Output layer: 1 neuron   (fraud probability)
#
# Total connections (weights):
# (5 × 8) + (8 × 4) + (4 × 1) = 40 + 32 + 4 = 76 weights + biases
# Even this tiny network has ~76 parameters to tune!

nn.fit(X_train_scaled, y_train)
nn_predictions = nn.predict(X_test_scaled)

print("\n=== NEURAL NETWORK RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, nn_predictions):.0%}")
print(classification_report(y_test, nn_predictions, target_names=['Normal', 'Fraud']))

# ============================================================
# STEP 6: COMPARE WITH DECISION TREE
# ============================================================
# Same data, no scaling needed
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)  # note: using UNSCALED data
tree_predictions = tree.predict(X_test)

print("=== DECISION TREE RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, tree_predictions):.0%}")
print(classification_report(y_test, tree_predictions, target_names=['Normal', 'Fraud']))

# ============================================================
# STEP 7: SEE WHAT THE TREE LEARNED (interpretable!)
# ============================================================
print("=== WHAT THE DECISION TREE LEARNED ===")
feature_names = ['meal', 'accommodation', 'transport', 'days', 'distance']
tree_rules = tree.tree_
for i in range(tree_rules.node_count):
    if tree_rules.feature[i] >= 0:
        print(f"  If {feature_names[tree_rules.feature[i]]} <= {tree_rules.threshold[i]:.1f}")

# The tree probably learned something like:
# "If meal_expenses > 700 → fraud"
# Simple, interpretable, and works great for this tabular data.
#
# The neural network might be MORE accurate on complex data,
# but you can't easily see WHY it made a decision.
# That's the black box trade-off.

# ============================================================
# STEP 8: PREDICT ON NEW DATA
# ============================================================
print("\n=== PREDICT NEW EXPENSES ===")
new_claims = pd.DataFrame({
    'meal_expenses': [50, 850],
    'accommodation_expenses': [110, 600],
    'transport_expenses': [30, 450],
    'days': [3, 1],
    'distance_km': [200, 40]
})

new_scaled = scaler.transform(new_claims)
nn_result = nn.predict(new_scaled)
tree_result = tree.predict(new_claims)

for i, row in new_claims.iterrows():
    print(f"\nClaim: meals={row['meal_expenses']}, hotel={row['accommodation_expenses']}, "
          f"transport={row['transport_expenses']}, days={row['days']}, distance={row['distance_km']}km")
    print(f"  Neural Network says: {'FRAUD' if nn_result[i] else 'Normal'}")
    print(f"  Decision Tree says:  {'FRAUD' if tree_result[i] else 'Normal'}")
