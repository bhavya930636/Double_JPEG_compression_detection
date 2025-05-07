import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import joblib

# === Load NPZ files ===
single_train = np.load('./generated_data/EBSF_single_train.npz')
double_train = np.load('./generated_data/EBSF_double_train.npz')
single_test = np.load('./generated_data/EBSF_single_test.npz')
double_test = np.load('./generated_data/EBSF_double_test.npz')

# === Extract feature vectors ===
X_single_train = single_train['single_vec_train'][:, :-1]
X_double_train = double_train['double_vec_train'][:, :-1]
X_single_test = single_test['single_vec_test'][:, :-1]
X_double_test = double_test['double_vec_test'][:, :-1]

# === Create labels ===
y_single_train = np.zeros(X_single_train.shape[0])
y_double_train = np.ones(X_double_train.shape[0])
y_single_test = np.zeros(X_single_test.shape[0])
y_double_test = np.ones(X_double_test.shape[0])

# === Combine training and testing data ===
X_train = np.vstack((X_single_train, X_double_train))
y_train = np.hstack((y_single_train, y_double_train))
X_test = np.vstack((X_single_test, X_double_test))
y_test = np.hstack((y_single_test, y_double_test))

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  - Single compressed: {X_single_train.shape[0]} samples")
print(f"  - Double compressed: {X_double_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"  - Single compressed: {X_single_test.shape[0]} samples")
print(f"  - Double compressed: {X_double_test.shape[0]} samples")

# === Standardize ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Basic SVM ===
print("\nTraining basic SVM model...")
svm = SVC(kernel='rbf')
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\nBasic SVM Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# === Grid Search ===
accuracy_best = accuracy
conf_matrix_best = conf_matrix
class_report_best = class_report

print(f"\nOptimized SVM Performance:")
print(f"Accuracy: {accuracy_best:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix_best)
print("\nClassification Report:")
print(class_report_best)

# === Visualize confusion matrices ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 1, 1)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
            xticklabels=['Single', 'Double'], 
            yticklabels=['Single', 'Double'])
plt.title(f'Basic SVM\nAccuracy: {accuracy:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('svm_confusion_matrices.png', dpi=300)
plt.show()


# import scipy.io
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
# import seaborn as sns

# # Load all four mat files
# single_train = scipy.io.loadmat('EBSF_single_train.mat')
# double_train = scipy.io.loadmat('EBSF_double_train.mat')
# single_test = scipy.io.loadmat('EBSF_single_test.mat')
# double_test = scipy.io.loadmat('EBSF_double_test.mat')

# # Extract feature vectors
# X_single_train = single_train['single_vec_train'][:, :-1]  # Exclude the label column
# X_double_train = double_train['double_vec_train'][:, :-1]  # Exclude the label column
# X_single_test = single_test['single_vec_test'][:, :-1]  # Exclude the label column
# X_double_test = double_test['double_vec_test'][:, :-1]  # Exclude the label column

# # Create labels (0 for single compression, 1 for double compression)
# y_single_train = np.zeros(X_single_train.shape[0])
# y_double_train = np.ones(X_double_train.shape[0])
# y_single_test = np.zeros(X_single_test.shape[0])
# y_double_test = np.ones(X_double_test.shape[0])

# # Combine training and testing data
# X_train = np.vstack((X_single_train, X_double_train))
# y_train = np.hstack((y_single_train, y_double_train))
# X_test = np.vstack((X_single_test, X_double_test))
# y_test = np.hstack((y_single_test, y_double_test))

# # Print dataset sizes
# print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
# print(f"  - Single compressed: {X_single_train.shape[0]} samples")
# print(f"  - Double compressed: {X_double_train.shape[0]} samples")
# print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
# print(f"  - Single compressed: {X_single_test.shape[0]} samples")
# print(f"  - Double compressed: {X_double_test.shape[0]} samples")

# # Standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Basic SVM classifier
# print("\nTraining basic SVM model...")
# svm = SVC(kernel='rbf')
# svm.fit(X_train_scaled, y_train)

# # Make predictions
# y_pred = svm.predict(X_test_scaled)

# # Evaluate performance
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

# print(f"\nBasic SVM Performance:")
# print(f"Accuracy: {accuracy:.4f}")
# print("\nConfusion Matrix:")
# print(conf_matrix)
# print("\nClassification Report:")
# print(class_report)

# # Hyperparameter tuning with Grid Search
# print("\nPerforming hyperparameter tuning...")
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'linear']
# }

# grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)
# grid_search.fit(X_train_scaled, y_train)

# print(f"\nBest parameters: {grid_search.best_params_}")
# print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# # Evaluate the best model
# best_svm = grid_search.best_estimator_
# y_pred_best = best_svm.predict(X_test_scaled)

# accuracy_best = accuracy_score(y_test, y_pred_best)
# conf_matrix_best = confusion_matrix(y_test, y_pred_best)
# class_report_best = classification_report(y_test, y_pred_best)

# print(f"\nOptimized SVM Performance:")
# print(f"Accuracy: {accuracy_best:.4f}")
# print("\nConfusion Matrix:")
# print(conf_matrix_best)
# print("\nClassification Report:")
# print(class_report_best)

# # Visualize confusion matrices
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
#             xticklabels=['Single', 'Double'], 
#             yticklabels=['Single', 'Double'])
# plt.title(f'Basic SVM\nAccuracy: {accuracy:.4f}')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')

# plt.subplot(1, 2, 2)
# sns.heatmap(conf_matrix_best, annot=True, cmap='Blues', fmt='d', 
#             xticklabels=['Single', 'Double'], 
#             yticklabels=['Single', 'Double'])
# plt.title(f'Optimized SVM\nAccuracy: {accuracy_best:.4f}')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')

# plt.tight_layout()
# plt.savefig('svm_confusion_matrices.png', dpi=300)
# plt.show()

# # Save trained model
# import joblib
# joblib.dump(best_svm, 'jpeg_compression_detector.pkl')
# joblib.dump(scaler, 'jpeg_compression_scaler.pkl')

# print("\nTrained model saved as 'jpeg_compression_detector.pkl'")
# print("Feature scaler saved as 'jpeg_compression_scaler.pkl'")

# # Feature importance analysis (for linear kernel only)
# if best_svm.kernel == 'linear':
#     # Get feature importance from SVM weights
#     importance = np.abs(best_svm.coef_[0])
    
#     # Plot feature importance
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(importance)), importance)
#     plt.title('Feature Importance (SVM Linear Kernel)')
#     plt.xlabel('Feature Index')
#     plt.ylabel('Absolute Coefficient Value')
#     plt.grid(alpha=0.3)
#     plt.savefig('feature_importance.png', dpi=300)
#     plt.show()
    
#     # Print top important features
#     top_indices = np.argsort(importance)[::-1]
#     print("\nTop Important Features:")
#     for i, idx in enumerate(top_indices):
#         print(f"Feature {idx}: {importance[idx]:.4f}")
#         if i >= 5:  # Show top 5 features
#             break