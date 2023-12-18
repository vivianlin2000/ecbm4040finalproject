from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def test(model, saved_models_dir, test_generator, plots=True):
    """
    Load the best model and evaluate it on the test set.

    Args:
        saved_models_dir (str): Directory where models are saved.
        filename (str): Model filename.
        test_generator (tf.keras.utils.Sequence): Test data generator.
    """

    print("Testing on the test dataset")
    
    # Evaluate the model using the test generator
    test_results = model.evaluate(test_generator)

    print("\nTest Loss:", test_results[0])
    print("Test Accuracy:", test_results[1])

    with open(f"{saved_models_dir}/test_accuracy.txt", 'w') as file:
        file.write(f'Test Accuracy: {test_results[1]}\n')
        
    print(f"\nSaved test accuracy to {saved_models_dir}/test_accuracy.txt")
    
    if plots:
        plot_CM(model, saved_models_dir, test_generator)
        plot_ROC(model, saved_models_dir, test_generator)
    
        
def plot_CM(model, saved_models_dir, test_generator):
    print("\nPlotting confusion matrix")
    test_generator.reset()  # Reset generator to the beginning

    # Get the model predictions
    y_pred = model.predict(test_generator)

    # Extract true labels from the generator
    y_true_labels = test_generator.classes

    # Convert predictions to class labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    class_labels = list(test_generator.class_indices.keys())

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
    
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig( f"{saved_models_dir}/ConfusionMatrix.png" )
    plt.show()

    
    
def plot_ROC(model, saved_models_dir, test_generator):
    print("\nPlotting the ROC curves")
    # Assuming test_generator is created using flow_from_dataframe
    test_generator.reset()  # Reset generator to the beginning

    # Get the model predictions
    y_pred = model.predict(test_generator)

    # Extract true labels from the generator
    y_true_labels = test_generator.classes

    # Compute ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Get the number of classes
    n_classes = len(test_generator.class_indices)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_labels == i, y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig( f"{saved_models_dir}/ROC_Curve.png" )
          
    plt.show()