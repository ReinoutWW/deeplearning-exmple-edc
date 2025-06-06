import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_synthetic_data(n_samples=1000, n_features=4, random_state=None):
    """
    Create a synthetic binary classification dataset for learning
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of input features
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, y_train, X_test, y_test as PyTorch tensors
    """
    print("\n" + "="*50)
    print("🎲 CREATING SYNTHETIC DATASET")
    print("="*50)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    print(f"📊 Generated {n_samples} samples with {n_features} features")
    print(f"   • This is a binary classification problem (2 classes: 0 and 1)")
    print(f"   • Think of it like: 'Is this email spam (1) or not spam (0)?'")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    print(f"\n📂 Split data into:")
    print(f"   • Training set: {len(X_train)} samples (80%)")
    print(f"   • Test set: {len(X_test)} samples (20%)")
    print(f"   ℹ️  Why split? We train on one set and test on another to check if")
    print(f"      our model can generalize to new, unseen data!")
    
    # Standardize features (important for neural networks!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\n🔧 Standardized features (mean=0, std=1)")
    print(f"   ℹ️  This helps neural networks train better because:")
    print(f"      • All features are on the same scale")
    print(f"      • Prevents features with large values from dominating")
    print(f"   Example: Before scaling, feature values might range from 0.1 to 1000")
    print(f"            After scaling, they're typically between -3 and 3")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    print(f"\n✅ Data ready! Converted to PyTorch tensors")
    
    return X_train, y_train, X_test, y_test


class SimpleNeuralNetwork(nn.Module):
    """
    A simple 2-layer neural network for binary classification
    Architecture: Input -> Hidden Layer -> Output
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        
        print("\n" + "="*50)
        print("🧠 BUILDING NEURAL NETWORK")
        print("="*50)
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        print(f"📐 Network Architecture:")
        print(f"   • Input layer: {input_size} neurons (one for each feature)")
        print(f"   • Hidden layer: {hidden_size} neurons")
        print(f"   • Output layer: {output_size} neurons (one for each class)")
        print(f"\n🔗 Connections:")
        print(f"   • Layer 1: {input_size} inputs → {hidden_size} outputs = {input_size * hidden_size} weights")
        print(f"   • Layer 2: {hidden_size} inputs → {output_size} outputs = {hidden_size * output_size} weights")
        print(f"   • Total parameters: {input_size * hidden_size + hidden_size + hidden_size * output_size + output_size}")
        print(f"\n⚡ Activation function: ReLU (Rectified Linear Unit)")
        print(f"   ℹ️  ReLU(x) = max(0, x) - helps network learn non-linear patterns!")
        
    def forward(self, x):
        """
        Forward pass: how data flows through the network
        """
        # Layer 1: Linear transformation + ReLU activation
        x = self.fc1(x)
        x = self.relu(x)
        
        # Layer 2: Linear transformation (no activation for final layer)
        x = self.fc2(x)
        
        return x


def train_model(model, X_train, y_train, epochs=20, learning_rate=0.01):
    """
    Train the neural network with detailed logging
    """
    print("\n" + "="*50)
    print("🏋️ TRAINING THE NEURAL NETWORK")
    print("="*50)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"⚙️ Training Configuration:")
    print(f"   • Loss function: CrossEntropyLoss")
    print(f"     ℹ️  Measures how wrong our predictions are")
    print(f"     ℹ️  Lower loss = better predictions")
    print(f"   • Optimizer: Adam (learning rate={learning_rate})")
    print(f"     ℹ️  Adam is like a smart student that adjusts how fast it learns")
    print(f"   • Epochs: {epochs}")
    print(f"     ℹ️  One epoch = seeing all training data once")
    
    # Training loop
    print(f"\n📈 Training Progress:")
    print(f"{'Epoch':>6} | {'Loss':>10} | {'Status':>30}")
    print("-" * 50)
    
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Calculate gradients
        optimizer.step()       # Update weights
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # Determine status message based on loss
        if epoch == 0:
            status = "🚀 Starting training!"
        elif loss_value > 1.0:
            status = "😰 High loss - still learning"
        elif loss_value > 0.5:
            status = "🤔 Making progress..."
        elif loss_value > 0.3:
            status = "😊 Getting better!"
        else:
            status = "🎯 Excellent! Low loss"
        
        # Print progress for key epochs
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"{epoch+1:6d} | {loss_value:10.4f} | {status}")
    
    # Analyze training
    print("\n📊 Training Analysis:")
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"   • Initial loss: {initial_loss:.4f}")
    print(f"   • Final loss: {final_loss:.4f}")
    print(f"   • Improvement: {improvement:.1f}%")
    
    if improvement > 80:
        print(f"   ✅ EXCELLENT: Model learned very well!")
    elif improvement > 50:
        print(f"   ✅ GOOD: Model learned effectively")
    elif improvement > 20:
        print(f"   ⚠️  OKAY: Model learned something, but could be better")
    else:
        print(f"   ❌ POOR: Model didn't learn much - try more epochs or adjust parameters")
    
    # Check for overfitting indicators
    if final_loss < 0.01:
        print(f"\n⚠️  Warning: Very low training loss might indicate overfitting!")
        print(f"   ℹ️  The model might have memorized the training data")
        print(f"   💡 Tip: Check performance on test data to confirm")


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return the loss
    """
    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # Don't calculate gradients during evaluation
        outputs = model(X_test)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, y_test)
    model.train()  # Set back to training mode
    return loss.item()


def run_complete_example():
    """
    Run a complete deep learning example with explanations
    """
    print("\n" + "🌟 " * 25)
    print("🎓 DEEP LEARNING TUTORIAL FOR BEGINNERS")
    print("🌟 " * 25)
    print("\nWelcome! Let's learn how neural networks work by building one together.")
    print("We'll create a simple classifier that learns to distinguish between two types of data points.")
    
    # Create data
    X_train, y_train, X_test, y_test = create_synthetic_data(
        n_samples=1000,
        n_features=4,
        random_state=42
    )
    
    # Create model
    model = SimpleNeuralNetwork(input_size=4, hidden_size=16, output_size=2)
    
    # Train model
    train_model(model, X_train, y_train, epochs=50, learning_rate=0.01)
    
    # Evaluate model
    print("\n" + "="*50)
    print("🔍 EVALUATING MODEL PERFORMANCE")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        # Test set performance
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / len(y_test) * 100
        
        # Get loss
        criterion = nn.CrossEntropyLoss()
        test_loss = criterion(outputs, y_test).item()
    
    print(f"📊 Test Set Results:")
    print(f"   • Accuracy: {accuracy:.1f}%")
    print(f"   • Loss: {test_loss:.4f}")
    
    # Interpret results
    print(f"\n🎯 Performance Interpretation:")
    if accuracy >= 90:
        print(f"   ✅ EXCELLENT: Your model is performing very well!")
        print(f"   • It correctly classifies {accuracy:.1f}% of new, unseen examples")
    elif accuracy >= 80:
        print(f"   ✅ GOOD: Your model learned the patterns well")
        print(f"   • Room for improvement, but solid performance")
    elif accuracy >= 70:
        print(f"   ⚠️  FAIR: Your model learned something useful")
        print(f"   • Consider training longer or adjusting architecture")
    else:
        print(f"   ❌ NEEDS IMPROVEMENT: Model is struggling")
        print(f"   • Try more hidden neurons, different learning rate, or more epochs")
    
    # Key concepts summary
    print("\n" + "="*50)
    print("📚 KEY CONCEPTS YOU'VE LEARNED:")
    print("="*50)
    print("1. 📊 Data Preparation: Splitting and standardizing data")
    print("2. 🧠 Neural Network: Connected layers of artificial neurons")
    print("3. 📉 Loss Function: Measures how wrong predictions are")
    print("4. 🎯 Training: Adjusting weights to minimize loss")
    print("5. 📈 Evaluation: Testing on unseen data to check generalization")
    
    print("\n💡 WHAT MAKES A GOOD MODEL?")
    print("   • Low loss on training data (learned the patterns)")
    print("   • Good accuracy on test data (can generalize)")
    print("   • Balance between the two (not overfitting)")
    
    print("\n🚀 NEXT STEPS:")
    print("   • Try changing hidden_size (e.g., 8, 32, 64)")
    print("   • Experiment with learning_rate (e.g., 0.001, 0.1)")
    print("   • Add more layers to create a deeper network")
    print("   • Try different activation functions (e.g., Sigmoid, Tanh)")
    
    return model, accuracy, test_loss


if __name__ == "__main__":
    # Run the complete example
    model, accuracy, loss = run_complete_example() 