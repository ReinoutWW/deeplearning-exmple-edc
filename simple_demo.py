"""
Simple Deep Learning Demo - No External Libraries Required!
This script demonstrates core deep learning concepts using only Python's built-in libraries.
"""

import random
import math
import time


class SimpleNeuron:
    """A single neuron - the basic building block of neural networks"""
    
    def __init__(self, num_inputs, learning_rate=0.01):
        """Initialize a neuron with random weights"""
        # Weights: how much each input matters
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        # Bias: the neuron's tendency to activate
        self.bias = random.uniform(-1, 1)
        self.learning_rate = learning_rate
        
        print(f"🧠 Created neuron with {num_inputs} inputs")
        print(f"   • Initial weights: {[f'{w:.3f}' for w in self.weights]}")
        print(f"   • Initial bias: {self.bias:.3f}")


def sigmoid(x):
    """Activation function - converts any value to 0-1 range"""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0 if x < 0 else 1


def create_simple_dataset():
    """Create a simple AND gate dataset for learning"""
    print("\n" + "="*50)
    print("📊 CREATING SIMPLE DATASET")
    print("="*50)
    
    # AND gate truth table
    data = [
        ([0, 0], 0),  # 0 AND 0 = 0
        ([0, 1], 0),  # 0 AND 1 = 0
        ([1, 0], 0),  # 1 AND 0 = 0
        ([1, 1], 1),  # 1 AND 1 = 1
    ]
    
    print("📋 Dataset: Learning the AND gate")
    print("   • Input [0, 0] → Output: 0")
    print("   • Input [0, 1] → Output: 0")
    print("   • Input [1, 0] → Output: 0")
    print("   • Input [1, 1] → Output: 1")
    print("\n💡 This is like teaching a computer to understand:")
    print("   'Both conditions must be true for the result to be true'")
    
    return data


def forward_pass(neuron, inputs):
    """Calculate the neuron's output given inputs"""
    # Calculate weighted sum
    total = sum(w * x for w, x in zip(neuron.weights, inputs))
    total += neuron.bias
    
    # Apply activation function
    output = sigmoid(total)
    return output


def train_simple_network(neuron, data, epochs=100):
    """Train the neuron to learn the pattern"""
    print("\n" + "="*50)
    print("🏋️ TRAINING THE NEURAL NETWORK")
    print("="*50)
    print(f"⚙️ Training for {epochs} epochs")
    print(f"   ℹ️  Each epoch = showing all examples once")
    
    # Track progress
    errors_history = []
    
    for epoch in range(epochs):
        total_error = 0
        
        # Train on each example
        for inputs, target in data:
            # Forward pass
            output = forward_pass(neuron, inputs)
            
            # Calculate error
            error = target - output
            total_error += abs(error)
            
            # Backpropagation (simplified)
            # Update weights based on error
            for i in range(len(neuron.weights)):
                neuron.weights[i] += neuron.learning_rate * error * inputs[i]
            
            # Update bias
            neuron.bias += neuron.learning_rate * error
        
        # Average error for this epoch
        avg_error = total_error / len(data)
        errors_history.append(avg_error)
        
        # Print progress at key points
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            status = "🚀 Starting!" if epoch == 0 else "🎯 Training..."
            if avg_error < 0.1:
                status = "✨ Excellent!"
            elif avg_error < 0.3:
                status = "😊 Good progress!"
            
            print(f"Epoch {epoch+1:3d} | Error: {avg_error:.4f} | {status}")
    
    # Training complete
    print("\n📊 Training Complete!")
    initial_error = errors_history[0]
    final_error = errors_history[-1]
    improvement = (initial_error - final_error) / initial_error * 100
    
    print(f"   • Initial error: {initial_error:.4f}")
    print(f"   • Final error: {final_error:.4f}")
    print(f"   • Improvement: {improvement:.1f}%")
    
    if final_error < 0.1:
        print("   ✅ EXCELLENT: The network learned the pattern perfectly!")
    elif final_error < 0.3:
        print("   ✅ GOOD: The network learned well")
    else:
        print("   ⚠️  NEEDS MORE TRAINING: Consider more epochs")
    
    return neuron


def test_network(neuron, data):
    """Test the trained network"""
    print("\n" + "="*50)
    print("🔍 TESTING THE TRAINED NETWORK")
    print("="*50)
    
    correct = 0
    print("\n📋 Test Results:")
    print("Input    | Target | Predicted | Correct?")
    print("-" * 45)
    
    for inputs, target in data:
        output = forward_pass(neuron, inputs)
        predicted = 1 if output > 0.5 else 0
        is_correct = predicted == target
        
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{inputs}    | {target}      | {predicted} ({output:.3f}) | {status}")
    
    accuracy = (correct / len(data)) * 100
    print(f"\n🎯 Accuracy: {accuracy:.1f}%")
    
    if accuracy == 100:
        print("   ✨ PERFECT! The network learned the AND gate completely!")
    elif accuracy >= 75:
        print("   ✅ GOOD! The network mostly understands the pattern")
    else:
        print("   ⚠️  More training needed")
    
    # Show final weights
    print(f"\n🧠 Final Network State:")
    print(f"   • Weight for input 1: {neuron.weights[0]:.3f}")
    print(f"   • Weight for input 2: {neuron.weights[1]:.3f}")
    print(f"   • Bias: {neuron.bias:.3f}")
    print(f"\n💡 Interpretation:")
    print(f"   • Positive weights mean 'this input makes output more likely'")
    print(f"   • The network learned that BOTH inputs need positive weights")
    print(f"   • This matches the AND gate logic!")


def visualize_decision_boundary():
    """Show how the network makes decisions"""
    print("\n" + "="*50)
    print("🎨 DECISION BOUNDARY VISUALIZATION")
    print("="*50)
    print("\nHow the network sees the input space:")
    print("(Lower numbers = closer to 0, Higher = closer to 1)")
    print()
    
    # Create a simple text-based visualization
    for y in range(11):
        row = ""
        for x in range(11):
            # Normalize to 0-1 range
            input1 = x / 10
            input2 = (10 - y) / 10
            
            # Get network output
            output = forward_pass(neuron, [input1, input2])
            
            # Convert to visual representation
            if output < 0.2:
                char = "  "
            elif output < 0.4:
                char = "░░"
            elif output < 0.6:
                char = "▒▒"
            elif output < 0.8:
                char = "▓▓"
            else:
                char = "██"
            
            row += char
        
        print(f"{1.0 - y/10:.1f} |{row}|")
    
    print("    " + "-" * 22)
    print("     0.0  0.2  0.4  0.6  0.8  1.0")
    print("\n💡 Notice: High activation (██) only in top-right corner")
    print("   This is where both inputs are close to 1!")


# Run the demonstration
if __name__ == "__main__":
    print("\n" + "🌟 " * 25)
    print("🎓 DEEP LEARNING BASICS - SIMPLE DEMONSTRATION")
    print("🌟 " * 25)
    print("\nWelcome! Let's understand how neural networks learn.")
    print("We'll teach a single neuron to understand the AND gate.")
    
    # Create dataset
    data = create_simple_dataset()
    
    # Create a neuron
    print("\n" + "="*50)
    print("🏗️ BUILDING THE NEURAL NETWORK")
    print("="*50)
    neuron = SimpleNeuron(num_inputs=2, learning_rate=0.5)
    
    # Wait a moment for effect
    time.sleep(1)
    
    # Train the network
    train_simple_network(neuron, data, epochs=100)
    
    # Test the network
    test_network(neuron, data)
    
    # Visualize decision boundary
    visualize_decision_boundary()
    
    # Summary
    print("\n" + "="*50)
    print("📚 KEY CONCEPTS YOU'VE LEARNED:")
    print("="*50)
    print("1. 🧠 NEURONS: Basic units with weights and bias")
    print("2. 📊 TRAINING DATA: Examples to learn from")
    print("3. ➡️ FORWARD PASS: Computing predictions")
    print("4. 📉 ERROR: Difference between prediction and target")
    print("5. 🔄 BACKPROPAGATION: Adjusting weights based on error")
    print("6. 📈 EPOCHS: Complete passes through the data")
    print("7. 🎯 TESTING: Checking if learning worked")
    
    print("\n💡 DEEP LEARNING IN A NUTSHELL:")
    print("   • Start with random weights")
    print("   • Make predictions")
    print("   • Measure how wrong they are")
    print("   • Adjust weights to be less wrong")
    print("   • Repeat until predictions are good!")
    
    print("\n🚀 NEXT STEPS:")
    print("   • Try changing the learning rate (currently 0.5)")
    print("   • Implement OR, XOR gates")
    print("   • Add more neurons for complex patterns")
    print("   • Stack layers to create 'deep' networks")
    
    print("\n✨ Congratulations! You've understood the core of deep learning!") 