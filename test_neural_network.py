import unittest
import numpy as np
import torch
import torch.nn as nn
from neural_network import SimpleNeuralNetwork, train_model, evaluate_model, create_synthetic_data


class TestNeuralNetwork(unittest.TestCase):
    """Test suite for our beginner-friendly neural network example"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a small synthetic dataset for testing
        self.X_train, self.y_train, self.X_test, self.y_test = create_synthetic_data(
            n_samples=100, 
            n_features=4, 
            random_state=42
        )
        
    def test_synthetic_data_creation(self):
        """Test that synthetic data is created with correct shapes"""
        X_train, y_train, X_test, y_test = create_synthetic_data(
            n_samples=100, 
            n_features=4
        )
        
        # Check shapes
        self.assertEqual(X_train.shape, (80, 4))  # 80% train
        self.assertEqual(X_test.shape, (20, 4))   # 20% test
        self.assertEqual(y_train.shape, (80,))
        self.assertEqual(y_test.shape, (20,))
        
        # Check data types
        self.assertIsInstance(X_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)
        
        # Check labels are binary (0 or 1)
        self.assertTrue(torch.all((y_train == 0) | (y_train == 1)))
        
    def test_neural_network_initialization(self):
        """Test that the neural network initializes correctly"""
        model = SimpleNeuralNetwork(input_size=4, hidden_size=8, output_size=2)
        
        # Check that model has the expected layers
        self.assertIsInstance(model.fc1, nn.Linear)
        self.assertIsInstance(model.fc2, nn.Linear)
        self.assertIsInstance(model.relu, nn.ReLU)
        
        # Check layer dimensions
        self.assertEqual(model.fc1.in_features, 4)
        self.assertEqual(model.fc1.out_features, 8)
        self.assertEqual(model.fc2.in_features, 8)
        self.assertEqual(model.fc2.out_features, 2)
        
    def test_forward_pass(self):
        """Test that forward pass produces correct output shape"""
        model = SimpleNeuralNetwork(input_size=4, hidden_size=8, output_size=2)
        
        # Create sample input
        batch_size = 10
        input_tensor = torch.randn(batch_size, 4)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 2))
        
    def test_training_improves_loss(self):
        """Test that training actually improves the loss"""
        model = SimpleNeuralNetwork(input_size=4, hidden_size=8, output_size=2)
        
        # Get initial loss
        initial_loss = evaluate_model(model, self.X_test, self.y_test)
        
        # Train the model
        train_model(model, self.X_train, self.y_train, epochs=10, learning_rate=0.01)
        
        # Get final loss
        final_loss = evaluate_model(model, self.X_test, self.y_test)
        
        # Loss should decrease after training
        self.assertLess(final_loss, initial_loss)
        
    def test_model_accuracy(self):
        """Test that model can achieve reasonable accuracy on simple data"""
        model = SimpleNeuralNetwork(input_size=4, hidden_size=16, output_size=2)
        
        # Train the model with more epochs for better accuracy
        train_model(model, self.X_train, self.y_train, epochs=50, learning_rate=0.01)
        
        # Evaluate accuracy
        with torch.no_grad():
            outputs = model(self.X_test)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == self.y_test).sum().item()
            accuracy = correct / len(self.y_test)
        
        # Should achieve at least 70% accuracy on this simple dataset
        self.assertGreater(accuracy, 0.7)


if __name__ == '__main__':
    unittest.main() 