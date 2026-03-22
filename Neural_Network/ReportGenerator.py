import numpy as np
from Network import Network
from Activations import Sigmoid, TanH
from datetime import datetime

class NetworkEvaluator:
    def __init__(self):
        self.results = []
    
    def calculate_accuracy(self, network, data):
        """Calculate accuracy on given dataset"""
        correct = 0
        for x, y in data:
            x_reshaped = x.reshape(-1, 1)
            prediction = network.feed_forward(x_reshaped)
            predicted_class = np.argmax(prediction)
            if predicted_class == y:
                correct += 1
        return (correct / len(data)) * 100
    
    def evaluate_configuration(self, layers, lr, epochs, activation_name, activation_func):
        """Train and evaluate a specific configuration"""
        print(f"\n{'='*70}")
        print(f"Testing: {activation_name} | Layers: {layers} | LR: {lr} | Epochs: {epochs}")
        print(f"{'='*70}")
        
        try:
            # Create and train network
            network = Network(layers, lr, epochs, True, activation_func)
            network.train()
            
            # Calculate accuracies
            train_acc = self.calculate_accuracy(network, network.train_data)
            test_acc = self.calculate_accuracy(network, network.test_data)
            
            # Store results
            result = {
                'activation': activation_name,
                'layers': layers,
                'learning_rate': lr,
                'epochs': epochs,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            }
            self.results.append(result)
            
            print(f"\nResults:")
            print(f"  Training Accuracy: {train_acc:.2f}%")
            print(f"  Test Accuracy: {test_acc:.2f}%")
            print(f"  Overfitting: {train_acc - test_acc:.2f}%")
            
            return result
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return None
    
    def generate_report(self):
        """Generate a comprehensive report of all experiments"""
        print("\n" + "="*70)
        print("NEURAL NETWORK HYPERPARAMETER TUNING REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Group by activation function
        sigmoid_results = [r for r in self.results if r['activation'] == 'Sigmoid']
        tanh_results = [r for r in self.results if r['activation'] == 'TanH']
        
        # Sigmoid Results
        print("\n" + "-"*70)
        print("SIGMOID ACTIVATION FUNCTION")
        print("-"*70)
        self.print_results_table(sigmoid_results)
        
        # TanH Results
        print("\n" + "-"*70)
        print("TANH ACTIVATION FUNCTION")
        print("-"*70)
        self.print_results_table(tanh_results)
        
        # Best configurations
        print("\n" + "="*70)
        print("BEST CONFIGURATIONS")
        print("="*70)
        
        if self.results:
            best_test = max(self.results, key=lambda x: x['test_accuracy'])
            best_train = max(self.results, key=lambda x: x['train_accuracy'])
            best_generalization = min(self.results, 
                                     key=lambda x: abs(x['train_accuracy'] - x['test_accuracy']))
            
            print(f"\nBest Test Accuracy:")
            self.print_single_result(best_test)
            
            print(f"\nBest Training Accuracy:")
            self.print_single_result(best_train)
            
            print(f"\nBest Generalization (least overfitting):")
            self.print_single_result(best_generalization)
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        self.print_recommendations()
    
    def print_results_table(self, results):
        """Print results in a formatted table"""
        if not results:
            print("No results available")
            return
        
        print(f"\n{'Config':<8} {'Layers':<20} {'LR':<8} {'Epochs':<8} {'Train %':<10} {'Test %':<10} {'Overfit':<10}")
        print("-"*70)
        
        for i, r in enumerate(results, 1):
            layers_str = str(r['layers'])
            overfit = r['train_accuracy'] - r['test_accuracy']
            print(f"{i:<8} {layers_str:<20} {r['learning_rate']:<8} {r['epochs']:<8} "
                  f"{r['train_accuracy']:<10.2f} {r['test_accuracy']:<10.2f} {overfit:<10.2f}")
    
    def print_single_result(self, result):
        """Print a single result in detail"""
        print(f"  Activation: {result['activation']}")
        print(f"  Layers: {result['layers']}")
        print(f"  Learning Rate: {result['learning_rate']}")
        print(f"  Epochs: {result['epochs']}")
        print(f"  Training Accuracy: {result['train_accuracy']:.2f}%")
        print(f"  Test Accuracy: {result['test_accuracy']:.2f}%")
        print(f"  Overfitting: {result['train_accuracy'] - result['test_accuracy']:.2f}%")
    
    def print_recommendations(self):
        """Print recommendations based on results"""
        if not self.results:
            print("No results to analyze")
            return
        
        # Compare activation functions
        sigmoid_avg = np.mean([r['test_accuracy'] for r in self.results if r['activation'] == 'Sigmoid'])
        tanh_avg = np.mean([r['test_accuracy'] for r in self.results if r['activation'] == 'TanH'])
        
        print(f"\nAverage Test Accuracy:")
        print(f"  Sigmoid: {sigmoid_avg:.2f}%")
        print(f"  TanH: {tanh_avg:.2f}%")
        
        if sigmoid_avg > tanh_avg:
            print(f"\n→ Sigmoid performs better on average (+{sigmoid_avg - tanh_avg:.2f}%)")
        else:
            print(f"\n→ TanH performs better on average (+{tanh_avg - sigmoid_avg:.2f}%)")
        
        # Check for overfitting
        high_overfit = [r for r in self.results if (r['train_accuracy'] - r['test_accuracy']) > 10]
        if high_overfit:
            print(f"\n⚠ {len(high_overfit)} configuration(s) show significant overfitting (>10% gap)")
            print("  Consider: reducing model complexity, increasing training data, or regularization")


def main():
    """Run experiments with different hyperparameters"""
    evaluator = NetworkEvaluator()
    
    # Define configurations to test
    configurations = [
        # Sigmoid experiments
        {'layers': [5, 10, 3], 'lr': 0.1, 'epochs': 50, 'activation': 'Sigmoid'},
        {'layers': [5, 10, 3], 'lr': 0.5, 'epochs': 50, 'activation': 'Sigmoid'},
        {'layers': [5, 15, 8, 3], 'lr': 0.1, 'epochs': 50, 'activation': 'Sigmoid'},
        {'layers': [5, 20, 10, 3], 'lr': 0.3, 'epochs': 100, 'activation': 'Sigmoid'},
        {'layers': [5, 8, 3], 'lr': 0.2, 'epochs': 75, 'activation': 'Sigmoid'},
        
        # TanH experiments
        {'layers': [5, 10, 3], 'lr': 0.1, 'epochs': 50, 'activation': 'TanH'},
        {'layers': [5, 10, 3], 'lr': 0.5, 'epochs': 50, 'activation': 'TanH'},
        {'layers': [5, 15, 8, 3], 'lr': 0.1, 'epochs': 50, 'activation': 'TanH'},
        {'layers': [5, 20, 10, 3], 'lr': 0.3, 'epochs': 100, 'activation': 'TanH'},
        {'layers': [5, 8, 3], 'lr': 0.2, 'epochs': 75, 'activation': 'TanH'},
    ]
    
    # Run experiments
    for config in configurations:
        activation_func = Sigmoid() if config['activation'] == 'Sigmoid' else TanH()
        evaluator.evaluate_configuration(
            config['layers'],
            config['lr'],
            config['epochs'],
            config['activation'],
            activation_func
        )
    
    # Generate final report
    evaluator.generate_report()


if __name__ == "__main__":
    print("Starting Neural Network Hyperparameter Tuning...")
    print("This may take several minutes...\n")
    main()
    print("\nReport generation complete!")