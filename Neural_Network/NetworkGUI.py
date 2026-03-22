import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from Network import Network
from Activations import Sigmoid, TanH
from DataLoader import DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Training Interface")
        self.root.geometry("1000x800")
        
        self.network = None
        self.loader = DataLoader()
        self.trained = False
        
        # Create main container
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        
        # Configuration Frame
        config_frame = ttk.LabelFrame(scrollable_frame, text="Network Configuration", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Activation Function
        ttk.Label(config_frame, text="Activation Function:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.activation_var = tk.StringVar(value="sigmoid")
        activation_frame = ttk.Frame(config_frame)
        activation_frame.grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(activation_frame, text="Sigmoid", variable=self.activation_var, 
                       value="sigmoid").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(activation_frame, text="TanH", variable=self.activation_var, 
                       value="tanh").pack(side=tk.LEFT, padx=5)
        
        # Layer Configuration
        ttk.Label(config_frame, text="Hidden Layers:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(config_frame, text="(e.g., 10,8 for two hidden layers)").grid(row=1, column=2, sticky=tk.W, pady=5)
        self.layers_var = tk.StringVar(value="10,8")
        ttk.Entry(config_frame, textvariable=self.layers_var, width=30).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Learning Rate
        ttk.Label(config_frame, text="Learning Rate:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.StringVar(value="0.1")
        ttk.Entry(config_frame, textvariable=self.lr_var, width=30).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Epochs
        ttk.Label(config_frame, text="Epochs:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(config_frame, textvariable=self.epochs_var, width=30).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Bias Checkbox (placeholder)
        self.bias_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Use Biases (placeholder)", 
                       variable=self.bias_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Train Button
        self.train_button = ttk.Button(config_frame, text="Train Network", command=self.train_network)
        self.train_button.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Results Frame
        results_frame = ttk.LabelFrame(scrollable_frame, text="Training Results", padding="10")
        results_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Accuracy Display
        self.accuracy_label = ttk.Label(results_frame, text="Training Accuracy: N/A\nTest Accuracy: N/A", 
                                       font=('Arial', 12, 'bold'))
        self.accuracy_label.grid(row=0, column=0, pady=10)
        
        # Confusion Matrix
        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, pady=10)
        
        # Prediction Frame
        pred_frame = ttk.LabelFrame(scrollable_frame, text="Make Prediction", padding="10")
        pred_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Feature inputs
        self.feature_vars = []
        feature_names = ["CulmenLength", "CulmenDepth", "FlipperLength", 
                        "BodyMass", "OriginLocation"]
        
        for i, name in enumerate(feature_names):
            ttk.Label(pred_frame, text=f"{name}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar()
            self.feature_vars.append((var,name))
            ttk.Entry(pred_frame, textvariable=var, width=30).grid(row=i, column=1, sticky=tk.W, pady=2)
        
        # Predict Button
        self.predict_button = ttk.Button(pred_frame, text="Predict", command=self.make_prediction, state=tk.DISABLED)
        self.predict_button.grid(row=len(feature_names), column=0, columnspan=2, pady=10)
        
        # Prediction Result
        self.prediction_label = ttk.Label(pred_frame, text="", font=('Arial', 12, 'bold'))
        self.prediction_label.grid(row=len(feature_names)+1, column=0, columnspan=2, pady=5)
        
    def parse_layers(self):
        """Parse layer configuration and ensure input=5, output=3"""
        try:
            hidden = [int(x.strip()) for x in self.layers_var.get().split(',') if x.strip()]
            layers = [5] + hidden + [3]  # Input: 5 features, Output: 3 classes
            return layers
        except:
            messagebox.showerror("Error", "Invalid layer configuration. Use comma-separated numbers.")
            return None
    
    def train_network(self):
        try:
            # Get parameters
            layers = self.parse_layers()
            if layers is None:
                return
            
            lr = float(self.lr_var.get())
            epochs = int(self.epochs_var.get())
            
            # Get activation function
            if self.activation_var.get() == "sigmoid":
                activation = Sigmoid()
            else:
                activation = TanH()
            
            # Create and train network
            self.train_button.config(state=tk.DISABLED, text="Training...")
            self.root.update()
            
            self.network = Network(layers, lr, epochs, self.bias_var.get(), activation)
            self.network.train()
            
            # Calculate accuracies
            train_acc = self.calculate_accuracy(self.network.train_data)
            test_acc = self.calculate_accuracy(self.network.test_data)
            
            # Update accuracy display
            self.accuracy_label.config(
                text=f"Training Accuracy: {train_acc:.2f}%\nTest Accuracy: {test_acc:.2f}%"
            )
            
            # Generate and display confusion matrix
            self.plot_confusion_matrix(self.network.test_data)
            
            # Enable prediction
            self.trained = True
            self.predict_button.config(state=tk.NORMAL)
            self.train_button.config(state=tk.NORMAL, text="Train Network")
            
            messagebox.showinfo("Success", "Training completed successfully!")
            
        except Exception as e:
            self.train_button.config(state=tk.NORMAL, text="Train Network")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def calculate_accuracy(self, data):
        """Calculate accuracy on given dataset"""
        correct = 0
        for x, y in data:
            x_reshaped = x.reshape(-1, 1)
            prediction = self.network.feed_forward(x_reshaped)
            predicted_class = np.argmax(prediction)
            if predicted_class == y:
                correct += 1
        return (correct / len(data)) * 100
    
    def plot_confusion_matrix(self, data):
        """Generate and display confusion matrix"""
        y_true = []
        y_pred = []
        
        for x, y in data:
            x_reshaped = x.reshape(-1, 1)
            prediction = self.network.feed_forward(x_reshaped)
            predicted_class = np.argmax(prediction)
            y_true.append(y)
            y_pred.append(predicted_class)
        
        # Create confusion matrix
        cm = np.zeros((3, 3), dtype=int)
        for true, pred in zip(y_true, y_pred):
            cm[int(true)][pred] += 1
        
        # Plot
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        species_names = self.loader.species_encoder.classes_
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=species_names, yticklabels=species_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        self.canvas.draw()
    
    def make_prediction(self):
        """Make prediction based on user input"""
        try:
            if not self.trained:
                messagebox.showwarning("Warning", "Please train the network first!")
                return
            
            # Parse features
            features = []
            for i, var in enumerate(self.feature_vars):
                value = var[0].get().strip()
                if not value:
                    messagebox.showwarning("Warning", "Please fill all feature fields!")
                    return
                
                # Handle location encoding (last feature)
                if i == 4:
                    try:
                        # Try to encode as location name
                        encoded = self.loader.location_encoder.transform([value])[0]
                        features.append(encoded)
                    except:
                        # If fails, assume it's already a number
                        features.append(float(value))
                else:
                    features.append( (float(value)- self.loader.means.get(var[1])) / self.loader.stds.get(var[1]))
            
            # Make prediction

            x = np.array(features).reshape(-1, 1)
            prediction = self.network.feed_forward(x)
            predicted_class = np.argmax(prediction)
            
            # Decode species name
            species_name = self.loader.species_encoder.inverse_transform([predicted_class])[0]
            confidence = prediction[predicted_class][0] * 100
            
            self.prediction_label.config(
                text=f"Predicted Species: {species_name}\nConfidence: {confidence:.2f}%"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()