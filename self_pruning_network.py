import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import time

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights and bias
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Gate scores, same shape as weight
        self.gate_scores = nn.Parameter(torch.empty((out_features, in_features)))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gate scores to positive values (e.g. 1.0) so initial gates are ~0.73
        # Allows gradients to flow initially
        nn.init.normal_(self.gate_scores, mean=1.0, std=0.1)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningNN(nn.Module):
    def __init__(self, hidden_sizes=[512, 256]):
        super(SelfPruningNN, self).__init__()
        self.flatten = nn.Flatten()
        
        self.layers = nn.ModuleList()
        in_size = 3 * 32 * 32
        for h in hidden_sizes:
            self.layers.append(PrunableLinear(in_size, h))
            in_size = h
            
        self.output_layer = PrunableLinear(in_size, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

def calculate_sparsity_loss(model):
    sparsity_loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            sparsity_loss += torch.sum(gates)
    return sparsity_loss

def calculate_sparsity_level(model, threshold=1e-2):
    total_weights = 0
    pruned_weights = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                pruned_weights += torch.sum(gates < threshold).item()
                total_weights += gates.numel()
    return (pruned_weights / total_weights) * 100.0

def get_all_gates(model):
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.extend(gates)
    return all_gates

def train_and_evaluate(lambda_val, epochs=5, device='cpu'):
    print(f"\n--- Training with lambda = {lambda_val} ---")
    
    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Using download=True in case it's not downloaded
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = SelfPruningNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            cls_loss = criterion(outputs, labels)
            
            # Add sparsity loss
            sparsity_loss = calculate_sparsity_loss(model)
            loss = cls_loss + lambda_val * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        acc = 100 * correct / total
        sparsity = calculate_sparsity_level(model)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}% | Sparsity: {sparsity:.2f}% | Time: {time.time()-start_time:.2f}s")
        
    return acc, sparsity, get_all_gates(model)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    lambdas_to_test = [0.0, 0.0001, 0.001]
    results = []
    best_lambda = None
    best_gates = None
    best_acc = 0
    best_sparsity = 0
    
    for l_val in lambdas_to_test:
        # Increase epochs for better convergence, 10 is reasonable for CIFAR10 FFN
        acc, sparsity, gates = train_and_evaluate(l_val, epochs=10, device=device)
        results.append({'lambda': l_val, 'accuracy': acc, 'sparsity': sparsity})
        
        # Save gates for best pruning model (we want decent acc and high sparsity)
        # Assuming lambda=0.001 gives the best sparsity while lambda=0.0 doesn't prune.
        if l_val == 0.001:
            best_gates = gates

    # Print summary
    print("\n=== Summary ===")
    print("Lambda | Test Accuracy | Sparsity Level (%)")
    print("-----------------------------------------")
    for res in results:
        print(f"{res['lambda']:<6} | {res['accuracy']:<13.2f} | {res['sparsity']:.2f}%")

    if best_gates is not None:
        plt.figure(figsize=(8, 5))
        plt.hist(best_gates, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Distribution of Gate Values (Lambda = 0.001)')
        plt.xlabel('Gate Value (after sigmoid)')
        plt.ylabel('Frequency')
        plt.yscale('log') # Use log scale for better visualization if spike at 0 is huge
        plt.grid(True, alpha=0.3)
        plt.savefig('gate_distribution.png')
        print("\nSaved distribution plot to gate_distribution.png")
