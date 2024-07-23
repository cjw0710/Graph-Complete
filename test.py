import torch
import random
import numpy as np
import os
import dgl
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATv2Conv as GATConv
from dgl.data import citation_graph as citegrh

# Set random seeds for reproducibility
seed = 42  # You can use any seed value
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Load the PubMed dataset
data = citegrh.load_citeseer()

# Load learned embeddings
embeds = torch.load('embeds.pt')
print(f'embeds: {len(embeds)}')
embeds_cpu = embeds.cpu().detach().numpy()

# Calculate cosine similarity matrix
# cosine_sim_matrix = cosine_similarity(embeds_cpu, embeds_cpu)

# # Get indices of node pairs with similarity greater than 0.8
# indices = np.where(cosine_sim_matrix > 0.9)

# # Extract node pairs and their corresponding similarity values
# similar_pairs = list(zip(indices[0], indices[1], cosine_sim_matrix[indices]))

# # Sort node pairs to ensure consistent node order
# sorted_similar_pairs = [tuple(sorted(pair[:2])) + (pair[2],) for pair in similar_pairs]

# # Remove duplicate records
# unique_similar_pairs = list(set(sorted_similar_pairs))

# Assume data[0].ndata['feat'] has shape (num_nodes, num_features)
num_nodes, num_features = data[0].ndata['feat'].shape

# Create a new graph, ensuring the correct number of nodes
g = dgl.DGLGraph()
g.add_nodes(num_nodes)


# Add edges from the original graph
src_nodes, dst_nodes = data[0].edges()
g.add_edges(src_nodes, dst_nodes)
num=g.number_of_edges()
# Add edges based on unique_similar_pairs
# for pair in unique_similar_pairs:
#     i, j, _ = pair
    
#     # Check if the edge already exists before adding
#     if i!=j:
#         g.add_edges([i], [j])

# print("Number of Edges After:", g.number_of_edges()-num)
g = dgl.add_self_loop(g)

# Set node features and labels
g.ndata['feat'] = torch.FloatTensor(data[0].ndata['feat'])
g.ndata['label'] = torch.LongTensor(data[0].ndata['label'])

# Define GAT model
class GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, num_heads, dropout_rate):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_size, num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = GATConv(hidden_size * num_heads, num_classes, num_heads)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features).flatten(1))
        x = self.dropout1(x)
        x = self.conv2(g, x).mean(1)
        x = self.dropout2(x)
        return x

# Initialize and train GAT model 
in_feats = data[0].ndata['feat'].shape[1]
hidden_size = 256
num_classes = len(torch.unique(data[0].ndata['label']))
num_heads = 8
dropout_rate = 0.6

# Split the dataset
train_mask = torch.BoolTensor(data[0].ndata['train_mask'])
val_mask = torch.BoolTensor(data[0].ndata['val_mask'])
test_mask = torch.BoolTensor(data[0].ndata['test_mask'])

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(g, g.ndata['feat'])
    loss = criterion(output[train_mask], g.ndata['label'][train_mask])
    loss.backward()
    optimizer.step()

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        accuracy = correct.item() / len(labels)
        return accuracy


f_f_acc = []

# Train the model
for r in range(5):  
    model = GAT(in_feats, hidden_size, num_classes, num_heads, dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    experiment_test_accuracies = []  # 放到实验循环内部
    
    for experiment in range(100):  
        best_val_accuracy = 0.0
        best_epoch = 0
        early_stop_counter = 0
        max_early_stop_counter = 10
        test_accuracy = 0.0
        
        all_test_accuracies = []  # 每次实验都重新初始化
        
        for epoch in range(200):
            train(epoch)
            val_accuracy = evaluate(model, g, g.ndata['feat'], g.ndata['label'], val_mask)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                early_stop_counter = 0
                torch.save(model.state_dict(), 'best_model.pkl')
            else:
                early_stop_counter += 1
                if early_stop_counter >= max_early_stop_counter:
                    print(f"Early stopping at epoch {epoch}")
                    break

            best_model = GAT(in_feats, hidden_size, num_classes, num_heads, dropout_rate)
            best_model.load_state_dict(torch.load('best_model.pkl'))
            best_model.eval()
            
            with torch.no_grad():
                test_logits = best_model(g, g.ndata['feat'])
            
            _, test_indices = torch.max(test_logits[test_mask], dim=1)
            test_correct = torch.sum(test_indices == g.ndata['label'][test_mask])
            test_accuracy = test_correct.item() / len(g.ndata['label'][test_mask])
            all_test_accuracies.append(test_accuracy)
        
        print(f'Experiment {experiment + 1}, Test Accuracy: {np.mean(all_test_accuracies)}')
        experiment_test_accuracies.append(np.mean(all_test_accuracies))
    
    f_acc.append(np.mean(experiment_test_accuracies))

mean_f_acc = np.mean(f_acc)
std_f_acc = np.std(f_acc)

print(f'\nMean Test Accuracy: {mean_f_acc:.4f}')
print(f'Standard Deviation Test Accuracy: {std_f_acc:.4f}')


