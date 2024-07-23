import argparse
import os.path as osp
import random
from time import perf_counter as t
import numpy as np
import torch
import torch.nn.functional as F

from model import Model
from utils import knn_graph
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# def train(model, x, edge_index, kg_edge_index, optimizer):
#     model.train()
#     optimizer.zero_grad()
#     h0, h1, z1, z2 = model(x, edge_index, kg_edge_index)
#     loss = model.loss(h0, h1, z1, z2)
#     loss.backward()
#     optimizer.step()

#     return loss.item()

def train(model, x, edge_index, kg_edge_index, optimizer, batch_size):
    model.train()
    total_loss = 0
    num_batches = x.size(0) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size

        optimizer.zero_grad()
        batch_x = x[start_idx:end_idx]
        batch_edge_index = edge_index[start_idx:end_idx]
        batch_kg_edge_index = kg_edge_index[start_idx:end_idx]

        h0, h1, z1, z2 = model(batch_x, batch_edge_index, batch_kg_edge_index)
        loss = model.loss(h0, h1, z1, z2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches  # 返回平均损失


def run(data, num_epochs):
    model = Model(data.num_features, args.num_hidden, args.tau1, args.tau2, args.l1, args.l2).to(device)
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_p, weight_decay=args.wd_p, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_p, weight_decay=args.wd_p)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_p, weight_decay=args.wd_p)

    start = t()
    prev = start

    cnt_wait = 0
    best = 1e9
    best_t = 0
    patience = 20
    print('======')
    for epoch in range(1, num_epochs + 1):

        loss = train(model, data.x, data.edge_index, data.kg_edge_index, optimizer, batch_size=128)
        print("loss:",loss)
        now = t()
        if loss < best:
            best = loss
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        prev = now

        if cnt_wait == patience:
            break

    model.load_state_dict(torch.load('model.pkl'))
    embeds = model.embed(data.x, data.edge_index, data.kg_edge_index)
    torch.save(embeds, 'obgnembeds.pt')
    print('finished')
    return embeds



def get_subgraph(ratio = 50):
    dataset = PygNodePropPredDataset(name='ogbn-mag')
    data = dataset[0]
   # 根据指定的边索引抽取子图
    sub_edge_index = data.edge_index_dict[('paper', 'cites', 'paper')]
    # subsample_ratio = 1 /50
    sub_num_edges = sub_edge_index.size(1) // ratio
    sub_edge_index = sub_edge_index[:, :sub_num_edges]

    # 获取唯一的节点索引
    unique_nodes = torch.unique(sub_edge_index.reshape(-1))

    # 创建一个有效节点掩码，表示节点是否存在于x_dict中
    valid_nodes_mask = torch.zeros(data.x_dict['paper'].size(0), dtype=torch.bool)
    valid_nodes_mask[unique_nodes] = True

    # 确保sub_edge_index中的索引不超过x_dict的最大索引值
    max_index = data.x_dict['paper'].size(0) - 1
    sub_edge_index = sub_edge_index[:, (sub_edge_index[0] <= max_index) & (sub_edge_index[1] <= max_index)]

    # 应用掩码过滤出有效的边
    mask = valid_nodes_mask[sub_edge_index[0]] & valid_nodes_mask[sub_edge_index[1]]
    sub_edge_index = sub_edge_index[:, mask]

    # 获取有效边的唯一节点索引
    filtered_unique_nodes = torch.unique(sub_edge_index.reshape(-1))
    

    # 根据相同的节点索引抽取其他特征
    sub_x_paper = data.x_dict['paper'][filtered_unique_nodes]
    sub_node_year_paper = data.node_year['paper'][filtered_unique_nodes]
    sub_y_paper = data.y_dict['paper'][filtered_unique_nodes]
    sub_num_nodes_paper = filtered_unique_nodes.size(0)

  
    #R如果截断需要更新索引
    if sub_x_paper.size(0) < sub_edge_index.size(1):
         sub_edge_index = sub_edge_index[:, :sub_x_paper.size(0)]
         max_index = sub_x_paper.size(0) - 1
         sub_edge_index = sub_edge_index[:, (sub_edge_index[0] <= max_index) & (sub_edge_index[1] <= max_index)]
        
    if sub_x_paper.size(0) > sub_edge_index.size(1):
        sub_x_paper = sub_x_paper[:sub_edge_index.size(1),...]
        max_index = sub_edge_index.size(0) - 1
        sub_edge_index = sub_edge_index[:, (sub_edge_index[0] <= max_index) & (sub_edge_index[1] <= max_index)]
    


   
        
    # 创建子图
    sub_data = {
        'num_nodes_dict': {'paper': sub_num_nodes_paper},
        'edge_index_dict': {('paper', 'cites', 'paper'): sub_edge_index},
        'x_dict': {'paper': sub_x_paper},
        'node_year': {'paper': sub_node_year_paper},
        'edge_reltype': {('paper', 'cites', 'paper'): data.edge_reltype[('paper', 'cites', 'paper')][:sub_edge_index.size(1)]},
        'y_dict': {'paper': sub_y_paper}
    }
            
    
    class SubGraph():
        def __init__(self,subdata):
            self.num_nodes_dict = subdata['num_nodes_dict']
            self.edge_index_dict = sub_data['edge_index_dict']
            self.x_dict = sub_data['x_dict']
            self.node_year = sub_data['node_year']
            self.edge_reltype = sub_data['edge_reltype']
            self.y_dict = sub_data['y_dict']
            self.num_features = torch.tensor(128)
            self.x = sub_data['x_dict']['paper']
            self.edge_index = sub_data['edge_index_dict'][('paper', 'cites', 'paper')]
            self.kg_edge_index = None
    return SubGraph(sub_data)



from ogb.nodeproppred import PygNodePropPredDataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default=0)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--lr_p', type=float, default=0.001)
    parser.add_argument('--lr_m', type=float, default=0.01)
    parser.add_argument('--wd_p', type=float, default=0.0)
    parser.add_argument('--wd_m', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--tau1', type=float, default=1.1)
    parser.add_argument('--tau2', type=float, default=1.1)
    parser.add_argument('--l1', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=1.0)
    parser.add_argument('--metric', type=str, default='cosine')
    parser.add_argument('--optimizer', type=str, default='adam')
    args = parser.parse_args()

    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    np.random.seed(args.SEED)
    torch.backends.cudnn.deterministic = True

    data =  get_subgraph(ratio=50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.kg_edge_index = knn_graph(data.x, k=args.K, metric=args.metric)
    
    data.x = data.x.to(device)
    print("data_x:",data.x.shape)
    data.edge_index = data.edge_index.to(device)
    print(" data.edge_index", data.edge_index.shape)
    data.kg_edge_index = data.kg_edge_index.to(device)
    print("data.kg_edge_index",data.kg_edge_index.shape)
    run(data, args.num_epochs)
    
    print("fish!")
