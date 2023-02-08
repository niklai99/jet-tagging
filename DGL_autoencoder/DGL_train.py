
import numpy as np
import torch
import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dgl.nn import NNConv, EdgeConv
from dgl.nn.pytorch import Sequential as dglSequential
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

data_path = '../data/graphdataset/'


class GraphDataset(Dataset):
    def __init__(self, data_path, n_files=None, lbl_to_load=None, transform=None):

        self.fnames    = [fname for fname in os.listdir(data_path) if fname.endswith('.dgl')]
        self.data_path = data_path

        if n_files:
            self.fnames = self.fnames[n_files]

        self.transform = transform     
        self.lbl_2_load = lbl_to_load

        self.load_graphs()


    def load_graphs(self):
        self.graphs = []

        for fname in tqdm(self.fnames, 'Reading files'):
            graphs, _ = dgl.load_graphs(os.path.join(self.data_path, fname))
            if self.lbl_2_load:
                self.graphs.extend([graph for graph in graphs if graph.ndata['labels'][0].nonzero().squeeze() in self.lbl_2_load])
            else:
                self.graphs.extend(graphs)


    def __len__(self):

        return len(self.graphs)
    

    def __getitem__(self, idx):

        x = self.graphs[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x
            

class Encoder(nn.Module):
    
    def __init__(self, latent_space_dim, ch=[256,128,64,32]):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, ch[0]),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(ch[0],ch[1]),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(ch[1],ch[2]),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(ch[2], ch[3]),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(ch[3], 7*128),
            nn.Sigmoid()
            )
        
        self.conv = dglSequential(
            NNConv(
                in_feats  = 7,  # number of node features
                out_feats = 128, # output number of node features
                edge_func = self.mlp),
            EdgeConv(128, 64, batch_norm=True),
            EdgeConv(64, 32, batch_norm=True),
            EdgeConv(32, latent_space_dim, batch_norm=True)
        )

    def forward(self, graph, n_feat=None):

        x = self.conv(graph, n_feat if n_feat else graph.ndata['f'], graph.edata['d'])
        return x
    

class Decoder(nn.Module):
    
    def __init__(self, latent_space_dim, n_feat=7):
        super().__init__()

        self.shared_path = dglSequential(
            EdgeConv(latent_space_dim, 32, batch_norm=True),
            EdgeConv(32, 64, batch_norm=True),
            EdgeConv(64, 128, batch_norm=True)
        )
        
        self.node_reconstruct = EdgeConv(128, n_feat)    # output are the reconstructed node features

        self.edge_reconstruct1 = dglSequential(
            EdgeConv(128, 32, batch_norm=True),
            EdgeConv(32,16, batch_norm=True),
            EdgeConv(16,8, batch_norm=True)
        )

        self.edge_reconstruct2 = dglSequential(
            EdgeConv(128, 32, batch_norm=True),
            EdgeConv(32,16, batch_norm=True),
            EdgeConv(16,8, batch_norm=True)
        )

        self.edge_reconstruct3 = dglSequential(
            EdgeConv(128, 32, batch_norm=True),
            EdgeConv(32,16, batch_norm=True),
            EdgeConv(16,8, batch_norm=True)
        )

    def forward(self, graph, n_feat=None):
        
        if n_feat is None:
            n_feat = graph.ndata['l']

        # shared path
        shared = self.shared_path(graph, n_feat)

        # node reconstruction
        n = self.node_reconstruct(graph, shared)
        
        # edges reconstruction
        e1 = self.edge_reconstruct1(graph, shared)
        e2 = self.edge_reconstruct2(graph, shared)
        e3 = self.edge_reconstruct3(graph, shared)

        # inner product o matmul?
        e1 = torch.inner(e1, e1)  #their elements are A_{ij}
        e2 = torch.inner(e2, e2) 
        e3 = torch.inner(e3, e3)

        return n, torch.stack([e1, e2, e3], 2)
    

def nodes_loss(output, target):

    loss_fn = torch.nn.MSELoss()
    return torch.sqrt(loss_fn(output, target))   

def edges_loss(output, target):
    #dim = int(np.sqrt(target.shape[0]))
    loss_fn = torch.nn.MSELoss(reduction='none')
    edge_loss = torch.sum(torch.sqrt(torch.mean(loss_fn(output, target), dim=[0, 1])))

    return edge_loss


### Training function
def train_epoch2(autoencoder, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder

    autoencoder.train()

    # abbiamo già definito l'optimizer nella cella precedente
    
    losses = []
    l_node = 0.3
    l_edge = 1
    pbar = tqdm(dataloader, 'Steps')
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for batch in pbar: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        
        batch = batch.to(device)

        #decode
        pred_node, pred_edge = autoencoder(batch)

        node_loss = nodes_loss(pred_node, batch.ndata['f'])

        # get target adj matrix
        sparse_coalesce = batch.adj().coalesce()
        indices = sparse_coalesce.indices().to(device)
        size = sparse_coalesce.size() + (3,)
        values = batch.edata['d']

        target_adj = torch.sparse_coo_tensor(indices, values, size).to_dense()
        edge_loss = edges_loss(pred_edge, target_adj)

        del target_adj, indices
        loss = l_node * node_loss + l_edge * edge_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.detach().cpu().numpy())
        pbar.set_postfix_str(f'loss: {np.mean(losses):.2f}')
        
    losses = np.mean(losses)
    return losses


### Training function
def val_epoch2(autoencoder, device, dataloader):
    # Set train mode for both the encoder and the decoder

    autoencoder.eval()

    # abbiamo già definito l'optimizer nella cella precedente
    
    losses = []
    l_node = 0.3
    l_edge = 1
    pbar = tqdm(dataloader, 'Val Steps')
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    with torch.no_grad():
        for batch in pbar: # with "_" we just ignore the labels (the second element of the dataloader tuple)
            
            batch = batch.to(device)

            #decode
            pred_node, pred_edge = autoencoder(batch)

            node_loss = nodes_loss(pred_node, batch.ndata['f'])

            # get target adj matrix
            sparse_coalesce = batch.adj().coalesce()
            indices = sparse_coalesce.indices().to(device)
            size = sparse_coalesce.size() + (3,)
            values = batch.edata['d']

            target_adj = torch.sparse_coo_tensor(indices, values, size).to_dense()
            edge_loss = edges_loss(pred_edge, target_adj)
            del target_adj, indices
            loss = l_node * node_loss + l_edge * edge_loss
            
            losses.append(loss.detach().cpu().numpy())
            pbar.set_postfix_str(f'loss: {np.mean(losses):.2f}')
            
    losses = np.mean(losses)
    return losses




if __name__ == '__main__':

    labels_2_load = [2, 4]
    train_files = slice(15)
    val_files   = slice(15, 20)

    train_dataset = GraphDataset(data_path, train_files, labels_2_load)
    val_dataset   = GraphDataset(data_path, val_files, labels_2_load)

    train_dataloader = GraphDataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader   = GraphDataLoader(val_dataset, batch_size=64, shuffle=True)

    ### Initialize the two networks
    encoded_space_dim = 10
    encoder = Encoder(latent_space_dim=encoded_space_dim)
    decoder = Decoder(latent_space_dim=encoded_space_dim)

    autoencoder = dglSequential(encoder, decoder)

    lr = 1e-3 # Learning rate

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    """
    usando questo bisogna fare scheduler.step(val_loss) alla fine di un epoca
    """
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        factor=0.5,
        patience=5,
        cooldown=5,
        min_lr=10e-8, # this should be the minimum value of the lr at which we stop the training
        threshold = 1e-2
        )


    encoder.to(device)
    decoder.to(device)
    autoencoder.to(device)

    # train the model
    epochs = 20
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # train for one epoch
        print('EPOCH %d' % (epoch + 1))

        train_loss = train_epoch2(
            autoencoder=autoencoder, 
            device=device, 
            dataloader=train_dataloader,
            optimizer=optimizer)
        print(f'TRAIN - EPOCH {epoch+1} - loss: {train_loss}')

        train_losses.append(train_loss)

        # evaluate the model
        val_loss = val_epoch2(
            autoencoder=autoencoder, 
            device=device, 
            dataloader=val_dataloader)
        # Print Validationloss
        print(f'VALIDATION - EPOCH {epoch+1} - loss: {val_loss}\n')

        val_losses.append(val_loss)

        # schedule the learning rate
        lr_scheduler.step(val_loss)

        if optimizer.param_groups[0]['lr'] < 1e-8:
            break

    model_name = f"_{epochs}ep64batch_wt"
    print('Saving loss history')
    np.save('trainloss_history'+model_name, np.array(train_losses))
    np.save('valloss_history'+model_name, np.array(val_losses))

    
    print('Saving state_dict')
    torch.save(autoencoder.state_dict(), 'models/autoenc_sd' + model_name + '.pkl')
    print('Other saves')
    torch.save(encoder, 'models/encoder' + model_name + '.pkl')
    torch.save(decoder, 'models/decoder' + model_name + '.pkl')

    print('All done!')


    