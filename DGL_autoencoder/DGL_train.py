
import numpy as np
import torch
import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import autoencoders

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
    

def nodes_loss(output, target):

    loss_fn = torch.nn.MSELoss()
    return torch.sqrt(loss_fn(output, target))   

def edges_loss(output, target):
    #dim = int(np.sqrt(target.shape[0]))
    loss_fn = torch.nn.MSELoss(reduction='none')
    edge_loss = torch.sum(torch.sqrt(torch.mean(loss_fn(output, target), dim=[0, 1])))

    return edge_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


### Training function
def train_epoch2(autoencoder, device, dataloader, optimizer, scheduler):
    # Set train mode for both the encoder and the decoder

    autoencoder.train()

    # abbiamo già definito l'optimizer nella cella precedente
    
    losses = []
    l_node = 0.3
    l_edge = 1
    pbar = tqdm(dataloader, 'Steps')
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for batch in pbar:
        
        batch.ndata.pop('labels', None)
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
        scheduler.step()
        
        losses.append(loss.detach().cpu().item())
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
            
            losses.append(loss.detach().cpu().item())
            pbar.set_postfix_str(f'loss: {np.mean(losses):.2f}')
            
    losses = np.mean(losses)
    return losses


if __name__ == '__main__':

    labels_2_load = [0, 2, 4]
    batch_size = 40
    encoded_space_dim = 15
    epochs = 9
    e = 3
    d = 1

    labels_2_load_str = [f'{item}' for item in labels_2_load]
    model_name = f"_{epochs}ep{batch_size}batch{encoded_space_dim}ld_" + "".join(labels_2_load_str) + f"_E{e}D{d}"

    train_files = slice(12)
    val_files   = slice(12,15)

    train_dataset = GraphDataset(data_path, train_files, labels_2_load)
    val_dataset   = GraphDataset(data_path, val_files, labels_2_load)

    train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader   = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    ### Initialize the two networks
    encoder = getattr(autoencoders, f'Encoder{e}')(latent_space_dim=encoded_space_dim)
    decoder = getattr(autoencoders, f'Decoder{d}')(latent_space_dim=encoded_space_dim)

    autoencoder = dglSequential(encoder, decoder)

    print("Number of trainable parameters:", count_parameters(autoencoder))

    lr = 1e-5 # Learning rate

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    n_steps = len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, lr, 1e-3, n_steps, 2*n_steps, mode='triangular2', cycle_momentum=False)

    encoder.to(device)
    decoder.to(device)
    autoencoder.to(device)

    # train the model
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # train for one epoch
        print('EPOCH %d' % (epoch + 1))

        train_loss = train_epoch2(
            autoencoder=autoencoder, 
            device=device, 
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=lr_scheduler)
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


    print('Saving loss history')
    np.save('models/trainloss_history'+model_name, np.array(train_losses))
    np.save('models/valloss_history'+model_name, np.array(val_losses))

    
    print('Saving state_dict')
    torch.save(autoencoder.state_dict(), 'models/autoenc_sd' + model_name + '.pkl')
    print('Other saves')
    torch.save(encoder, 'models/encoder' + model_name + '.pkl')
    torch.save(decoder, 'models/decoder' + model_name + '.pkl')

    print('All done!')


    
