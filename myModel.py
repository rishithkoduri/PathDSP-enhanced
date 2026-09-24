"""
Feedforward model construct
    number of hidden layers:5

    neural units of hidden layers: [2000, 1000, 800, 500, 100]
    activation function: elu
"""

import torch as tch

class FNN(tch.nn.Module):
    def __init__(self, n_inputs):
        # call constructors from superclass
        super(FNN, self).__init__()
       
        # define network layers
        self.hidden1 = tch.nn.Linear(n_inputs, 1000)
        self.hidden2 = tch.nn.Linear(1000, 800)
        self.hidden3 = tch.nn.Linear(800, 500)
        self.hidden4 = tch.nn.Linear(500, 100)
        self.output = tch.nn.Linear(100, 1)
        
        # dropout
        self.dropout = tch.nn.Dropout(p=0.1)
        # activate
        self.fnn = tch.nn.Sequential(self.hidden1, tch.nn.ELU(), self.dropout,
                                     self.hidden2, tch.nn.ELU(), self.dropout,
                                     self.hidden3, tch.nn.ELU(), self.dropout,
                                     self.hidden4, tch.nn.ELU(), self.dropout,
                                     self.output)
    def forward(self, x):
        return self.fnn(x)

if __name__ == "__main__":
    net = FNN(756) #Feedforward_bn(100)
    print('initiating an feed forward network....')
    print('    construct=\n    {:}'.format(net))

class LateFusionFNN(tch.nn.Module):
    def __init__(self, n_drug_features, n_cell_features):
        super(LateFusionFNN, self).__init__()
        
        # Branch A: Drug Features (e.g. 1024-bit Morgan Fingerprint)
        self.drug_branch = tch.nn.Sequential(
            tch.nn.Linear(n_drug_features, 512),
            tch.nn.ELU(),
            tch.nn.Dropout(p=0.1),
            tch.nn.Linear(512, 256),
            tch.nn.ELU(),
            tch.nn.Dropout(p=0.1)
        )
        
        # Branch B: Cell Line Multi-omics (e.g. Gene Expression, Mutations, CNA)
        self.cell_branch = tch.nn.Sequential(
            tch.nn.Linear(n_cell_features, 1024),
            tch.nn.ELU(),
            tch.nn.Dropout(p=0.1),
            tch.nn.Linear(1024, 512),
            tch.nn.ELU(),
            tch.nn.Dropout(p=0.1)
        )
        
        # Fusion Layer: Combining latent representations
        self.fusion = tch.nn.Sequential(
            tch.nn.Linear(256 + 512, 512),
            tch.nn.ELU(),
            tch.nn.Dropout(p=0.1),
            tch.nn.Linear(512, 100),
            tch.nn.ELU(),
            tch.nn.Dropout(p=0.1),
            tch.nn.Linear(100, 1)
        )

    def forward(self, x_drug, x_cell):
        out_drug = self.drug_branch(x_drug)
        out_cell = self.cell_branch(x_cell)
        
        # Concatenate along feature dimension
        out_fused = tch.cat((out_drug, out_cell), dim=1)
        return self.fusion(out_fused)
