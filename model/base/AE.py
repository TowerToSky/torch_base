import torch
import torch.nn as nn

class simpleAE(nn.Module):
    def __init__(self):
        super(simpleAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(31*150,2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,2048),
            nn.ReLU(),
            nn.Linear(2048,31*150),
            nn.Sigmoid()
        )
    def forward(self, x):
        batchsz = x.size(0)
        x = x.view(batchsz, -1)
        x = self.encoder(x)

        x = self.decoder(x)

        x = x.view(batchsz, 31, 150)

        return x