import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

import utils_nn as utils


model_args = {}
model_args['batch_size'] = 64
# learning rate is how fast it will descend
model_args['lr'] = 3.e-3
# the number of epochs is the number of times you go through the full dataset 
model_args['epochs'] = 250
# speedup the traninig
model_args['momentum'] = .8
# L@ regularization
model_args['weight_decay'] = 1.e-3
# get device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
     

train_loader, val_loader= utils.parse_dataset_classification('./generated', 
                                                             splitratio=[.7,.3],
                                                             batchsize=model_args['batch_size'], 
                                                             normalize=True,
                                                             downsample=True)

model = utils.Autoencoder(num_input_channels=1, latent_dim = 512, device=device)

#criterion = nn.TripletMarginLoss(m = 0.1)
optimizer = optim.SGD(model.parameters(), 
                      lr=model_args['lr'], 
                      momentum=model_args['momentum'],
                      weight_decay=model_args['weight_decay'])
experiment = utils.Experiments(device, model, train_loader, val_loader, optimizer)

train_loss, val_loss = experiment.run_training_ae(model_args['epochs'])
utils.plot_loss(train_loss, val_loss, 'AE model')
