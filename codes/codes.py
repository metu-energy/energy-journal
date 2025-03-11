import wandb
import numpy as np
import os
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sklearn.preprocessing as preprocessing
import numpy as np
import yaml
import configparser
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from copy import deepcopy
from sklearn.metrics import r2_score

def read_config_file(file_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(file_path)
    
    api_key = config['USER_INFO']['wandb_api_key']
    task = config['USER_INFO']['task']
    path_and_name_to_model = config['USER_INFO']['path_and_name_to_model']
    project_name = config['USER_INFO']['project_name']
    wandb_entity = config['USER_INFO']['wandb_entity']
    number_of_experiments_to_run = config['USER_INFO']['number_of_experiments_to_run']
    
    return api_key, task, path_and_name_to_model, project_name, wandb_entity, number_of_experiments_to_run

api_key, task, path_and_name_to_model, project_name, wandb_entity, number_of_experiments_to_run = read_config_file()

wandb.login(key = api_key)

sweep_config = {
'method': 'random'
    }

with open("config.yaml", "r") as config_file:
    yaml_config = yaml.safe_load(config_file)

sweep_config["parameters"] = yaml_config["parameters"]

torch.manual_seed(29)
np.random.seed(29)
random.seed(29)

SCALER_HEAT = None
SCALER_IOD = None
SCALERS = {}

def set_scalers():

    df_20 = "../datasets/2020_dataset.csv" # path to 2020_dataset.csv
    df_20 = pd.read_csv(df_20, delimiter=",")

    df_50 = "../datasets/2050_dataset.csv" # path to 2050_dataset.csv
    df_50 = pd.read_csv(df_50, delimiter=",")

    df_80 = "../datasets/2080_dataset.csv" # path to 2080_dataset.csv
    df_80 = pd.read_csv(df_80, delimiter=",")

    combined = pd.concat([df_20, df_50, df_80])

    combined.drop('Year', inplace=True, axis=1)
    combined.drop('Version', inplace=True, axis=1)
    combined.drop('unit_id', inplace=True, axis=1)

    global SCALERS
    global SCALER_HEAT
    global SCALER_IOD

    IOD_scaler = preprocessing.StandardScaler()
    Heat_scaler = preprocessing.StandardScaler()

    for ind in combined:
        if ind != "Q_heating" and ind != "IOD" and ind != "verticalPos":
            combined[ind] = combined[ind].astype(float)
            scaler = preprocessing.StandardScaler()
            scaler = scaler.fit(np.array(combined[ind]).reshape(-1, 1))
            combined[ind] = scaler.transform(np.array(combined[ind]).reshape(-1, 1))
            SCALERS[ind] = deepcopy(scaler)

    SCALER_IOD = deepcopy(IOD_scaler.fit(np.array(combined["IOD"]).reshape(-1, 1)))
    SCALER_HEAT = deepcopy(Heat_scaler.fit(np.array(combined["Q_heating"]).reshape(-1, 1)))

set_scalers()

def data_prep(df, task="heat"):

    global SCALER_HEAT
    global SCALER_IOD
    global SCALERS

    for ind in df:
        df[ind] = df[ind].astype(float)

    for ind in df:
        if ind != "Q_heating" and ind != "IOD" and ind != "verticalPos":
            df[ind] = SCALERS[ind].transform(np.array(df[ind]).reshape(-1, 1))

    df["IOD"] = SCALER_IOD.transform(np.array(df["IOD"]).reshape(-1, 1))
    df["Q_heating"] = SCALER_HEAT.transform(np.array(df["Q_heating"]).reshape(-1, 1))

    x = pd.get_dummies(df["verticalPos"])
    df.drop("verticalPos", inplace=True, axis=1)

    df = pd.concat([df, x], axis=1)

    if task == "heat":
        y = df.iloc[:,[0]]
        df.drop("Annual Sum Cooling Degree Days", inplace= True, axis=1)
    elif task == "iod":
        y = df.iloc[:,[1]]
        df.drop("Annual Sum Heating Degree Days", inplace= True, axis=1)

    x = df.iloc[:,2:]

    return (x,y)

class dataset_gen(Dataset):

    def __init__(self, task="heat" ,transform=None, target_transform=None, date = 2020):

        df_20 = "../datasets/2020_dataset.csv" # path to 2020_dataset.csv
        df_20 = pd.read_csv(df_20, delimiter=",")

        df_50 = "../datasets/2050_dataset.csv" # path to 2050_dataset.csv
        df_50 = pd.read_csv(df_50, delimiter=",")

        df_80 = "../datasets/2080_dataset.csv" # path to 2080_dataset.csv
        df_80 = pd.read_csv(df_80, delimiter=",")

        df_20.drop('Year', inplace=True, axis=1)
        df_20.drop('Version', inplace=True, axis=1)
        df_20.drop('unit_id', inplace=True, axis=1)

        df_50.drop('Year', inplace=True, axis=1)
        df_50.drop('Version', inplace=True, axis=1)
        df_50.drop('unit_id', inplace=True, axis=1)

        df_80.drop('Year', inplace=True, axis=1)
        df_80.drop('Version', inplace=True, axis=1)
        df_80.drop('unit_id', inplace=True, axis=1)

        if date == 2020:
            combined = df_20
        elif date == 2050:
            combined = df_50
        elif date == 2080:
            combined = df_80
        else:
            raise("INVALID DATE")

        self.transform = transform
        self.target_transform = target_transform
        self.data = combined

        self.data = self.data.dropna(axis = "rows", how = "any")
        (self.x, self.y) = data_prep(self.data, task = task)

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, index):

        if isinstance(index, torch.Tensor):
            index = index.tolist()

        return [np.array(self.x.iloc[index].values),np.array(self.y.iloc[index])]

class Model(torch.nn.Module):

    def __init__(self, s_in = 13, s_out = 1, layer_num=32, layer_size=16):

        super().__init__()

        self.in_layer = torch.nn.Linear(s_in, layer_size)

        self.hidden_layers = torch.nn.ModuleList([])

        for _ in range(layer_num-2):
            self.hidden_layers.append(torch.nn.Linear(layer_size, layer_size))

        self.out1 = torch.nn.Linear(layer_size, 1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        x = self.in_layer(x)
        x = self.relu(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)

        y = self.out1(x)

        return y.squeeze()

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = torch.nn.MSELoss()
        eps = 1e-6
        loss = torch.sqrt(criterion(x, y)+eps)
        loss = loss.squeeze()
        return loss

def plotter(outputs, labels, task):


    if task == "iod":
        labels = SCALER_IOD.inverse_transform(labels.reshape(-1, 1))
        outputs = SCALER_IOD.inverse_transform(outputs.reshape(-1, 1))

    else:
        labels = SCALER_HEAT.inverse_transform(labels.reshape(-1, 1))
        outputs = SCALER_HEAT.inverse_transform(outputs.reshape(-1, 1))

    fig, ax = plt.subplots()

    ax.scatter(outputs, labels)

    ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], c="green", linestyle="dotted")

    ax.set_xlabel("PREDICTIONS")
    ax.set_ylabel("LABELS")

    return fig

data_20 = dataset_gen(task = task, date=2020)
data_50 = dataset_gen(task = task, date=2050)
data_80 = dataset_gen(task = task, date=2080)


test_size = int(len(data_20)*0.15)
val_size = int(test_size)
train_size = len(data_20) - test_size -val_size

train_20, test_20, val_20 = random_split(data_20, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(29))

test_20_loader = DataLoader(test_20, batch_size=len(test_20), shuffle=True, drop_last=False)
train_20_loader = DataLoader(train_20, batch_size=len(train_20), shuffle=True, drop_last=False)
val_20_loader = DataLoader(val_20, batch_size=len(val_20), shuffle=True, drop_last=False)


########################################

test_size = int(len(data_50)*0.15)
val_size = int(test_size)
train_size = len(data_50) - test_size -val_size

train_50, test_50, val_50 = random_split(data_50, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(29))

test_50_loader = DataLoader(test_50, batch_size=len(test_50), shuffle=True, drop_last=False)
train_50_loader = DataLoader(train_50, batch_size=len(train_50), shuffle=True, drop_last=False)
val_50_loader = DataLoader(val_50, batch_size=len(val_50), shuffle=True, drop_last=False)

########################################

test_size = int(len(data_80)*0.15)
val_size = int(test_size)
train_size = len(data_80) - test_size -val_size

train_80, test_80, val_80 = random_split(data_80, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(29))

test_80_loader = DataLoader(test_80, batch_size=len(test_80), shuffle=True, drop_last=False)
train_80_loader = DataLoader(train_80, batch_size=len(train_80), shuffle=True, drop_last=False)
val_80_loader = DataLoader(val_80, batch_size=len(val_80), shuffle=True, drop_last=False)


def train(config=None ,n_epochs = 100,early_stop = True, delta = 0.00002, patience = 2, batch_size=128 ):

    wandb.init(config = config)
    config = wandb.config

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader = DataLoader(torch.utils.data.ConcatDataset([train_20, train_50, train_80]), batch_size=config.batch_size ,shuffle=True)
    testloader =  DataLoader(torch.utils.data.ConcatDataset([test_20, test_50, test_80]), batch_size=len(test_20)+len(test_50)+len(test_80) ,shuffle=True)
    valloader =  DataLoader(torch.utils.data.ConcatDataset([val_20, val_50, val_80]), batch_size=len(val_20)+len(val_50)+len(val_80) ,shuffle=True)

    net = Model(s_in=27, s_out=1, layer_num=config.layer_num, layer_size=config.layer_size).to(device)

    net.train()

    #############################################

    loss_func = RMSELoss()

    #############################################

    optimizer = torch.optim.Adam(net.parameters(),lr=config.learning_rate ,weight_decay = 0.000006)

    train_loss = 0
    loss_iter = []
    loss_batch = []
    loss_epoch = []
    patience_count = 0

    for epoch in range(30):

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs.float())

            loss = loss_func(outputs, labels.float().flatten())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_iter.append(loss.item())

            R2 = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy() )

            metrics = {"RUNNING LOSS    ": loss,
                        "RUNNING R2"    : R2 ,
            "train/epoch": (i + 1 + (wandb.config.batch_size * epoch)) / wandb.config.batch_size}

            wandb.log(metrics)

        loss_epoch.append(running_loss)

        if early_stop == True:

            if len(loss_epoch) >= 2 and loss_epoch[-1] >= loss_epoch[-2] + delta:
                patience_count += 1
            else:
                patience_count = 0

            if patience_count == patience:
                print("EARLY STOPPING")
                break

        print("train_loss ", loss.item(), " epoch  ", epoch+1)
        running_loss = 0.0

    #to get everything in one step
    net.eval()

    dataiter = iter(valloader)
    inputs, labels = next(dataiter)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())

    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"val" : wandb.Image(fig)})

    val_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()
    R2_val_final = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())

    ##################################################################

    dataiter = iter(testloader)
    inputs, labels = next(dataiter)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())


    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"test" : wandb.Image(fig)})

    test_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()
    R2_test_final = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy() )

    ##################################################################
    
    dindon = torch.utils.data.ConcatDataset([train_20 ,train_50,train_80])


    trainloader_oneshot = DataLoader(dindon, batch_size=len(dindon), shuffle=False)
    dataiter = iter(trainloader_oneshot)
    inputs, labels = next(dataiter)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())

    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"train" : wandb.Image(fig)})

    train_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()
    R2_train_final = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy() )

    ##################################################################
    
    dataiter = iter(val_20_loader)
    inputs, labels = next(dataiter)


    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())

    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"val_20" : wandb.Image(fig)})

    val_20_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()

    R2_val_20 = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy() )

    ###################################################################

    dataiter = iter(val_50_loader)
    inputs, labels = next(dataiter)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())

    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"val_50" : wandb.Image(fig)})

    val_50_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()
    R2_val_50 = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy() )

    ###################################################################
    
    dataiter = iter(val_80_loader)
    inputs, labels = next(dataiter)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())

    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"val_80" : wandb.Image(fig)})

    val_80_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()
    R2_val_80 = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy() )

    ###################################################################
    
    dataiter = iter(test_20_loader)
    inputs, labels = next(dataiter)


    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())

    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"test_20" : wandb.Image(fig)})

    test_20_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()
    R2_test_20 = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
    
    ###################################################################
    
    dataiter = iter(test_50_loader)
    inputs, labels = next(dataiter)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())

    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"test_50" : wandb.Image(fig)})

    test_50_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()
    R2_test_50 = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
    
    ###################################################################
    
    dataiter = iter(test_80_loader)
    inputs, labels = next(dataiter)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())

    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"test_80" : wandb.Image(fig)})

    test_80_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()
    R2_test_80 = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
    
    ###################################################################
    
    dataiter = iter(train_20_loader)
    inputs, labels = next(dataiter)


    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())

    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"train_20" : wandb.Image(fig)})

    train_20_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()
    R2_train_20 = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
    
    ###################################################################
    
    dataiter = iter(train_50_loader)
    inputs, labels = next(dataiter)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())

    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"train_50" : wandb.Image(fig)})

    train_50_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()
    R2_train_50 = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
    
    ###################################################################
    
    dataiter = iter(train_80_loader)
    inputs, labels = next(dataiter)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs.float())

    fig = plotter(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), task)
    wandb.log({"train_80" : wandb.Image(fig)})

    train_80_loss = loss_func(outputs, labels.float().flatten()).detach().cpu().numpy()
    R2_train_80 = r2_score( labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
    
    ###################################################################

    val_metrics = { "val_20_loss" : val_20_loss,
                    "R2_VAL_20" : R2_val_20,
                   "val_50_loss" : val_50_loss,
                    "R2_VAL_50" : R2_val_50,
                   "val_80_loss" : val_80_loss,
                    "R2_VAL_80" : R2_val_80,
                    "vol_loss": val_loss,
                   "R2_val":R2_val_final
                    }

    train_metrics = { "train_20_loss": train_20_loss,
                      "train_20_R2": R2_train_20,
                      "train_50_loss": train_50_loss,
                      "train_50_R2": R2_train_50,
                      "train_80_loss": train_80_loss,
                      "train_80_R2": R2_train_80
    }

    test_metrics = {"test_loss_20" : test_20_loss,
                    "R2_20_TEST" : R2_test_20,
                    "test_loss_50" : test_50_loss,
                    "R2_50_TEST" : R2_test_50,
                    "test_loss_80" : test_80_loss,
                    "R2_80_TEST" : R2_test_80,
                    "R2_TRAIN_FINAL" : R2_train_final,
                    "train_loss" : train_loss,
                    "test_loss": test_loss,
                   "R2_test":R2_test_final
                    }

    wandb.log({**metrics, **test_metrics, **val_metrics, **train_metrics})

    wandb.finish()

    torch.save(net.state_dict(), path_and_name_to_model)

    return [net, loss_func]

sweep_id = wandb.sweep(sweep_config, project=project_name, entity=wandb_entity)

wandb.agent(sweep_id, train, count=int(number_of_experiments_to_run))
