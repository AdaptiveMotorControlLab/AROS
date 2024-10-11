import torch
from robustbench.utils import load_model
import torch.nn as nn
from torch.nn.parameter import Parameter
import utils
from utils import *
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset, SubsetRandomSampler, ConcatDataset
import numpy as np
from tqdm import tqdm


weight_diag = 10
weight_offdiag = 0
weight_f = 0.1

weight_norm = 0
weight_lossc =  0

exponent = 1.0
exponent_off = 0.1
exponent_f = 50
time_df = 1
trans = 1.0
transoffdig = 1.0
numm = 16

batches_per_epoch = 128

ODE_FC_odebatch = 64
epoch1=1
epoch2=1
epoch3=1


robust_feature_savefolder = './CIFAR10_resnet_Nov_1'
train_savepath='./CIFAR10_train_resnetNov1.npz'
test_savepath='./CIFAR10_test_resnetNov1.npz'
ODE_FC_save_folder =  './CIFAR10_resnet_Nov_1'



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ODEBlocktemp(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlocktemp, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 5]).float()
    def forward(self, x):
        out = self.odefunc(0, x)
        return out
    @property
    def nfe(self):
        return self.odefunc.nfe
    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)


        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)



def save_training_feature(model, dataset_loader, fake_embeddings_loader=None ):
    x_save = []
    y_save = []
    modulelist = list(model)

    for x, y in dataset_loader:
        x = x.to(device)
        y_ = y.numpy()  # No need to use np.array here

        # Forward pass through the model up to the desired layer
        for l in modulelist[0:2]:
            x = l(x)
        xo = x

        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)

    # Processing fake embeddings if provided
    if fake_embeddings_loader is not None:
        for x, y in fake_embeddings_loader:
            x = x.to(device)
            y_ = y.numpy()  # No need to use np.array here

            # Forward pass through the model up to the desired layer
            for l in modulelist[1:2]:
                x = l(x)
            xo = x

            x_ = xo.cpu().detach().numpy()
            x_save.append(x_)
            y_save.append(y_)

    # Concatenate all collected data before saving
    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)


    # Save the concatenated arrays to a file
    np.savez(train_savepath, x_save=x_save, y_save=y_save)



def save_testing_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
    layernum = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())

        for l in modulelist[0:2]:
              x = l(x)
        xo = x
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)

    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)

    np.savez(test_savepath, x_save=x_save, y_save=y_save)











def stability_loss_function_(trainloader,testloader,robust_backbone,class_numbers,fake_loader,last_layer,args):


    robust_backbone = load_model(model_name=args.model_name, dataset=args.in_dataset, threat_model=args.threat_model).to(device)
    last_layer_name, last_layer = list(robust_backbone.named_children())[-1]
    setattr(robust_backbone, last_layer_name, nn.Identity())

    

    robust_backbone_fc_features = MLP_OUT_ORTH1024(last_layer.in_features)

    fc_layers_phase1 = MLP_OUT_BALL(class_numbers)

    for param in fc_layers_phase1.parameters():
        param.requires_grad = False

    net_save_robustfeature = nn.Sequential(robust_backbone, robust_backbone_fc_features, fc_layers_phase1).to(device)

    for param in robust_backbone.parameters():
        param.requires_grad = False




    print(net_save_robustfeature)
    net_save_robustfeature = net_save_robustfeature.to(device)
    data_gen = inf_generator(trainloader)
    batches_per_epoch = len(trainloader)
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(net_save_robustfeature.parameters(), lr=5e-3, eps=1e-2, amsgrad=True)

    def train_save_robustfeature(epoch):
        print('\nEpoch: %d' % epoch)
        net_save_robustfeature.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer1.zero_grad()
            x = inputs
            outputs = net_save_robustfeature(x)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer1.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test_save_robustfeature(epoch):
        best_acc=0
        net_save_robustfeature.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                x = inputs
                outputs = net_save_robustfeature(x)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {'net_save_robustfeature': net_save_robustfeature.state_dict(),'acc': acc,'epoch': epoch}
            torch.save(state, robust_feature_savefolder+'/ckpt.pth')
            best_acc = acc
            save_training_feature(net_save_robustfeature, trainloader,fake_embeddings_loader=fake_loader)
            print('----')
            save_testing_feature(net_save_robustfeature, testloader)
            print('------------')

    makedirs(robust_feature_savefolder)
    for epoch in range(0, args.epoch1):
        train_save_robustfeature(epoch)
        test_save_robustfeature(epoch)
        print('save robust feature to ' + robust_feature_savefolder)


    def df_dz_regularizer(odefunc, z):
        regu_diag = 0.
        regu_offdiag = 0.0
        for ii in np.random.choice(z.shape[0], min(numm,z.shape[0]),replace=False):
            batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(time_df).to(device), x), z[ii:ii+1,...], create_graph=True)
            batchijacobian = batchijacobian.view(z.shape[1],-1)
            if batchijacobian.shape[0]!=batchijacobian.shape[1]:
                raise Exception("wrong dim in jacobian")

            tempdiag = torch.diagonal(batchijacobian, 0)
            regu_diag += torch.exp(exponent*(tempdiag+trans))
            offdiat = torch.sum(torch.abs(batchijacobian)*((-1*torch.eye(batchijacobian.shape[0]).to(device)+0.5)*2), dim=0)
            off_diagtemp = torch.exp(exponent_off*(offdiat+transoffdig))
            regu_offdiag += off_diagtemp

        return regu_diag/numm, regu_offdiag/numm


    def f_regularizer(odefunc, z):
        tempf = torch.abs(odefunc(torch.tensor(time_df).to(device), z))
        regu_f = torch.pow(exponent_f*tempf,2)

        return regu_f

 

 

    makedirs(ODE_FC_save_folder)

    odefunc = ODEfunc_mlp(0)
    feature_layers = ODEBlocktemp(odefunc)
    fc_layers = MLP_OUT_LINEAR(class_numbers)
    for param in fc_layers.parameters():
        param.requires_grad = False
    ODE_FCmodel = nn.Sequential(feature_layers, fc_layers).to(device)

    train_loader_ODE = DataLoader(DenseDatasetTrain(),batch_size=ODE_FC_odebatch,shuffle=True, num_workers=2)
    test_loader_ODE = DataLoader(DenseDatasetTest(),batch_size=ODE_FC_odebatch,shuffle=True, num_workers=2)
    data_gen = inf_generator(train_loader_ODE)
    batches_per_epoch = len(train_loader_ODE)




    optimizer2 = torch.optim.Adam(ODE_FCmodel.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)



    for epoch in range(args.epoch2):
        for itr in tqdm(range(args.epoch2 * batches_per_epoch), desc="Training ODE block with loss function"):
            optimizer2.zero_grad()
            x, y = data_gen.__next__()
            x = x.to(device)

            modulelist = list(ODE_FCmodel)
            y0 = x
            x = modulelist[0](x)
            y1 = x

            y00 = y0
            regu1, regu2 = df_dz_regularizer(odefunc, y00)
            regu1 = regu1.mean()
            regu2 = regu2.mean()

            regu3 = f_regularizer(odefunc, y00)
            regu3 = regu3.mean()

            loss = weight_f*regu3 + weight_diag*regu1 + weight_offdiag*regu2

            loss.backward()
            optimizer2.step()
            torch.cuda.empty_cache()

            tqdm.write(f"Loss: {loss.item()}")


        current_lr = optimizer2.param_groups[0]['lr']
        tqdm.write(f"Epoch {epoch+1}, Learning Rate: {current_lr}")





    def one_hot(x, K):
        return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


 
    feature_layers = ODEBlock(odefunc)
    fc_layers = MLP_OUT_LINEAR(class_numbers)
    ODE_FCmodel = nn.Sequential(feature_layers, fc_layers).to(device)

    for param in odefunc.parameters():
        param.requires_grad = True
    for param in robust_backbone_fc_features.parameters():
        param.requires_grad = False
    for param in robust_backbone.parameters():
        param.requires_grad = False

    new_model_full = nn.Sequential(robust_backbone, robust_backbone_fc_features, ODE_FCmodel).to(device)
    optimizer3 = torch.optim.Adam([{'params': odefunc.parameters(), 'lr': 1e-5, 'eps':1e-6,},{'params': fc_layers.parameters(), 'lr': 1e-2, 'eps':1e-4,}], amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    def train(net, epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer3.zero_grad()
            x = inputs
            outputs = net(x)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer3.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    for epoch in range(0, args.epoch3):
        train(new_model_full, epoch)
    return new_model_full
