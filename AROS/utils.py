import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import geotorch
from torchdiffeq import odeint_adjoint as odeint
import os
import time
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, TensorDataset
from torch.nn.parameter import Parameter
import math
import numpy as np
from collections import OrderedDict

robust_feature_savefolder = './CIFAR10_resnet_Nov_1'
train_savepath='./CIFAR10_train_resnetNov1.npz'
test_savepath='./CIFAR10_test_resnetNov1.npz'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




weight_diag = 10
weight_offdiag = 0
weight_f = 0.1

weight_norm = 0
weight_lossc =  0

exponent = 1.0
exponent_off = 0.1
exponent_f = 20
time_df = 1
trans = 1.0
transoffdig = 1.0
numm = 16



 
 
ODE_FC_odebatch = 32

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class ConcatFC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
    def forward(self, t, x):
        return self._layer(x)

class ODEfunc_mlp(nn.Module):
    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(128, 128)
        self.act1 = torch.sin
        self.nfe = 0
    def forward(self, t, x):
        self.nfe += 1
        out = -1*self.fc1(t, x)
        out = self.act1(out)
        return out




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

class MLP_OUT_ORTH1024(nn.Module):
    def __init__(self,layer_dim_):
        super(MLP_OUT_ORTH1024, self).__init__()
        self.layer_dim_ = layer_dim_
        self.fc0 = ORTHFC(self.layer_dim_, 128, False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

class newLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(newLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
#         self.weight = self.weighttemp.T
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight.T, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ORTHFC(nn.Module):
    def __init__(self, dimin, dimout, bias):
        super(ORTHFC, self).__init__()
        if dimin >= dimout:
            self.linear = newLinear(dimin, dimout, bias=bias)
        else:
            self.linear = nn.Linear(dimin, dimout, bias=bias)
        geotorch.orthogonal(self.linear, "weight")
    def forward(self, x):
        return self.linear(x)

class MLP_OUT_LINEAR(nn.Module):
    def __init__(self,class_numbers):
        self.class_numbers = class_numbers
        super(MLP_OUT_LINEAR, self).__init__()
        self.fc0 = nn.Linear(128, class_numbers)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

class MLP_OUT_BALL(nn.Module):
    def __init__(self,class_numbers):
        super(MLP_OUT_BALL, self).__init__()
        self.class_numbers = class_numbers
        self.fc0 = nn.Linear(128, class_numbers, bias=False)
        self.fc0.weight.data = torch.randn([class_numbers,128])
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1




criterion = nn.CrossEntropyLoss()


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

# Try to get terminal width, default to 80 if it fails
try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except ValueError:
    term_width = 80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []

    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()





def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)




class DenseDatasetTrain(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(train_savepath)

        self.x = npzfile['x_save']
        self.y = npzfile['y_save']
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx,...]
        y = self.y[idx]

        return x,y
class DenseDatasetTest(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(test_savepath)

        self.x = npzfile['x_save']
        self.y = npzfile['y_save']
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx,...]
        y = self.y[idx]

        return x,y







class OrthogonalBinaryLayer(nn.Module):
    def __init__(self, dimin, dimout=2, bias=True):
        super(OrthogonalBinaryLayer, self).__init__()
        if dimin >= dimout:
            # Use custom linear layer if dimin >= dimout
            self.linear = newLinear(dimin, dimout, bias=bias)
        else:
            self.linear = nn.Linear(dimin, dimout, bias=bias)
        
        # Apply orthogonality constraint on the weight
        geotorch.orthogonal(self.linear, "weight")
        
        # Binarization of weights using a sign function
        self.binarize_weights()

    def binarize_weights(self):
        with torch.no_grad():
            # Replace weights with their sign (-1 or +1)
            self.linear.weight.data = torch.sign(self.linear.weight.data)

    def forward(self, x):
        # Ensure the weights are binary at forward pass
        self.binarize_weights()
        return self.linear(x)




 
def train(net, epoch,trainloader,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs
 

        outputs = net(x)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))




def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)



class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 5]).float()
    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]
    @property
    def nfe(self):
        return self.odefunc.nfe
    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
  





def train_save_robustfeature(epoch,net_save_robustfeature,trainloader,optimizer,criterion):
    print('\nEpoch: %d' % epoch)
    net_save_robustfeature.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs
#         print(inputs.shape)

        outputs = net_save_robustfeature(x)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test_save_robustfeature(epoch,net_save_robustfeature,train_eval_loader,fake_loader,testloader):
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net_save_robustfeature': net_save_robustfeature.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
 
        torch.save(state, robust_feature_savefolder+'/ckpt.pth')
        best_acc = acc

        save_training_feature(net_save_robustfeature, train_eval_loader,fake_embeddings_loader=fake_loader)
        print('----')
        save_testing_feature(net_save_robustfeature, testloader)
        print('------------')




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




class DensemnistDatasetTrain(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(train_savepath)

        self.x = npzfile['x_save']
        self.y = npzfile['y_save']
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx,...]
        y = self.y[idx]

        return x,y
class DensemnistDatasetTest(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(test_savepath)

        self.x = npzfile['x_save']
        self.y = npzfile['y_save']
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx,...]
        y = self.y[idx]

        return x,y




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


    x_save = []
    y_save = []
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

 
    x_save = np.concatenate(x_save)
 
    y_save = np.concatenate(y_save)

 
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




