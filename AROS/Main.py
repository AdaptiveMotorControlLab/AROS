import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import EmpiricalCovariance
from robustbench.utils import load_model
import torch.nn.functional as F
from torch.utils.data import TensorDataset


num_vclasses=100


num_samples_needed=1

fast=True
epoch1=1
epoch2=1
epoch3=1


model_name_='Wang2023Better_WRN-70-16'
in_dataset='cifar100'
threat_model_='Linf'
 


# cifa10_models=['Ding2020MMA','Rebuffi2021Fixing_70_16_cutmix_extra'] 50000//num_classes

# cifar100_models=['Wang2023Better_WRN-70-16','Rice2020Overfitting']



trainloader,testloader,ID_OOD_loader=get_loaders(in_dataset=in_dataset)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




robust_backbone = load_model(model_name=model_name_, dataset=in_dataset, threat_model=threat_model_).to(device)
last_layer_name, last_layer = list(robust_backbone.named_children())[-1]
setattr(robust_backbone, last_layer_name, nn.Identity())



embeddings, labels = [], []

with torch.no_grad():
    for imgs, lbls in trainloader:
        imgs = imgs.to(device, non_blocking=True)
        embed = robust_backbone(imgs).cpu()  # move to CPU only once per batch
        embeddings.append(embed)
        labels.append(lbls)
embeddings = torch.cat(embeddings).numpy()
labels = torch.cat(labels).numpy()


print("embedding")


if fast==False:
  gmm_dict = {}
  for cls in np.unique(labels):
      cls_embed = embeddings[labels == cls]
      gmm = GaussianMixture(n_components=1, covariance_type='full').fit(cls_embed)
      gmm_dict[cls] = gmm

  print("fake start")

  fake_data = []


  for cls, gmm in gmm_dict.items():
      samples, likelihoods = [], []
      while len(samples) < num_samples_needed:
          s = gmm.sample(100)[0]
          likelihood = gmm.score_samples(s)
          samples.append(s[likelihood < np.quantile(likelihood, 0.001)])
          likelihoods.append(likelihood[likelihood < np.quantile(likelihood, 0.001)])
          if sum(len(smp) for smp in samples) >= num_samples_needed:
              break
      samples = np.vstack(samples)[:num_samples_needed]
      fake_data.append(samples)

  fake_data = np.vstack(fake_data)
  fake_data = torch.tensor(fake_data).float()
  fake_data = F.normalize(fake_data, p=2, dim=1)

  fake_labels = torch.full((fake_data.shape[0],), 10)
  fake_loader = DataLoader(TensorDataset(fake_data, fake_labels), batch_size=128, shuffle=True)

if fast==True:


    noise_std = 0.1  # standard deviation of noise
    noisy_embeddings = torch.tensor(embeddings) + noise_std * torch.randn_like(torch.tensor(embeddings))

    # Normalize Noisy Embeddings
    noisy_embeddings = F.normalize(noisy_embeddings, p=2, dim=1)[:len(trainloader.dataset)//num_classes]

    # Convert to DataLoader if needed
    fake_labels = torch.full((noisy_embeddings.shape[0],), num_classes)[:len(trainloader.dataset)//num_classes]
    fake_loader = DataLoader(TensorDataset(noisy_embeddings, fake_labels), batch_size=128, shuffle=True)



final_model=stability_loss_function_(trainloader,testloader,robust_backbone,num_classes,fake_loader,last_layer)



attack_eps = 8/255
attack_steps = 10
attack_alpha = 2.5 * attack_eps / attack_steps
test_attack = PGD_AUC(final_model, eps=attack_eps, steps=attack_steps, alpha=attack_alpha, num_classes=num_classes)



get_clean_AUC(final_model, ID_OOD_loader , device, num_classes)

adv_auc = get_auc_adversarial(model=final_model,  test_loader=ID_OOD_loader, test_attack=test_attack, device=device, num_classes=num_classes)
