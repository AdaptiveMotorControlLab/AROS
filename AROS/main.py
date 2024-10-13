
!pip install -r requirements.txt
import argparse
import torch
import torch.nn as nn
from evaluate import *
from utils import *
from tqdm.notebook import tqdm
from data_loader import *
from stability_loss_function import *

def main():
    parser = argparse.ArgumentParser(description="Hyperparameters for the script")

    # Define the hyperparameters controlled via CLI 'Ding2020MMA' 
    parser.add_argument('--fast', type=bool, default=True, help='Toggle between fast and full fake data generation modes')
    parser.add_argument('--epoch1', type=int, default=2, help='Number of epochs for stage 1')
    parser.add_argument('--epoch2', type=int, default=1, help='Number of epochs for stage 2')
    parser.add_argument('--epoch3', type=int, default=2, help='Number of epochs for stage 3')
    parser.add_argument('--in_dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='The in-distribution dataset to be used')
    parser.add_argument('--threat_model', type=str, default='Linf', help='Adversarial threat model for robust training')
    parser.add_argument('--noise_std', type=float, default=1, help='Standard deviation of noise for generating noisy fake embeddings')
    parser.add_argument('--attack_eps', type=float, default=8/255, help='Perturbation bound (epsilon) for PGD attack')
    parser.add_argument('--attack_steps', type=int, default=10, help='Number of steps for the PGD attack')
    parser.add_argument('--attack_alpha', type=float, default=2.5 * (8/255) / 10, help='Step size (alpha) for each PGD attack iteration')

    args = parser.parse_args('')

    # Set the default model name based on the selected dataset
    if args.in_dataset == 'cifar10':
        default_model_name = 'Rebuffi2021Fixing_70_16_cutmix_extra'
    elif args.in_dataset == 'cifar100':
        default_model_name = 'Wang2023Better_WRN-70-16'

    parser.add_argument('--model_name', type=str, default=default_model_name, choices=['Rebuffi2021Fixing_70_16_cutmix_extra', 'Wang2023Better_WRN-70-16'], help='The pre-trained model to be used for feature extraction')

    # Re-parse arguments to include model_name selection based on the dataset
    args = parser.parse_args('')
    num_classes = 10 if args.in_dataset == 'cifar10' else 100

    trainloader, testloader,test_set, ID_OOD_loader = get_loaders(in_dataset=args.in_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    robust_backbone = load_model(model_name=args.model_name, dataset=args.in_dataset, threat_model=args.threat_model).to(device)
    last_layer_name, last_layer = list(robust_backbone.named_children())[-1]
    setattr(robust_backbone, last_layer_name, nn.Identity())
    fake_loader=None


    num_fake_samples = len(trainloader.dataset) // num_classes




    embeddings, labels = [], []

    with torch.no_grad():
        for imgs, lbls in trainloader:
            imgs = imgs.to(device, non_blocking=True)
            embed = robust_backbone(imgs).cpu()  # move to CPU only once per batch
            embeddings.append(embed)
            labels.append(lbls)
    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()


    print("embedding computed...")


    if args.fast==False:
      gmm_dict = {}
      for cls in np.unique(labels):
          cls_embed = embeddings[labels == cls]
          gmm = GaussianMixture(n_components=1, covariance_type='full').fit(cls_embed)
          gmm_dict[cls] = gmm

      print("fake crafing...")

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

    if args.fast==True:


        noise_std = 0.1  # standard deviation of noise
        noisy_embeddings = torch.tensor(embeddings) + noise_std * torch.randn_like(torch.tensor(embeddings))

        # Normalize Noisy Embeddings
        noisy_embeddings = F.normalize(noisy_embeddings, p=2, dim=1)[:len(trainloader.dataset)//num_classes]

        # Convert to DataLoader if needed
        fake_labels = torch.full((noisy_embeddings.shape[0],), num_classes)[:len(trainloader.dataset)//num_classes]
        fake_loader = DataLoader(TensorDataset(noisy_embeddings, fake_labels), batch_size=128, shuffle=True) 


    final_model = stability_loss_function_(trainloader, testloader, robust_backbone, num_classes, fake_loader, last_layer, args)

 
    test_attack = PGD_AUC(final_model, eps=args.attack_eps, steps=args.attack_steps, alpha=args.attack_alpha, num_classes=num_classes)
    get_clean_AUC(final_model, ID_OOD_loader , device, num_classes)
    adv_auc = get_auc_adversarial(model=final_model,  test_loader=ID_OOD_loader, test_attack=test_attack, device=device, num_classes=num_classes)



if __name__ == "__main__":
    main()

