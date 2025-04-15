import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torchvision.models as models


# Vérification de l'appareil GPU
use_gpu = False
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
print(f"Exécution sur {device}")

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Ajuste la taille des images
    transforms.ToTensor(),         # Convertit en tenseur PyTorch
    transforms.Normalize((0.5,), (0.5,))  # Normalisation (pour centrer les valeurs autour de 0)
])

# Répertoire des images
image_dir = "C:/Users/adele/Documents/4A local/S2/Projet_WEB/data_celebia/Img-20250407T213121Z-001/Img/img_align_celeba/img_align_celeba"

# Dataset personnalisé
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, max_images=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if max_images:
            self.image_files = self.image_files[:max_images]  # Limiter le nombre d'images
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Force RGB ici
        if self.transform:
            image = self.transform(image)
        return image

# DataLoader
dataset = CustomImageDataset(image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Définition du VAE
latent_dimension = 512
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Décodeur
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 4, 4)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

vae = VAE(latent_dimension)

# Modèle pré-entraîné pour la perte perceptuelle
vgg = models.vgg16(pretrained=True).features[:16].eval()
for param in vgg.parameters():
    param.requires_grad = False
vgg = vgg.to(device)

# Calcul de la perte perceptuelle
def perceptual_loss(x, y, model):
    x_vgg = model(x)
    y_vgg = model(y)
    return F.l1_loss(x_vgg, y_vgg)

# Fonction de perte du VAE
def loss_function(recon_x, x, mu, logvar, perceptual_weight=1.0):
    pixel_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalisation pour VGG
    def normalize_for_vgg(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    x_norm = normalize_for_vgg(x)
    recon_x_norm = normalize_for_vgg(recon_x)
    p_loss = perceptual_loss(recon_x_norm, x_norm, vgg)

    return pixel_loss + 0.2 * kl_loss + perceptual_weight * p_loss

# Optimiseur
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Entraînement (actuellement désactivé)
training = False
if training:
    num_epochs = 15
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images in pbar:
            images = images.to(device)
            optimizer.zero_grad()

            recon_images, mu, logvar = vae(images)
            loss = loss_function(recon_images, images, mu, logvar)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (pbar.n + 1)

            pbar.set_description(f"Epoch {epoch+1}: loss = {avg_loss:.3f}")
    torch.save(vae.state_dict(), "/vae_weights_4.4.0.pth")
else:
    vae.load_state_dict(torch.load("C:/Users/adele/Documents/4A local/S2/Projet_WEB/vae_weights_4.4.0.pth", map_location=device))

vae.eval()

# Génération d'images aléatoires
z = torch.randn(16, latent_dimension).to(device)
generated_images = vae.decode(z).cpu().detach()

# Affichage des images générées
# (N.B. L'affichage est optionnel, on l'enlève si nécessaire)
# grid = vutils.make_grid(generated_images, nrow=4, normalize=True)
# plt.figure(figsize=(8, 8))
# plt.imshow(grid.permute(1, 2, 0))
# plt.axis("off")
# plt.show()

# Visualisation des reconstructions (optionnel)
# def visualize_reconstructions(net, images, device=device):
#     with torch.no_grad():
#         images = images.to(device)
#         reconstructions = net(images)[0]
#         return make_grid(reconstructions[1:50], 10, 5).cpu()

# Fonction pour extraire les codes latents
def extract_latent_codes(model, dataloader, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            mu, log_var = model.encode(images)
            latents.append(mu.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    return latents

# Fonction pour tracer l'espace latent
def plot_latent_space(model, dataloader, device, method="tsne", title="Latent Space"):
    latents = extract_latent_codes(model, dataloader, device)

    if latents.shape[1] > 2:
        if method == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=2)
        latents_2d = reducer.fit_transform(latents)
    else:
        latents_2d = latents

    # Affichage optionnel (désactivé ici)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(latents_2d[:, 0], lat


# Utilisation du dataset
# dataset = CustomImageDataset(image_dir, transform=transform, max_images=50000)
dataset = CustomImageDataset(image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Récupérer un batch d'images
images = next(iter(dataloader))


import pandas as pd

# Chemin vers le CSV
ATTR_CSV = "C:/Users/adele/Documents/4A local/S2/Projet_WEB/list_attr_celeba.csv"  # Fichier CSV des attributs modifié avec les noms d'images corrigés

# Chargement du fichier CSV des attributs
attr_df = pd.read_csv(ATTR_CSV, index_col=0)

# Conversion des -1 en 0 pour que tout soit binaire
attr_df[attr_df == -1] = 0

# Exemple d'accès : attr_df.loc['202599.jpg']['Smiling']

###################### ALGO GENETIQUE ##############################

import os
import torch
import pandas as pd
import numpy as np
from torchvision import transforms, utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Paramètres pour les attributs ===

IMG_DIR = "C:/Users/adele/Documents/4A local/S2/Projet_WEB/data_celebia/Img-20250407T213121Z-001/Img/img_align_celeba/img_align_celeba"

# === Paramètres pour les autres traitements ===
NB_SAMPLES = 5000  # Nombre d'échantillons pour calculer les directions
IMG_SIZE = 64  # Taille de l'image après redimensionnement
attribute_directions = {}  # Dictionnaire pour stocker les directions des attributs

# === Paramètres d'entraînement et de génération ===
mutation_strength = 0.5  # Force de la mutation dans la génération
attribute_strength = 1.0  # Force du changement lors du forçage d'attributs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- 1. Nettoyer le CSV en enlevant l'extension .jpg -----
df = pd.read_csv(ATTR_CSV, sep=';')

# ----- 2. Encode les images sélectionnées -----
def get_latents_from_images(images, model, device):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        mu, _ = model.encode(images)
        return mu.cpu().numpy()

# ----- 3. Génère 10 enfants par crossover + mutation -----
def generate_initial_population(parents_latents, n_children=10, mutation_strength=0.5):
    children = []
    for _ in range(n_children):
        p1, p2 = np.random.choice(len(parents_latents), 2, replace=True)
        alpha = np.random.rand()
        child = alpha * parents_latents[p1] + (1 - alpha) * parents_latents[p2]
        mutation = np.random.randn(*child.shape) * mutation_strength
        child += mutation
        children.append(child)
    return np.array(children)

# ----- 4. Decode les latents en images -----
def decode_latents_to_images(latents, model, device):
    model.eval()
    with torch.no_grad():
        latents = torch.tensor(latents, dtype=torch.float32).to(device)
        images = model.decode(latents)
        return images.cpu()

# ----- 5. Affichage des images -----
def show_images(images, title="Génération actuelle"):
    grid = vutils.make_grid(images, nrow=5, normalize=True)
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

# ----- 6. Sélection de l'utilisateur -----
def ask_user_selection(n=10):
    while True:
        selection = input(f"Quels visages sélectionnez-vous ? (1-{n}, 0=regénérer, 'undo'=retour) : ").strip()
        if selection.lower() == "undo":
            return "undo"
        elif selection == "0":
            return []
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split()]
                if all(0 <= idx < n for idx in indices) and len(indices) > 0:
                    return indices
                else:
                    print("❌ Indices invalides. Réessaye.")
            except ValueError:
                print("❌ Format invalide. Réessaye.")

# ----- 7. Force des attributs -----
def force_attributes_on_latents(children_latents, attr_list, attribute_directions, attribute_strength=1.0):
    for i in range(len(children_latents)):
        for attr in attr_list:
            children_latents[i] += attribute_directions[attr] * attribute_strength
    return children_latents

# ----- 8. Calcul des directions des attributs depuis le CSV -----
def compute_attribute_directions_from_csv(model, device):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    print("🔄 Chargement et encodage des images CelebA pour calcul des attributs...")

    attr_df = pd.read_csv("C:/Users/adele/Documents/4A local/S2/Projet_WEB/list_attr_celeba.csv", sep=';', index_col=0)
    attr_names = list(attr_df.columns)
    selected_df = attr_df.sample(n=NB_SAMPLES, random_state=42)
    images, attrs = [], []

    for img_name, row in tqdm(selected_df.iterrows(), total=NB_SAMPLES):
        img_name_with_extension = img_name.strip()
        path = os.path.join(IMG_DIR, img_name_with_extension)

        if not os.path.exists(path):
            print(f"⚠️ L'image {img_name_with_extension} n'a pas été trouvée dans le dossier.")
            continue

        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)
            images.append(img)
            attrs.append(row.values)
        except Exception as e:
            print(f"⚠️ Échec de chargement de l'image {img_name_with_extension} : {e}")
            continue

    if len(images) == 0:
        raise RuntimeError("Aucune image valide n'a été chargée.")

    images = torch.stack(images).to(device)

    with torch.no_grad():
        mu, _ = model.encode(images)
        latents = mu.cpu().numpy()

    attrs = np.array(attrs)

    return latents, attrs, attr_names

# ----- 9. Boucle évolutive manuelle -----
def genetic_manual_loop(initial_images, model, device, mutation_strength=0.5):
    global attribute_directions
    if len(attribute_directions) == 0:
        latents, attrs, attr_names = compute_attribute_directions_from_csv(model, device)

        attribute_directions = {}
        for i, attr in enumerate(attr_names):
            try:
                positive = latents[attrs[:, i] == 1]
                negative = latents[attrs[:, i] == -1]
                if len(positive) > 0 and len(negative) > 0:
                    direction = positive.mean(axis=0) - negative.mean(axis=0)
                    attribute_directions[attr] = direction
                else:
                    print(f"⚠️ Pas assez d'exemples pour l'attribut '{attr}', ignoré.")
            except Exception as e:
                print(f"⚠️ Erreur lors du calcul de la direction pour '{attr}' : {e}")
                continue

        print(f"✅ {len(attribute_directions)} directions d'attributs calculées.")

    parents_latents = get_latents_from_images(initial_images, model, device)
    history = [parents_latents.copy()]
    generation = 1

    while True:
        print(f"\n🔁 Génération {generation}")
        children_latents = generate_initial_population(parents_latents, mutation_strength=mutation_strength)
        children_images = decode_latents_to_images(children_latents, model, device)

        #show_images(children_images, title=f"Génération {generation}")

        attr_input = input("\n📢 Veux-tu forcer des attributs sur la prochaine génération ?\n"
                           f"Liste dispo : {', '.join(list(attribute_directions.keys())[:10])}...\n"
                           "Entre des attributs séparés par un espace (ex: Smiling Male), ou ENTER pour rien : ").strip()

        attr_list = [attr for attr in attr_input.split() if attr in attribute_directions]
        if attr_input and not attr_list:
            print("⚠️ Certains attributs sont inconnus ou non supportés.")
        if attr_list:
            print(f"✨ Forçage des attributs : {', '.join(attr_list)}")
            children_latents = force_attributes_on_latents(children_latents, attr_list, attribute_directions, attribute_strength)


        selection = ask_user_selection(n=10)

        if selection == "undo":
            if generation == 1:
                print("⚠️ Tu es déjà à la première génération, impossible de revenir en arrière.")
                continue
            history.pop()
            parents_latents = history[-1]
            generation -= 1
            print("↩️ Retour à la génération précédente.")
            continue

        elif selection == []:
            print("🔁 Nouvelle génération avec les mêmes parents...")
            continue

        else:
            parents_latents = children_latents[selection]
            history.append(parents_latents.copy())
            generation += 1