import pickle
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# === Paramètres pour les autres traitements ===
IMG_SIZE = 64  # Taille de l'image après redimensionnement
attribute_directions = {}  # Dictionnaire pour stocker les directions des attributs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Charger le fichier pickle ===
OUTPUT_PICKLE = "C:/Users/adele/Documents/4A local/S2/Projet_WEB/latents_with_attrs.pkl"  # Nom du fichier pickle
with open(OUTPUT_PICKLE, "rb") as f:
    data = pickle.load(f)

latents = data["latents"]
filenames = data["filenames"]
attributes = data["attributes"]
attr_names = data["attribute_names"]

unique_vals = np.unique(attributes)
print("Valeurs uniques dans attributes :", unique_vals)


print(f"✅ Fichier pickle chargé avec {len(latents)} entrées.")

# === Modèle VAE ===
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

vae = VAE(latent_dimension).to(device)

# Fonction pour décoder les latents en images
def decode_latents_to_images(latents, model, device):
    latents = torch.tensor(latents).float().to(device)
    with torch.no_grad():
        images = model.decode(latents)
    return images.cpu()

# === Calcul des directions des attributs ===
def compute_attribute_directions_from_latents(latents, attributes, attr_names):
    directions = {}
    for i, attr in enumerate(attr_names):
        # Remplacer les 0 par -1
        attributes[:, i] = np.where(attributes[:, i] == 0, -1, attributes[:, i])

        positive = latents[attributes[:, i] == 1]  # Exemples positifs
        negative = latents[attributes[:, i] == -1]  # Exemples négatifs
        
        print(f"Attribut: {attr}, Positifs: {len(positive)}, Négatifs: {len(negative)}")  # Vérification du comptage
        
        if len(positive) > 0 and len(negative) > 0:
            direction = positive.mean(axis=0) - negative.mean(axis=0)
            directions[attr] = direction
        else:
            print(f"⚠️ Pas assez d'exemples pour l'attribut '{attr}', ignoré.")
    return directions



# Calculer les directions des attributs à partir des latents préexistants
attribute_directions = compute_attribute_directions_from_latents(latents, attributes, attr_names)

# Sauvegarder les directions des attributs
with open("attribute_directions.pkl", "wb") as f:
    pickle.dump(attribute_directions, f)

print(f"✅ Les directions d'attributs ont été sauvegardées dans 'attribute_directions.pkl'.")

# === Sélection des attributs et génération ===
def force_attributes_on_latents(children_latents, attr_list, attribute_strength=1.0):
    for i in range(len(children_latents)):
        for attr in attr_list:
            if attr in attribute_directions:
                children_latents[i] += attribute_directions[attr] * attribute_strength
    return children_latents

def show_images(images, title="Génération actuelle"):
    grid = vutils.make_grid(images, nrow=5, normalize=True)
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

# Génération initiale de la population
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

# === Boucle Génétique ===
def genetic_manual_loop(latents, model, device, mutation_strength=0.1):
    parents_latents = latents  # Utiliser les latents directement à partir du fichier pickle
    history = [parents_latents.copy()]
    generation = 1

    while True:
        print(f"\n🔁 Génération {generation}")
        children_latents = generate_initial_population(parents_latents, mutation_strength=mutation_strength)
        children_images = decode_latents_to_images(children_latents, model, device)

        show_images(children_images, title=f"Génération {generation}")

        attr_input = input("\n📢 Veux-tu forcer des attributs sur la prochaine génération ?\n"
                           f"Liste dispo : {', '.join(list(attribute_directions.keys())[:10])}...\n"
                           "Entre des attributs séparés par un espace (ex: Smiling Male), ou ENTER pour rien : ").strip()

        attr_list = [attr for attr in attr_input.split() if attr in attribute_directions]
        if attr_input and not attr_list:
            print("⚠️ Certains attributs sont inconnus ou non supportés.")
        if attr_list:
            print(f"✨ Forçage des attributs : {', '.join(attr_list)}")
            children_latents = force_attributes_on_latents(children_latents, attr_list, attribute_strength)

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
            print("✅ Nouveaux parents sélectionnés pour la prochaine génération.")

# Initialiser le modèle et commencer la boucle
genetic_manual_loop(latents, vae, device, mutation_strength=0.1)
