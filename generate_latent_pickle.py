import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# ====== PARAMÈTRES ======
IMG_DIR = "C:/Users/adele/Documents/4A local/S2/Projet_WEB/data_celebia/Img-20250407T213121Z-001/Img/img_align_celeba/img_align_celeba"
ATTR_CSV = "C:/Users/adele/Documents/4A local/S2/Projet_WEB/list_attr_celeba.csv"
OUTPUT_PICKLE = "C:/Users/adele/Documents/4A local/S2/Projet_WEB/latents_with_attrs.pkl"
IMG_SIZE = 64
BATCH_SIZE = 64
LATENT_DIM = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Exécution sur {device}")

# ====== TRANSFORMATIONS ======
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ====== DATASET ======
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(self.image_files) == 0:
            raise RuntimeError("❌ Le dossier ne contient aucune image valide.")
        print(f"✅ {len(self.image_files)} images détectées dans {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        path = os.path.join(self.image_dir, img_name)
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

# ====== VAE DEFINITION (COMPLET) ======
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU()
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
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

# ====== CHARGEMENT DU MODELE ======
vae = VAE(LATENT_DIM).to(device)
model_path = os.path.join(os.path.dirname(__file__), "vae_weights_4.4.0.pth")
vae.load_state_dict(torch.load(model_path, map_location=device))
vae.eval()

# ====== DATA LOADER ======
dataset = CustomImageDataset(IMG_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====== ATTRIBUTS ======
attr_df = pd.read_csv(ATTR_CSV, sep=';')
print(attr_df.columns)
print(attr_df.head())

# Vérifier que les noms des fichiers sont dans la colonne 'nom'
names = attr_df['nom'].tolist()  # Créer la liste des noms de fichiers à partir de la colonne 'nom'

# Afficher les 10 premiers noms
for name in names[:10]:  # Affiche les 10 premiers noms
    print(name.split('.')[0])  # Affiche le nom sans l'extension '.jpg'
# ====== PREPARATION DES ATTRIBUTS ======
attr_df[attr_df == -1] = 0  # Remplacer les -1 par 0 pour les attributs
target_attrs = [
    "Smiling", "Male", "Young", "Eyeglasses", "No_Beard",
    "Black_Hair", "Blond_Hair", "Brown_Hair", "Bald",
    "Wavy_Hair", "Straight_Hair", "Heavy_Makeup", "Pale_Skin",
    "Bushy_Eyebrows", "Big_Lips", "Oval_Face"
]

attr_df = attr_df[["nom"] + target_attrs]  # Garde seulement les colonnes nécessaires

# Create a dictionary for faster lookups of attributes (remove .jpg from 'nom' column)
attr_dict = dict(zip(attr_df['nom'].str.replace('.jpg', ''), attr_df[target_attrs].values))

latents = []
filenames = []
attributes = []

print("📥 Encodage des images...")

# Encodage des images en latent space et récupération des attributs
for batch, names in tqdm(dataloader):
    batch = batch.to(device)
    with torch.no_grad():
        mu, _ = vae.encode(batch)  # récupération du mu uniquement
        latents.append(mu.cpu().numpy())
        filenames.extend(names)

        # Attributs correspondants
        # On récupère les attributs pour chaque image du batch
        batch_attrs = []
        for name in names:
            base_name = name.split('.')[0]  # Remove file extension from image name
            if base_name in attr_dict:
                batch_attrs.append(attr_dict[base_name])  # Retrieve attributes from the dictionary
            else:
                print(f"Nom {name} introuvable dans le DataFrame.")  # If name is not found
        attributes.extend(batch_attrs)

latents = np.concatenate(latents, axis=0)
attributes = np.array(attributes)

# ====== SAUVEGARDE ======
data = {
    "latents": latents,
    "filenames": filenames,
    "attributes": attributes,
    "attribute_names": target_attrs
}

with open(OUTPUT_PICKLE, "wb") as f:
    pickle.dump(data, f)

print(f"✅ Fichier sauvegardé dans {OUTPUT_PICKLE} avec {len(latents)} entrées.")