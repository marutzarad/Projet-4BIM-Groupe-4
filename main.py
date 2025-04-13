
from portrait_robot_complet import vae, device, decode_latents_to_images, generate_initial_population, force_attributes_on_latents
from intergraph3 import open_main_window
import tkinter as tk

# Tu peux ajouter des fonctions utilitaires pour transformer les images sélectionnées
# en latents, puis générer les enfants et décoder les images

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # On cache la fenêtre principale, lancée dans intergraph3

    # Lancement de l'interface
    open_main_window()

    root.mainloop()
