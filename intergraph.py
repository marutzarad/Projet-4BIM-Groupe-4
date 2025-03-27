import os
import random
import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd

###################################################
########### FUNCTIONS FOR BUTTONS ################
###################################################

def random_images(directory, images_list):
    """Chooses 10 random images from a directory containing photos"""
    images_to_show = []
    images_names = []
    while len(images_to_show) < 10:
        image = os.path.join(directory, random.choice(images_list))
        if image not in images_to_show:
            images_to_show.append(image)
            images_names.append(os.path.basename(image))
    return images_to_show, images_names  # Return selected images and their names


def show_images(frame, images_to_show, images_names):
    image_refs = []
    selected_vars = {}

    for index in range(len(images_to_show)):
        img_path = images_to_show[index]
        img_name = images_names[index]

        image = ImageTk.PhotoImage(Image.open(img_path).resize((150, 150)))
        image_refs.append(image)

        row, col = divmod(index, 5)

        # Affichage de l'image
        tk.Label(frame, image=image).grid(row=row * 2, column=col, padx=5, pady=5)

        var = tk.IntVar(value=0)
        selected_vars[img_name] = var
        tk.Checkbutton(frame, text="Choisir", variable=var,
                       command=lambda name=img_name, v=var: image_selection(name, v)
                       ).grid(row=row * 2 + 1, column=col, pady=5)

    frame.image_refs = image_refs  # Évite la suppression des images par le garbage collector


#CREATE A GLOBAL VARIABLE FOR THE SELECTED IMAGES
selected_images = []
def image_selection(img_name, var):
    """Handles selection and deselection of images"""
    global selected_images  # Access the global list

    if var.get() == 1:
        if img_name not in selected_images:
            selected_images.append(img_name)
    else:
        if img_name in selected_images:
            selected_images.remove(img_name)

    print("Images sélectionnées :", selected_images)  # Debugging output

def reset_selection(directory, images_list, frame):
    """Réinitialise les images et les sélections"""
    global selected_images
    selected_images.clear()
    #selected_vars.clear()
    img_list, img_name = random_images(directory, images_list)
    for widget in frame.winfo_children():
        widget.destroy()
    show_images(frame, img_list, img_name)

def new_window(attributes):
    second_window = tk.Tk()
    second_window.title("Application Dev Log")
    second_window.geometry("600x500")


    second_frame = tk.Frame(second_window)
    second_frame.pack(fill=tk.BOTH, expand=True)


    canvas = tk.Canvas(second_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


    scrollbar = tk.Scrollbar(second_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)


    content_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=content_frame, anchor="nw")

    content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))


    sec_title = tk.Label(content_frame, text="Choisissez des caractéristiques supplémentaires :", font=("Arial", 14))
    sec_title.grid(row=0, column=0, columnspan=2, pady=10)

    sec_wtd = tk.Label(content_frame, text="Sélectionnez les options que vous souhaitez voir dans le résultat final :")
    sec_wtd.grid(row=1, column=0, columnspan=2, pady=5)


    check_vars = {}
    for index in range(len(attributes)):
        attr = attributes[index]
        check_vars[attr] = tk.IntVar(value=0)
        row = (index // 2) + 2
        column = index % 2
        tk.Checkbutton(content_frame, text=attr, variable=check_vars[attr],
                       command=lambda v=check_vars[attr], a=attr: attr_selection(a, v)
                       ).grid(row=row, column=column, padx=10, pady=2, sticky="w")

    #Button to continue with the autoencodeur
    validatebutton = tk.Button(second_window, text = "Je valide tous mes choix!", command = lambda: third_window())
    validatebutton.pack(side = tk.RIGHT)

    #Button to go back to the main window
    backbutton = tk.Button(second_window, text = "Je veux revenir en arrière!", command = lambda: go_back(second_window))
    backbutton.pack(side = tk.RIGHT)

    second_window.mainloop()


selected_attributes = [] #create a global variable for selected attributes
def attr_selection(attr, v):
    global selected_attributes
    if var.get() == 1:
        if attr not in selected_attributes:
            selected_attributes.append(attr)
    else:
        if attr in selected_attributes:
            selected_attributes.remove(attr)

def go_back(window):
    window.destroy()


################################################
################# MAIN CODE ####################
################################################
# IMPORT CSV (if necessary)
traits = pd.read_csv('list_attr_celeba.csv', sep=';', header=0)
print(traits)

#GET ALL THE CHARACTERISTICS
attributes = list(traits)
attributes.pop(0)
print(attributes)

# CREATE MAIN WINDOW
main_window = tk.Tk()
main_window.title("Application Dev Log")
main_window.geometry("1000x600")  # Adjusted size

# ADD A MAIN TITLE AND SUBTITLE
main_label = tk.Label(main_window, text="Bienvenue sur notre application", font=("Arial", 14))
main_label.pack(pady=10)

second_label = tk.Label(main_window, text="You are free to choose your photos below", bg="pink", padx=300, pady=10)
second_label.pack(pady=5)

# IMPORT PHOTOS FROM THE SAMPLE DIRECTORY
photos_directory = os.path.expanduser("~/Desktop/DEVLOG/sample")
images_list = os.listdir(photos_directory)  # More efficient

# CHOOSE 10 RANDOM PHOTOS
images_show, images_name = random_images(photos_directory, images_list)

# CREATE A FRAME TO CONTAIN PHOTOS
frame = tk.Frame(main_window)
frame.pack(pady=10)

# SHOW THE PHOTOS ON THE FRAME
show_images(frame, images_show, images_name)

#ADD A BUTTON THAT REINITIALIZES THE 10 IMAGES IN CASE THE USER DOESN'T LIKE THEM
resetbutton = tk.Button(main_window, text="I don't like these photos, redo!",command=lambda: reset_selection(photos_directory, images_list, frame))
resetbutton.pack(pady=10)

#IF OK WITH SELECTION, WE CREATE A NEW BUTTON THAT OPENS A NEW WINDOW
validatebutton = tk.Button(main_window, text = "Validate my choice, let's move on!", command = lambda: new_window(attributes))
validatebutton.pack(pady= 5)
print(selected_images)
# Start the Tkinter event loop
main_window.mainloop()
