import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import os
import random
import pandas as pd


# GLOBAL VARIABLES
selected_images = []
selected_attributes = []

def on_enter(button):
    button.config(bg="#402408", fg="white")

def on_leave(button):
    button.config(bg="#ecd7c6", fg="black")

def random_images(directory, images_list):
    images_to_show = []
    images_names = []
    while len(images_to_show) < 10:
        image = os.path.join(directory, random.choice(images_list))
        if image not in images_to_show:
            images_to_show.append(image)
            images_names.append(os.path.basename(image))
    return images_to_show, images_names


def image_selection(img_name, var):
    global selected_images
    if var.get() == 1:
        if img_name not in selected_images:
            selected_images.append(img_name)
    else:
        if img_name in selected_images:
            selected_images.remove(img_name)
    print("Images sélectionnées :", selected_images)


def show_images(frame, images_to_show, images_names):
    image_refs = []
    for index in range(len(images_to_show)):
        img_path = images_to_show[index]
        img_name = images_names[index]

        image = ImageTk.PhotoImage(Image.open(img_path).resize((150, 150)))
        image_refs.append(image)

        row, col = divmod(index, 5)

        tk.Label(frame, image=image).grid(row=row * 2, column=col, padx=0, pady=0)

        var = tk.IntVar(value=0)
        tk.Checkbutton(frame, text="I choose you!", variable=var, bg = "#ad7e66",
                       command=lambda name=img_name, v=var: image_selection(name, v)
                       ).grid(row=row * 2 + 1, column=col, pady=5)

    frame.image_refs = image_refs


def reset_selection(directory, images_list, frame):
    global selected_images
    selected_images.clear()
    img_list, img_name = random_images(directory, images_list)
    for widget in frame.winfo_children():
        widget.destroy()
    show_images(frame, img_list, img_name)


def attr_selection(attr, v):
    global selected_attributes
    if v.get() == 1:
        if attr not in selected_attributes:
            selected_attributes.append(attr)
    else:
        if attr in selected_attributes:
            selected_attributes.remove(attr)
    print("Attributs sélectionnées :", selected_attributes)


def go_back(window):
    window.destroy()


def open_new_window(attributes, main_window):
    #create global variable that has the selected attributes

    second_window = tk.Toplevel()
    second_window.title("Application Dev Log")
    second_window.geometry("500x600")
    second_window.resizable(False, False)
    second_window.configure(bg="#ecd7c6")

    second_frame = tk.Frame(second_window, bg="#ecd7c6")
    second_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(second_frame, bg="#ecd7c6")
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    content_frame = tk.Frame(canvas, bg="#ecd7c6")
    canvas.create_window((0, 0), window=content_frame, anchor="nw")

    sec_title = tk.Label(content_frame, text="You can choose some extra characteristics :", font=("Century Gothic", 14, "bold"), bg="#ecd7c6")
    sec_title.grid(row=0, column=0, columnspan=2, pady=10)

    sec_wtd = tk.Label(content_frame, text="Select the attributes that you want to see in your final result :", font=("Century Gothic", 14, "bold"), bg="#ecd7c6")
    sec_wtd.grid(row=1, column=0, columnspan=2, pady=5)

    check_vars = {}
    for index in range(len(attributes)):
        attr = attributes[index]
        check_vars[attr] = tk.IntVar(value=0)
        row = (index // 2) + 2
        column = index % 2
        tk.Checkbutton(content_frame, text=attr, variable=check_vars[attr], bg="#ecd7c6",
                   command=lambda v=check_vars[attr], a=attr: attr_selection(a, v)
                   ).grid(row=row, column=column, padx=10, pady=2, sticky="w")

    validatebutton = tk.Button(second_window, text="I validate my choices!", bg="#ecd7c6", padx=10, pady=5, command=lambda: (main_window.withdraw(), second_window.withdraw(), open_third_window()))
    validatebutton.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.X)  # Make it fill horizontally and aligned to the right

    backbutton = tk.Button(second_window, text="I want to go back", bg="#ecd7c6", padx=10, pady=5, command=lambda: go_back(second_window))
    backbutton.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.X)  # Same for this button


def open_third_window():
    third_window = tk.Toplevel()
    third_window.title("Welcome to <NameofApplication>")
    third_window.geometry("600x600")
    third_window.resizable(False, False)


    third_frame = tk.Frame(third_window, width=600, height=600, bg = "#FFEBCD")
    third_frame.grid(row=0, column=0)

    third_label = tk.Label(third_frame, text="And here is the big reveal!",
                             fg="pink", font=("Times New Roman", 20, "bold"), bg = "#FFEBCD")
    third_label.place(x=300, y=200, anchor=tk.CENTER)





# ------------------- MAIN APPLICATION -------------------
def open_main_window():
    global selected_images
    selected_images = []

    traits = pd.read_csv('list_attr_celeba.csv', sep=';', header=0)
    attributes = list(traits)
    attributes.pop(0)

    attributes_modified = []
    for attr in attributes:
        attr = attr.replace("_", " ")
        #print(attr)
        attributes_modified.append(attr)

    #ERASE ATTRIBUTES THAT ARE USELESS, ALL NEEDED ATTRIBUTES ARE IN attributes_modified

    attributes_modified.pop(0)
    attributes_modified.pop(1)
    attributes_modified.remove("Blurry")
    attributes_modified.remove("Goatee")
    attributes_modified.remove("Mouth Slightly Open")
    attributes_modified.remove("Sideburns")
    attributes_modified.remove("Heavy Makeup")
    attributes_modified.remove("Bags Under Eyes")

    main_window = tk.Toplevel()
    main_window.title("Application Dev Log")
    main_window.geometry("1000x600")
    main_window.resizable(False, False)
    main_window.configure(bg="#ecd7c6")


    main_label = tk.Label(main_window, bg="#ecd7c6", text="And so we begin...", font=("Century Gothic", 20, "bold"))
    main_label.grid(row=0, column=0, pady=10, padx=10)


    second_label = tk.Label(main_window, text="You are free to choose from the photos below, if you don't find anything you like, you can refresh the page!",
                        bg="#cf7928", fg="white", padx=300, pady=10, font=("Century Gothic", 14))
    second_label.grid(row=1, column=0, pady=5, padx=10)


    photos_directory = os.path.expanduser("~/Desktop/DEVLOG/sample")
    images_list = os.listdir(photos_directory)
    images_show, images_name = random_images(photos_directory, images_list)


    frame = tk.Frame(main_window, bg="#ad7e66")
    frame.grid(row=2, column=0, pady=10)

    show_images(frame, images_show, images_name)


    resetbutton = tk.Button(main_window,
                        text="I don't like these photos, redo!",
                        bg="#ecd7c6",
                        command=lambda: reset_selection(photos_directory, images_list, frame),
                        padx=5, pady=5,
                        relief="flat")
    resetbutton.bind("<Enter>", lambda e: on_enter(resetbutton))
    resetbutton.bind("<Leave>", lambda e: on_leave(resetbutton))

    resetbutton.grid(row=3, column=0, pady=10, padx=10)

    validatebutton = tk.Button(main_window,
                           text="Validate my choice, let's move on!",
                           bg="#ecd7c6",
                           command=lambda: open_new_window(attributes_modified, main_window),
                           padx=5, pady=5,
                           relief="flat")
    validatebutton.bind("<Enter>", lambda e: on_enter(validatebutton))
    validatebutton.bind("<Leave>", lambda e: on_leave(validatebutton))

    validatebutton.grid(row=4, column=0, pady=5, padx=10)


    main_window.grid_rowconfigure(0, weight=0)
    main_window.grid_rowconfigure(1, weight=0)
    main_window.grid_rowconfigure(2, weight=1)
    main_window.grid_rowconfigure(3, weight=0)
    main_window.grid_rowconfigure(4, weight=0)

    main_window.grid_columnconfigure(0, weight=1)  # Stretch columns in grid



if __name__ == "__main__":
    welcome_window = tk.Tk()
    welcome_window.title("Welcome to <NameofApplication>")
    welcome_window.geometry("600x600")
    welcome_window.resizable(False, False)

    welcome_frame = tk.Frame(welcome_window, width=600, height=600, bg="#ecd7c6")
    welcome_frame.grid(row=0, column=0, sticky="nsew")

# Update grid weights to allow the frame to expand
    welcome_window.grid_rowconfigure(0, weight=1)
    welcome_window.grid_columnconfigure(0, weight=1)

    welcome_label = tk.Label(welcome_frame, text="Welcome to <NameofApplication>", fg="black", font=("Century Gothic", 28, "bold"), bg="#ecd7c6")
    welcome_label.place(x=300, y=200, anchor=tk.CENTER)

    description_string = (
        "The rules are pretty simple:\n"
        "By starting this journey, you will have to choose some photos that you like,\n"
        "choose some extra attributes for your characters, and we will do the job for you.\n"
        "We will suggest some photos that we have created, taking inspiration\n"
        "from the ones you just chose.\n"
        "No worries if you don't resonate with what we suggest\n"
        "We can always recreate new ideas!\n"
        "Enjoy the experience!\n"
        )

    description_label = tk.Label(welcome_frame, text=description_string, fg="white", font=("Century Gothic", 14, "bold"), bg="#ad7e66")
    description_label.place(x=300, y=350, anchor=tk.CENTER)

    start_button = tk.Button(welcome_window,
                         text="Let's start!",
                         bg="#ecd7c6",
                         padx=10, pady=5,
                         command=lambda: (welcome_window.withdraw(), open_main_window()),
                         relief="flat")

    start_button.bind("<Enter>", lambda e: on_enter(start_button))
    start_button.bind("<Leave>", lambda e: on_leave(start_button))

    start_button.grid(row=1, column=0, padx=5, pady=5)
    welcome_window.grid_rowconfigure(1, weight=0)

    welcome_window.update_idletasks()

    welcome_window.mainloop()
