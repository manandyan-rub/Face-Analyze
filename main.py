import face_analyze
import save_images
import webbrowser
import os
from tkinter import *
from PIL import ImageTk, Image
import cv2
import time
from marvel.marvel import Marvel

cap = None
lmain = None
root = None
root2 = None


def main_window():
    global root
    root = Tk()
    if root2 is not None:
        root2.destroy()

    root.geometry('500x500')
    root.title('Good Morning :)')
    root.bind('<KeyPress-q>', lambda e: root.quit())

    app = Frame(root, bg="white")
    app.pack(expand=True, fill=BOTH)

    button1 = Button(app, text="Submit", command=open_image_window, bg="green", fg="white", font=("Arial", 12))
    button1.pack(side=BOTTOM)

    global lmain
    lmain = Label(app)
    lmain.pack(expand=True, fill=BOTH)

    global cap
    cap = cv2.VideoCapture(0)
    video_stream()
    root.mainloop()


def video_stream():
    global cap

    if cap is not None:
        success, frame = cap.read()

        if success:
            face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade_db.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            root.update_idletasks()
            root.after(1, video_stream)
        else:
            print("Application is closing.")
            cap.release()
            root.destroy()
    else:
        print("Capture object 'cap' is not initialized.")


def open_image_window():
    global cap
    global root
    root.withdraw()
    if cap is not None:
        cam_port = 0
        cam = cv2.VideoCapture(cam_port, cv2.CAP_DSHOW)
        result, image = cam.read()

        if result:
            global root2
            root2 = Tk()
            root2.geometry('500x500')
            root2.title('Your age')
            app2 = Frame(root2, bg="white")
            app2.pack(expand=True, fill=BOTH)
            image_filename = f"image{time.time()}.png"
            directory_path = r"C:\Users\JARVIS\PycharmProjects\project\images and excel files\images"
            cv2.imwrite(os.path.join(directory_path, image_filename), image)
            image_path = os.path.join(directory_path, image_filename)

            age = face_analyze.face_analyze(image_path)

            button_frame = Frame(app2)
            button_frame.pack(side=BOTTOM)

            button_again = Button(button_frame, text="Again", command=main_window, bg="blue", fg="white",
                                  font=("Arial", 12))
            button_again.pack(side=LEFT)

            button_login = Button(button_frame, text="LogIn", command=lambda: log_in(age), bg="red", fg="white",
                                  font=("Arial", 12))
            button_login.pack(side=LEFT)

            label_age = Label(app2, text=f"{age}", font=("Arial", 200))
            label_age.pack()
            cv2.waitKey(0)
            img_binary = image.tobytes()
            save_images.save_images_as_binary(image_filename, img_binary)

        else:
            print("No image detected. Please try again")
    else:
        print("Capture object 'cap' is not initialized.")


iron_man_id = 1009368
deadpool_id = 1009268
spider_man_id = 1009610
thor_id = 1009664
captain_america_id = 1009220


def return_comics(id):
    PUBLIC_KEY = "PUBLIC_KEY"
    PRIVATE_KEY = "PRIVATE_KEY"
    marvel = Marvel(PUBLIC_KEY, PRIVATE_KEY)
    comics = marvel.characters.comics(id)
    comic_links = []

    for comic in comics['data']['results']:
        title = comic['title']
        links = comic['urls']

        comic_details = [f"Title: {title}"]
        for link in links:
            if link['type'] == "detail":
                comic_details.append(f"URL: {link['url']}")
        comic_links.append('\n'.join(comic_details))

    return comic_links


def open_url(url):
    webbrowser.open(url)


def log_in(age):
    if age > 120:
        character_id = deadpool_id
    else:
        character_id = iron_man_id

    comic_links = return_comics(character_id)  # Define this function to return comic links

    root3 = Tk()
    root3.title("Marvel Comic Links")

    comic_text = Text(root3)
    comic_text.pack()

    for comic_link in comic_links:
        parts = comic_link.split('\n')
        title = parts[0]
        url = parts[1].split(': ')[1]

        comic_text.insert(END, title + "\n")
        comic_text.insert(END, url + "\n")
        comic_text.tag_add(url, comic_text.index("end - 2 lines"), comic_text.index("end - 1 lines"))
        comic_text.tag_config(url, foreground="blue", underline=True)
        comic_text.insert(END, "\n\n")

        # Bind the URL to the open_url function when clicked
        comic_text.tag_bind(url, "<Button-1>", lambda event, url=url: open_url(url))

    root3.mainloop()


# def log_in(age):
#     print(age)
#     root2.destroy()
#     root3 = Tk()
#     root3.title("Marvel Comic Links")
#     if age > 12:
#         character_id = deadpool_id
#     else:
#         character_id = iron_man_id
#     comic_links = return_comics(character_id)
#
#     comic_text = Text(root3)
#     comic_text.pack()
#
#     for comic_link in comic_links:
#         parts = comic_link.split('\n')
#         title = parts[0]
#         url = parts[1].split(': ')[1]
#
#         comic_text.insert(END, title + "\n")
#         comic_text.insert(END, url + "\n", url)
#         comic_text.tag_add(url, comic_text.index("end - 2 lines"), comic_text.index("end - 1 lines"))
#         comic_text.tag_config(url, foreground="blue", underline=1)
#         comic_text.insert(END, "\n\n")
#         print(url)
#         def open_url(url):
#             webbrowser.open(url)
#         webbrowser.open(url)
#         comic_text.tag_bind(url, "<Button-1>", command=lambda: open_url(url))
#
#     root3.mainloop()


if __name__ == "__main__":
    main_window()
