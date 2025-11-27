import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np

PANEL_W = 500
PANEL_H = 400


def startCompressing():
    return

def upload_decompress():
    return

def upload():
    global originalImage, originalNp, originalSize
    file_path = filedialog.askopenfilename(title="select an image: ", filetypes=[("Image Files","*.png *.jpg")])
    if not file_path:
        return
    originalImage = Image.open(file_path).convert("RGB")
    originalNp = np.array(originalImage)
    originalSize = originalImage.size
    display_img = ImageOps.contain(originalImage, (PANEL_W, PANEL_H), Image.Resampling.LANCZOS)
    bg = Image.new("RGB", (PANEL_W, PANEL_H), (255, 255, 255))
    x = (PANEL_W-display_img.width)//2
    y = (PANEL_H-display_img.height)//2
    bg.paste(display_img, (x, y))
    photo = ImageTk.PhotoImage(bg)
    original_label.configure(image=photo, text="")
    original_label.image = photo
    proceLabel.configure(image="", text="Processed Image\n(Compressed/Decompressed)")

def padding():
    global imgNp, paddedImg, paddedSize, originalNp, originalSize

    if originalNp is None:
        return

    block_w = int(widthEn.get() or 8)
    block_h = int(heightEn.get() or 8)
    h, w, c = originalNp.shape
    pad_h = (block_h-(h % block_h)) % block_h
    pad_w = (block_w-(w % block_w)) % block_w
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(originalNp, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        imgNp = padded
        paddedImg = Image.fromarray(padded)
        paddedSize = (w + pad_w, h + pad_h)
    else:
        imgNp = originalNp.copy()
        paddedImg = originalImage.copy()
        paddedSize = originalSize

def split2blocks():
    global blocks, imgNp

    if imgNp is None:
        return None

    block_w = int(widthEn.get() or 8)
    block_h = int(heightEn.get() or 8)
    h, w, c = imgNp.shape
    blocks_arr = imgNp.reshape(h//block_h, block_h, w//block_w, block_w, 3)
    blocks_arr = blocks_arr.swapaxes(1, 2).reshape(-1, block_h, block_w, 3)
    blocks = blocks_arr
    return blocks

def create_gui():
    global original_label, proceLabel, widthEn, heightEn, ratio_label
    
    top_frame = tkinter.Frame(root, bg="#f0f0f0")
    top_frame.pack(pady=10, fill="x")
    tkinter.Button(top_frame, text="Upload image to compress", command=upload,width=25, height=1, bg="white", fg="black", font=(11), bd=1).pack(side=tkinter.LEFT, padx=5)
    



    images_frame = tkinter.Frame(root, bg="#f0f0f0")
    images_frame.pack(pady=20)
    original_panel = tkinter.Frame(images_frame, width=PANEL_W, height=PANEL_H, bg="white", bd=5, highlightbackground="#00ff00", highlightthickness=5)
    original_panel.grid(row=0, column=0, padx=10)
    original_panel.pack_propagate(False)
    original_label = tkinter.Label(original_panel, text="Original Image", font=(20), bg="white")
    original_label.pack(expand=True, fill="both")
    processed_panel = tkinter.Frame(images_frame, width=PANEL_W, height=PANEL_H,bg="white", bd=5, highlightbackground="#ff0000", highlightthickness=5)
    processed_panel.grid(row=0, column=1, padx=10)
    processed_panel.pack_propagate(False)
    proceLabel = tkinter.Label(processed_panel, text="Processed Image\n(Compressed/Decompresssed)", font=(20), bg="white")
    proceLabel.pack(expand=True, fill="both")
    con = tkinter.Frame(root, bg="#f0f0f0")
    con.pack(pady=30)
    


    tkinter.Label(con, text="Block Width: ", bg="#f0f0f0",font=(12)).grid(row=0,column=0,padx=5, pady=10)
    widthEn = tkinter.Entry(con, width=8, font=("Arial", 12), bd=1)
    widthEn.grid(row=0, column=1, padx=5, pady=10)
    
    tkinter.Label(con, text="Block Height: ", bg="#f0f0f0", font=(12)).grid(row=1, column=0, padx=5, pady=10)
    heightEn = tkinter.Entry(con, width=8, font=("Arial", 12), bd=1)
    heightEn.grid(row=1, column=1, padx=5, pady=10)
    


    
root = tkinter.Tk()
root.title("Image compression/ decompression")
create_gui()
root.mainloop()