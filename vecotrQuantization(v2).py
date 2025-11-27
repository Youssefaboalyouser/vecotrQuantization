import tkinter
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import json
import struct
import os
from sklearn.cluster import KMeans

PANEL_W = 500
PANEL_H = 400

originalImage = None
originalNp = None
originalSize = None
imgNp = None
paddedImg = None
paddedSize = None
blocks = None
is_processing = False

def upload():
    global originalImage, originalNp, originalSize
    
    try:
        file_path = filedialog.askopenfilename(
            title="Select an image: ",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        
        if not file_path:
            return
        
        if not (file_path.lower().endswith((".png", ".jpg", ".jpeg"))):
            messagebox.showerror("Invalid file", "Please select a valid image file!")
            return
        
        originalImage = Image.open(file_path).convert("RGB")
        originalNp = np.array(originalImage)
        originalSize = originalImage.size
        
        display_img = ImageOps.contain(originalImage, (PANEL_W, PANEL_H), Image.Resampling.LANCZOS)
        bg = Image.new("RGB", (PANEL_W, PANEL_H), (255, 255, 255))
        x = (PANEL_W - display_img.width) // 2
        y = (PANEL_H - display_img.height) // 2
        bg.paste(display_img, (x, y))
        photo = ImageTk.PhotoImage(bg)
        original_label.configure(image=photo, text="")
        original_label.image = photo
        
        proceLabel.configure(image="", text="Processed Image\n(Compressed/Decompressed)")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

def padding():
    global imgNp, paddedImg, paddedSize, originalNp, originalSize

    if originalNp is None:
        return False

    try:
        block_w = int(widthEn.get() or 8)
        block_h = int(heightEn.get() or 8)
        
        if block_w <= 0 or block_h <= 0:
            messagebox.showerror("Invalid Input", "Block width and height must be positive!")
            return False
        
        h, w, c = originalNp.shape
        pad_h = (block_h - (h % block_h)) % block_h
        pad_w = (block_w - (w % block_w)) % block_w
        
        if pad_h > 0 or pad_w > 0:
            padded = np.pad(originalNp, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            imgNp = padded
            paddedImg = Image.fromarray(padded)
            paddedSize = (w + pad_w, h + pad_h)
        else:
            imgNp = originalNp.copy()
            paddedImg = originalImage.copy()
            paddedSize = originalSize
        
        return True
        
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for block dimensions!")
        return False
    except Exception as e:
        messagebox.showerror("Error", f"Padding failed:\n{str(e)}")
        return False

def split2blocks():
    global blocks, imgNp

    if imgNp is None:
        return None

    try:
        block_w = int(widthEn.get() or 8)
        block_h = int(heightEn.get() or 8)
        
        h, w, c = imgNp.shape
        blocks_arr = imgNp.reshape(h // block_h, block_h, w // block_w, block_w, 3)
        blocks_arr = blocks_arr.swapaxes(1, 2).reshape(-1, block_h, block_w, 3)
        blocks = blocks_arr
        return blocks
    
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to split blocks:\n{str(e)}")
        return None

def vector_quantization(blocks, n_clusters=256):
    try:
        n_blocks, bh, bw, c = blocks.shape
        flattened = blocks.reshape(n_blocks, -1)
        n_clusters = min(n_clusters, n_blocks)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(flattened)
        codebook = kmeans.cluster_centers_.reshape(n_clusters, bh, bw, c)
        
        return codebook, labels
        
    except Exception as e:
        messagebox.showerror("Error", f"Vector quantization failed:\n{str(e)}")
        return None, None

def save_compressed_data(codebook, labels, block_w, block_h, padded_size, original_size, filepath):
    """Save compressed data to .bin file"""
    try:
        with open(filepath, 'wb') as f:
            # Write header information
            f.write(struct.pack('I', block_w))
            f.write(struct.pack('I', block_h))
            f.write(struct.pack('I', padded_size[0]))
            f.write(struct.pack('I', padded_size[1]))
            f.write(struct.pack('I', original_size[0]))
            f.write(struct.pack('I', original_size[1]))
            f.write(struct.pack('I', len(codebook)))
            f.write(struct.pack('I', len(labels)))
            
            for codeword in codebook:
                f.write(codeword.astype(np.uint8).tobytes())
 
            f.write(np.array(labels, dtype=np.uint16).tobytes())
        
        return True
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save .bin file:\n{str(e)}")
        return False

def save_codebook_json(codebook, filepath):
    """Save codebook to JSON file"""
    try:
        codebook_list = codebook.astype(int).tolist()
        
        with open(filepath, 'w') as f:
            json.dump({
                'codebook': codebook_list,
                'shape': list(codebook.shape)
            }, f, indent=2)
        
        return True
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save JSON file:\n{str(e)}")
        return False

def disable_buttons():
    """Disable all buttons except Clear"""
    uploadBtn.config(state='disabled')
    uploadBinBtn.config(state='disabled')
    processBtn.config(state='disabled')

def enable_buttons():
    """Enable all buttons"""
    uploadBtn.config(state='normal')
    uploadBinBtn.config(state='normal')
    processBtn.config(state='normal')

def process_compression():
    """Main compression process"""
    global is_processing
    
    if is_processing:
        return
    
    # Check if image is loaded
    if originalNp is None:
        messagebox.showwarning("No Image", "Please upload an image first!")
        return
    
    # Check block dimensions and show default message if empty
    block_w = widthEn.get().strip()
    block_h = heightEn.get().strip()
    
    if not block_w or not block_h:
        result = messagebox.askyesno(
            "Default Block Size",
            "Block dimensions not specified.\nUse default 8x8 blocks?",
            icon='question'
        )
        if result:
            widthEn.delete(0, tkinter.END)
            widthEn.insert(0, "8")
            heightEn.delete(0, tkinter.END)
            heightEn.insert(0, "8")
        else:
            return
    
    is_processing = True
    disable_buttons()
    
    try:
        # Step 1: Padding
        if not padding():
            enable_buttons()
            is_processing = False
            return
        
        # Step 2: Split into blocks
        blocks_arr = split2blocks()
        if blocks_arr is None:
            enable_buttons()
            is_processing = False
            return
        
        # Step 3: Vector quantization
        codebook, labels = vector_quantization(blocks_arr, n_clusters=256)
        if codebook is None or labels is None:
            enable_buttons()
            is_processing = False
            return
        
        # Step 4: Ask for save location
        save_path = filedialog.asksaveasfilename(
            title="Save compressed file as",
            defaultextension=".bin",
            filetypes=[("Binary Files", "*.bin")]
        )
        
        if not save_path:
            enable_buttons()
            is_processing = False
            return
        
        # Step 5: Save compressed data
        block_w_val = int(widthEn.get())
        block_h_val = int(heightEn.get())
        
        if not save_compressed_data(codebook, labels, block_w_val, block_h_val, 
                                     paddedSize, originalSize, save_path):
            enable_buttons()
            is_processing = False
            return
        
        # Step 6: Save codebook as JSON
        json_path = save_path.rsplit('.', 1)[0] + '_codebook.json'
        if not save_codebook_json(codebook, json_path):
            enable_buttons()
            is_processing = False
            return
        
    except Exception as e:
        messagebox.showerror("Error", f"Compression failed:\n{str(e)}")
    
    finally:
        enable_buttons()
        is_processing = False

def clear_all():
    """Clear all data and reset UI"""
    global originalImage, originalNp, originalSize, imgNp, paddedImg, paddedSize, blocks
    
    # Reset global variables
    originalImage = None
    originalNp = None
    originalSize = None
    imgNp = None
    paddedImg = None
    paddedSize = None
    blocks = None
    
    # Clear image displays
    original_label.configure(image="", text="Original Image")
    proceLabel.configure(image="", text="Processed Image\n(Compressed/Decompressed)")
    
    # Clear entry fields
    widthEn.delete(0, tkinter.END)
    heightEn.delete(0, tkinter.END)
    
    # Enable all buttons
    enable_buttons()

def uploadBINfile():
    return
def Save_files():
    return
def create_gui():
    global original_label, proceLabel, widthEn, heightEn
    global uploadBtn, uploadBinBtn, processBtn, clearBtn
    
    top_frame = tkinter.Frame(root, bg="#f0f0f0")
    top_frame.pack(pady=10, fill="x")
    
    left_frame = tkinter.Frame(top_frame, bg="#f0f0f0")
    left_frame.pack(side=tkinter.LEFT)
    uploadBtn = tkinter.Button(
        left_frame, text="Upload image to compress", 
        command=upload, width=25, height=1, bg="white"
    )
    uploadBtn.grid(row=0, column=0, padx=20, pady=20)

    uploadBinBtn = tkinter.Button(
        left_frame, text="Upload .bin to decompress", 
        command=uploadBINfile, width=25, height=1, bg="white"
    )
    uploadBinBtn.grid(row=0, column=1, padx=20, pady=20)

    right_frame = tkinter.Frame(top_frame, bg="#f0f0f0")
    right_frame.pack(side=tkinter.RIGHT, padx=20)

    clearBtn = tkinter.Button(
        right_frame, text="Clear", width=15, height=1, 
        command=clear_all, bg="white", fg="black", font=(11), bd=1
    )
    clearBtn.pack(padx=20,pady=20, side=tkinter.RIGHT)
    saveBtn = tkinter.Button(
        right_frame, text="Save", width=15, height=1, 
        command=Save_files, bg="white", fg="black", font=(11), bd=1
    )
    saveBtn.pack(padx=20,pady=20,side=tkinter.LEFT)

    images_frame = tkinter.Frame(root, bg="#f0f0f0")
    images_frame.pack(pady=20)
    
    original_panel = tkinter.Frame(
        images_frame, width=PANEL_W, height=PANEL_H, 
        bg="white", bd=5, highlightthickness=5
    )
    original_panel.grid(row=0, column=0, padx=10)
    original_panel.pack_propagate(False)
    
    original_label = tkinter.Label(
        original_panel, text="Original Image", 
        font=(20), bg="white"
    )
    original_label.pack(expand=True, fill="both")
    
    processed_panel = tkinter.Frame(
        images_frame, width=PANEL_W, height=PANEL_H,
        bg="white", bd=5, highlightthickness=5
    )
    processed_panel.grid(row=0, column=1, padx=10)
    processed_panel.pack_propagate(False)
    
    proceLabel = tkinter.Label(
        processed_panel, 
        text="Processed Image\n(Compressed/Decompressed)", 
        font=(20), bg="white"
    )
    proceLabel.pack(expand=True, fill="both")
    
    con = tkinter.Frame(root, bg="#f0f0f0")
    con.pack(padx=20, pady=30, side=tkinter.LEFT)
    
    tkinter.Label(
        con, text="Block Width: ", 
        bg="#f0f0f0", font=(12)
    ).grid(row=0, column=0, padx=5, pady=10)
    
    widthEn = tkinter.Entry(con, width=8, font=("Arial", 12), bd=1)
    widthEn.grid(row=0, column=1, padx=5, pady=10)
    
    tkinter.Label(
        con, text="Block Height: ", 
        bg="#f0f0f0", font=(12)
    ).grid(row=1, column=0, padx=5, pady=10)
    
    heightEn = tkinter.Entry(con, width=8, font=("Arial", 12), bd=1)
    heightEn.grid(row=1, column=1, padx=5, pady=10)
    
    runningButton = tkinter.Frame(root, bg="#f0f0f0")
    runningButton.pack(padx=20, side=tkinter.RIGHT)
    
    processBtn = tkinter.Button(
        runningButton, text="Process ▶", 
        command=process_compression,
        width=15, height=3, bg="white", fg="black", 
        font=(11), bd=1
    )
    processBtn.grid(row=0, column=0, padx=20, pady=20)

# Main application
root = tkinter.Tk()
root.title("Vecto")

script_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(script_dir, "icon.png")

try:
    icon = ImageTk.PhotoImage(Image.open(icon_path))
    root.iconphoto(False, icon)
except:
    pass  # Icon not found, continue without it

create_gui()
root.mainloop()