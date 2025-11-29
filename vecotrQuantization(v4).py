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
current_codebook = None
current_labels = None
current_reconstructed = None
mode = "compress"

def upload():
    global originalImage, originalNp, originalSize, mode
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
        
        mode = "compress"
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
def vector_quantization(blocks, n_clusters):
    try:
        n_blocks, bh, bw, c = blocks.shape
        flat = blocks.reshape(n_blocks, -1)
        n_clusters = min(n_clusters, n_blocks)

        means = flat.mean(axis=1)
        sorted_idx = np.argsort(means)
        sorted_blocks = flat[sorted_idx]

        groups = np.array_split(sorted_blocks, n_clusters)
        codebook = np.array([g.mean(axis=0) for g in groups], dtype=np.float32)
        diff = np.sum(np.abs(flat[:, None] - codebook[None, :]), axis=2)
        labels = np.argmin(diff, axis=1)
        codebook = codebook.reshape(n_clusters, bh, bw, c).astype(np.uint8)

        return codebook, labels

    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None, None

def save_compressed_data(codebook, labels, block_w, block_h, padded_size, original_size, filepath):
    try:
        with open(filepath, 'wb') as f:
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

def reconstruct_image(codebook, labels, block_h, block_w, padded_size, original_size):
    try:
        reconstructed_blocks = codebook[labels]
        
        w_blocks = padded_size[0] // block_w
        h_blocks = padded_size[1] // block_h
        reconstructed = reconstructed_blocks.reshape(h_blocks, w_blocks, block_h, block_w, 3)
        reconstructed = reconstructed.swapaxes(1, 2).reshape(padded_size[1], padded_size[0], 3)
        reconstructed = reconstructed[:original_size[1], :original_size[0], :]
        
        return reconstructed.astype(np.uint8)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to reconstruct image:\n{str(e)}")
        return None

def disable_buttons():
    uploadBtn.config(state='disabled')
    uploadBinBtn.config(state='disabled')
    processBtn.config(state='disabled')

def enable_buttons():
    uploadBtn.config(state='normal')
    uploadBinBtn.config(state='normal')
    processBtn.config(state='normal')

def process_compression():
    global is_processing, current_codebook, current_labels, current_reconstructed
    
    if is_processing:
        return
    
    if mode == "decompress":
        process_decompression()
        return
    if originalNp is None:
        messagebox.showwarning("No Image", "Please upload an image first!")
        return
    block_w = widthEn.get().strip()
    block_h = heightEn.get().strip()
    n_clusters_input = numBLOCKen.get().strip()
    n_clusters_val = 256
    if n_clusters_input:
        try:
            n_clusters_val = int(n_clusters_input)
            if n_clusters_val <= 0:
                messagebox.showerror("Invalid Input", "Number of blocks must be positive!")
                return
            elif n_clusters_val > 500:
                messagebox.showerror("Invalid Input", "Number of blocks is very big!")
                return

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for number of blocks!")
            return
    
    if not block_w:
        widthEn.delete(0, tkinter.END)
        widthEn.insert(0, "8")
    
    if not block_h:
        heightEn.delete(0, tkinter.END)
        heightEn.insert(0, "8")
    
    is_processing = True
    disable_buttons()
    
    try:
        if not padding():
            enable_buttons()
            is_processing = False
            return
        
        blocks_arr = split2blocks()
        if blocks_arr is None:
            enable_buttons()
            is_processing = False
            return
        
        codebook, labels = vector_quantization(blocks_arr, n_clusters=n_clusters_val)
        if codebook is None or labels is None:
            enable_buttons()
            is_processing = False
            return

        current_codebook = codebook
        current_labels = labels

        block_w_val = int(widthEn.get())
        block_h_val = int(heightEn.get())
        
        reconstructed = reconstruct_image(codebook, labels, block_h_val, block_w_val, 
                                          paddedSize, originalSize)
        
        if reconstructed is not None:
            current_reconstructed = reconstructed
            reconstructed_img = Image.fromarray(reconstructed)
            display_img = ImageOps.contain(reconstructed_img, (PANEL_W, PANEL_H), 
                                          Image.Resampling.LANCZOS)
            bg = Image.new("RGB", (PANEL_W, PANEL_H), (255, 255, 255))
            x = (PANEL_W - display_img.width) // 2
            y = (PANEL_H - display_img.height) // 2
            bg.paste(display_img, (x, y))
            photo = ImageTk.PhotoImage(bg)
            proceLabel.configure(image=photo, text="")
            proceLabel.image = photo
            
            original_size_bytes = originalSize[0] * originalSize[1] * 3
            codebook_size = len(codebook) * block_w_val * block_h_val * 3
            labels_size = len(labels) * 2
            compressed_size_bytes = codebook_size + labels_size + 32
            ratio = original_size_bytes / compressed_size_bytes
            
            messagebox.showinfo(
                "Compression Complete",
                f"Image compressed successfully!\n\n"
                f"Block size: {block_w_val}×{block_h_val}\n"
                f"Compression ratio: {ratio:.2f}:1\n\n"
                f"Press 'Save' to save the compressed file."
            )
        
    except Exception as e:
        messagebox.showerror("Error", f"Compression failed:\n{str(e)}")
    
    finally:
        enable_buttons()
        is_processing = False

def process_decompression():
    global current_reconstructed, decompress_data
    
    if 'decompress_data' not in globals() or decompress_data is None:
        messagebox.showwarning("No Data", "Please upload a .bin file first!")
        return
    
    try:
        reconstructed = reconstruct_image(
            decompress_data['codebook'],
            decompress_data['labels'],
            decompress_data['block_h'],
            decompress_data['block_w'],
            decompress_data['padded_size'],
            decompress_data['original_size']
        )
        
        if reconstructed is not None:
            current_reconstructed = reconstructed
            reconstructed_img = Image.fromarray(reconstructed)
            display_img = ImageOps.contain(reconstructed_img, (PANEL_W, PANEL_H), 
                                          Image.Resampling.LANCZOS)
            bg = Image.new("RGB", (PANEL_W, PANEL_H), (255, 255, 255))
            x = (PANEL_W - display_img.width) // 2
            y = (PANEL_H - display_img.height) // 2
            bg.paste(display_img, (x, y))
            photo = ImageTk.PhotoImage(bg)
            proceLabel.configure(image=photo, text="")
            proceLabel.image = photo
            
            messagebox.showinfo("Success", "Image decompressed successfully!\n\nPress 'Save' to save the image.")
    
    except Exception as e:
        messagebox.showerror("Error", f"Decompression failed:\n{str(e)}")

def save_files():
    global current_codebook, current_labels
    
    if mode == "decompress":
        if current_reconstructed is None:
            messagebox.showwarning("No Image", "Please process the decompression first!")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save reconstructed image as",
            defaultextension=".jpg",
            filetypes=[("JPEG Files", "*.jpg *.jpeg")]
        )
        
        if not save_path:
            return
        
        try:
            reconstructed_img = Image.fromarray(current_reconstructed)
            reconstructed_img.save(save_path, format='JPEG', quality=85, optimize=True)
            saved_size = os.path.getsize(save_path)
            saved_size_kb = saved_size / 1024
            messagebox.showinfo(
                "Success", 
                f"Image saved successfully!\n\n"
                f"File: {os.path.basename(save_path)}\n"
                f"Size: {saved_size_kb:.2f} KB"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
        return
    if current_codebook is None or current_labels is None:
        messagebox.showwarning("No Data", "Please process the compression first!")
        return
    
    save_path = filedialog.asksaveasfilename(
        title="Save compressed file as",
        defaultextension=".bin",
        filetypes=[("Binary Files", "*.bin")]
    )
    
    if not save_path:
        return
    
    try:
        block_w_val = int(widthEn.get() or 8)
        block_h_val = int(heightEn.get() or 8)
        
        if not save_compressed_data(current_codebook, current_labels, block_w_val, block_h_val, 
                                     paddedSize, originalSize, save_path):
            return
        
        json_path = save_path.rsplit('.', 1)[0] + '_codebook.json'
        if not save_codebook_json(current_codebook, json_path):
            return
        
        original_size_bytes = originalSize[0] * originalSize[1] * 3
        compressed_size_bytes = os.path.getsize(save_path)
        ratio = original_size_bytes / compressed_size_bytes
        
        messagebox.showinfo(
            "Save Complete",
            f"Files saved successfully!\n\n"
            f"Compressed file: {os.path.basename(save_path)}\n"
            f"Codebook file: {os.path.basename(json_path)}\n\n"
            f"Compression ratio: {ratio:.2f}:1"
        )
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save files:\n{str(e)}")
    clear_all()

def clear_all():
    global originalImage, originalNp, originalSize, imgNp, paddedImg, paddedSize, blocks
    
    originalImage = None
    originalNp = None
    originalSize = None
    imgNp = None
    paddedImg = None
    paddedSize = None
    blocks = None
    
    original_label.configure(image="", text="Original Image")
    proceLabel.configure(image="", text="Processed Image\n(Compressed/Decompressed)")
    widthEn.delete(0, tkinter.END)
    heightEn.delete(0, tkinter.END)
    numBLOCKen.delete(0, tkinter.END)
    enable_buttons()

def uploadBINfile():
    global mode, current_codebook, current_labels, originalImage, originalNp, originalSize
    
    filepath = filedialog.askopenfilename(
        title="Select compressed .bin file",
        filetypes=[("Binary Files", "*.bin"), ("All Files", "*.*")]
    )
    
    if not filepath:
        return

    if not filepath.lower().endswith(".bin"):
        messagebox.showerror("Invalid file", "Please select a valid .bin compressed file!")
        return

    try:
        mode = "decompress"
        
        with open(filepath, 'rb') as f:
            block_w = struct.unpack('I', f.read(4))[0]
            block_h = struct.unpack('I', f.read(4))[0]
            padded_w = struct.unpack('I', f.read(4))[0]
            padded_h = struct.unpack('I', f.read(4))[0]
            orig_w = struct.unpack('I', f.read(4))[0]
            orig_h = struct.unpack('I', f.read(4))[0]
            n_codewords = struct.unpack('I', f.read(4))[0]
            n_blocks = struct.unpack('I', f.read(4))[0]
            codebook_size = n_codewords * block_h * block_w * 3
            codebook_data = f.read(codebook_size)
            codebook = np.frombuffer(codebook_data, dtype=np.uint8)
            codebook = codebook.reshape(n_codewords, block_h, block_w, 3)
            
            labels_data = f.read(n_blocks * 2)
            labels = np.frombuffer(labels_data, dtype=np.uint16)
        
        global decompress_data
        decompress_data = {
            'codebook': codebook,
            'labels': labels,
            'block_w': block_w,
            'block_h': block_h,
            'padded_size': (padded_w, padded_h),
            'original_size': (orig_w, orig_h)
        }
        
        widthEn.delete(0, tkinter.END)
        widthEn.insert(0, str(block_w))
        heightEn.delete(0, tkinter.END)
        heightEn.insert(0, str(block_h))
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        agreement_path = os.path.join(script_dir, "aggrement.png")
        
        if os.path.exists(agreement_path):
            agreement_img = Image.open(agreement_path).convert("RGB")
            display_img = ImageOps.contain(agreement_img, (PANEL_W, PANEL_H), Image.Resampling.LANCZOS)
            bg = Image.new("RGB", (PANEL_W, PANEL_H), (255, 255, 255))
            x = (PANEL_W - display_img.width) // 2
            y = (PANEL_H - display_img.height) // 2
            bg.paste(display_img, (x, y))
            photo = ImageTk.PhotoImage(bg)
            original_label.configure(image=photo, text="")
            original_label.image = photo
        else:
            original_label.configure(image="", text="Compressed File Loaded\n(aggrement.png not found)")
        proceLabel.configure(image="", text="Processed Image\n(Compressed/Decompressed)")
        
        messagebox.showinfo("File Loaded", f"Compressed file loaded successfully!\n\nPress 'Process' to decompress.")
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load .bin file:\n{str(e)}")

def create_gui():
    global original_label, proceLabel, widthEn, heightEn,numBLOCKen
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
    clearBtn.pack(padx=20, pady=20, side=tkinter.RIGHT)
    
    saveBtn = tkinter.Button(
        right_frame, text="Save", width=15, height=1, 
        command=save_files, bg="white", fg="black", font=(11), bd=1
    )
    saveBtn.pack(padx=20, pady=20, side=tkinter.LEFT)

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

    tkinter.Label(
        con, text="number of CodeBook: ", 
        bg="#f0f0f0", font=(12)
    ).grid(row=1, column=2, padx=5, pady=10)
    
    numBLOCKen = tkinter.Entry(con, width=8, font=("Arial", 12), bd=1)
    numBLOCKen.grid(row=1, column=3, padx=5, pady=10)

    
    runningButton = tkinter.Frame(root, bg="#f0f0f0")
    runningButton.pack(padx=20, side=tkinter.RIGHT)
    
    processBtn = tkinter.Button(
        runningButton, text="Process ▶", 
        command=process_compression,
        width=15, height=3, bg="white", fg="black", 
        font=(11), bd=1
    )
    processBtn.grid(row=0, column=0, padx=20, pady=20)

root = tkinter.Tk()
root.title("Vecto")

script_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(script_dir, "icon.png")

try:
    icon = ImageTk.PhotoImage(Image.open(icon_path))
    root.iconphoto(False, icon)
except:
    pass

create_gui()
root.mainloop()