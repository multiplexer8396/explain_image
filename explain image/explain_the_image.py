from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import messagebox

# Model ve islemciyi yukle / upload model and processor
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# goruntu yukle / upload photo
image_path = "C:/Users/ethem/Downloads/323101.jpg"  # your photo's path
image = Image.open(image_path).convert("RGB")

# goruntuyu isle ve modelden aciklama uret / process image and generate comment
inputs = processor(image, return_tensors="pt")
output = model.generate(**inputs)

# read comment
caption = processor.decode(output[0], skip_special_tokens=True)

# Yorumu resim uzerine ekle / add comment on image
draw = ImageDraw.Draw(image)

# Yazı tipi ve boyutu / text type and scale 
try:
    font = ImageFont.truetype("arial.ttf", 20)  # i choose text type arial " (20) number is scale of text, you can change for your image "
except IOError:                                                                 # but if you change scale number, must check (image.height - 40)
    font = ImageFont.load_default()  # if there isn't arial use anything       

# text position on image
text_position = (10, image.height - 40)  
text_color = (255, 255, 255)  # white

# draw text
draw.text(text_position, caption, fill=text_color, font=font)

# show image
image.show()

# Tkinter pop-up message
root = tk.Tk()
root.withdraw()  
messagebox.showinfo("Artificial Intelligence Commentary", caption)
