from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import torch

# Model ve işlemciyi yükle / upload model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU varsa kullan, yoksa CPU kullan / Use GPU if available, otherwise use CPU
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

video_path = " your video path "

# Kamera başlat / Start the camera
cap = cv2.VideoCapture(video_path)  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alinamiyor. cikiliyor.")  # "Unable to capture frame from camera. Exiting."
        break

    # Kameradan alınan tüm görüntüyü PIL formatına çevir / Convert the captured frame to PIL format
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # BLIP ile yorum üret / Generate caption with BLIP
    inputs = processor(frame_pil, return_tensors="pt").to(device)  # Giriş verisini modele uygun cihaza taşı / Move the input data to the correct device
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)

    # Yorum ekle / Add the caption to the frame
    cv2.putText(frame, caption, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Görüntüyü göster / Show the frame
    cv2.imshow("Kamera", frame)

    # 'q' tuşu ile çıkış / Exit with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak / Release resources
cap.release()
cv2.destroyAllWindows()
