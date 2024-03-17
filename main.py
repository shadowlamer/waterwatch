from pyfeng import PyFeng
from model import Model, ResNet, BasicBlock
from PIL import ImageDraw, ImageFont

CAMERA_IMAGE_PATH = "/media/sl/5221-0001/VSFILE"
DEVICE = "cpu"
MODEL_PATH = "mynet.net"

model = Model(device=DEVICE, model_path=MODEL_PATH)
camera = PyFeng(CAMERA_IMAGE_PATH)

while True:
    image = camera.get()
    if image is not None:
        volume, percent = model.process(image)

        result = ImageDraw.Draw(image)
        myFont = ImageFont.truetype('FreeMono.ttf', 36)
        result.text((30, 320), f"Объем: {volume}л., Наполнение: {percent}%",
                    font=myFont, fill=(255, 255, 255), stroke_fill=(255, 255, 255), stroke_width=1)

        image.save('result.jpg')

