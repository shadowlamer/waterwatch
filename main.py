from pyfeng import PyFeng
from model import Model, ResNet, BasicBlock
from PIL import ImageDraw, ImageFont

CAMERA_IMAGE_PATH = "/media/sl/5221-0001/VSFILE"
DEVICE = "cpu"
MODEL_PATH = "mynet_40_1.net"

model = Model(device=DEVICE, model_path=MODEL_PATH)
camera = PyFeng(CAMERA_IMAGE_PATH, size=(720, 360))

while True:
    image = camera.get()
    if image is not None:
        volume, percent = model.process(image)
        print(volume, percent)

        result = ImageDraw.Draw(image)
        myFont = ImageFont.truetype('FreeMono.ttf', 36)
        result.text((30, 320), f"Объем: {volume}л., Наполнение: {percent}%",
                    font=myFont, fill=(255, 0, 0), stroke_fill=(255, 0, 0), stroke_width=1)

        image.save('result.jpg')

