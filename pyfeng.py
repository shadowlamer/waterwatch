import os
import warnings
from io import BytesIO
import directio
from PIL import Image

IMG_OFFSET_BYTES = 8704

class PyFeng:
    def __init__(self, file="/dev/zero"):
        self.file = file

    def get(self):
        fd = -1
        try:
            file_size = os.stat(self.file).st_size
            fd = os.open(self.file, os.O_RDONLY | os.O_DIRECT | os.O_SYNC)
            os.lseek(fd, IMG_OFFSET_BYTES, os.SEEK_SET)
            im_data = directio.read(fd, file_size - IMG_OFFSET_BYTES)
            image = Image.open(BytesIO(im_data))
            warnings.resetwarnings()
            return image
        except Exception as e:
            warnings.warn(str(e))
        if fd != -1:
            os.close(fd)
        return None
