from rembg import remove
from PIL import Image
import os

for i in os.listdir(r'D:\DATN\Detect Object and Failure Object\Project Push Git\Image'):
    j = i.rsplit('.', maxsplit=1)[0]
    input_path = r'D:\DATN\Detect Object and Failure Object\Project Push Git\Image\\' + i
    output_path = r'D:\DATN\Detect Object and Failure Object\Project Push Git\Result Remove Background\\' + j + ".png"
    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)