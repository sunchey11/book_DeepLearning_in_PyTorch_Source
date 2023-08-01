# Input and output as a PIL image

from rembg import remove,new_session
from PIL import Image
import os
import time

input_path = 'r4.jpg'
output_path = 'r4-output.png'

file_dir = os.path.split(os.path.abspath(__file__))[0]
input_path = os.path.join(file_dir, input_path)
output_path = os.path.join(file_dir, output_path)

input = Image.open(input_path)
print(type(input))
# 报错：retrieve() got an unexpected keyword argument 'progressbar'
# 重新安装pip install --upgrade pooch
session = new_session()
output = remove(input, session=session)
start=time.time()
output = remove(input, session=session)
end=time.time()
# 时间为0.05秒左右
print('程序运行时间为: %s Seconds'%(end-start))

print(type(output))
output.save(output_path)