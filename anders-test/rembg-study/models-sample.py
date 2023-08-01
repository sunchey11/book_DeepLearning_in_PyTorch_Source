# Input and output as a PIL image

from rembg import remove,new_session
from PIL import Image
import os
import time

input_path = 'r4.jpg'


file_dir = os.path.split(os.path.abspath(__file__))[0]
input_path = os.path.join(file_dir, input_path)

input = Image.open(input_path)
for model in [
        "u2net",
        "u2netp",#这个最快
        "u2net_human_seg",
        "u2net_cloth_seg",
        "silueta",
        "isnet-general-use",
        "isnet-anime",
        # "sam" #这个需要参数，不懂，会报错
    ]:
    output_path = 'r4-'+model+'-output.png'
    output_path = os.path.join(file_dir, output_path)    
    session = new_session(model)
    output = remove(input, session=session)
    start=time.time()
    output = remove(input, session=session)
    
    end=time.time()
    # 时间为0.05秒左右
    print('程序运行时间为: %s Seconds'%(end-start),model)
    output.save(output_path)
