from rembg import remove
import cv2
import os
import time

input_path = 'r4.jpg'
output_path = 'r4-output.png'

file_dir = os.path.split(os.path.abspath(__file__))[0]
input_path = os.path.join(file_dir, input_path)
output_path = os.path.join(file_dir, output_path)



input = cv2.imread(input_path)


start=time.time()
output = remove(input)
end=time.time()
# 时间为0.05秒左右
print('程序运行时间为: %s Seconds'%(end-start))

cv2.imwrite(output_path, output)