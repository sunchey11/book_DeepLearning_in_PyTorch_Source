# 常用的汉字
import os
import pygame

# https://codeleading.com/article/14385742689/
def GBK2312(head,body):
    val = f'{head:x} {body:x}'
    str = bytes.fromhex(val).decode('gb2312',errors="ignore")
    return str
def save_font_png(path, fontPath, fontName):
    if not os.path.exists(path):
        os.makedirs(path)

    pygame.init()
    # 256是生成汉字的字体大小
    font = pygame.font.Font(fontPath, 256)
    h_start,h_end = (0xb0, 0xf7) # 汉字编码范围
    b_start,b_end = (0xa1, 0xfe) # 汉字编码范围
    count = 0
    size = (h_end-h_start)*(b_end-b_start)
    for h in range(int(h_start), int(h_end)):
        for b in range(int(b_start), int(b_end)):
            word = GBK2312(h,b)
            if word == '':
                continue
            rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
            dir = os.path.join(path,word)
            if not os.path.exists(dir):
                os.makedirs(dir)
            pygame.image.save(rtext, os.path.join(dir, fontName+"_"+word + ".png"))
            count +=1
            if count % 1000 ==0:
                print(str(count)+"/"+str(size)," saved")
    print("all "+str(count)," saved")

a = GBK2312(215,250)
print(a)
chinese_path = "D:\\pytorch_data\\font_to_png\\train_gbk"


simfang_font_path = "C:\\Windows\\Fonts\\simfang.ttf"
save_font_png(chinese_path, simfang_font_path,"simfang")

# stsong_font_path = "C:\\Windows\\Fonts\\STSONG.TTF"
# save_font_png(chinese_path,stsong_font_path,"stsong")

# 当前目录下要有微软雅黑的字体文件msyh.ttc,或者去c:\Windows\Fonts目录下找
# msyh_font_path = "C:\\Windows\\Fonts\\msyh.ttc"
# save_font_png(chinese_path,msyh_font_path,"msyh")

# simhei_font_path = "C:\\Windows\\Fonts\\simhei.ttf"
# save_font_png(chinese_path,simhei_font_path,"simhei")

