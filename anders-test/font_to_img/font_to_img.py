import os
import pygame


def save_font_png(path, fontPath, fontName):
    if not os.path.exists(path):
        os.makedirs(path)

    pygame.init()
    # 256是生成汉字的字体大小
    font = pygame.font.Font(fontPath, 256)
    start,end = (0x4E00, 0x9FA5) # 汉字编码范围
    count = 0
    size = end - start
    for codepoint in range(int(start), int(end)):
        word = chr(codepoint)
        
        rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
        dir = os.path.join(path,word)
        if not os.path.exists(dir):
            os.makedirs(dir)
        pygame.image.save(rtext, os.path.join(dir, fontName+"_"+word + ".png"))
        count +=1
        if count % 1000 ==0:
            print(str(count)+"/"+str(size)," saved")
    print("all "+str(size)," saved")
chinese_path = "D:\\pytorch_data\\font_to_png"


simfang_font_path = "C:\\Windows\\Fonts\\simfang.ttf"
save_font_png(chinese_path, simfang_font_path,"simfang")

# stsong_font_path = "C:\\Windows\\Fonts\\STSONG.TTF"
# save_font_png(chinese_path,stsong_font_path,"stsong")

# 当前目录下要有微软雅黑的字体文件msyh.ttc,或者去c:\Windows\Fonts目录下找
# msyh_font_path = "C:\\Windows\\Fonts\\msyh.ttc"
# save_font_png(chinese_path,msyh_font_path,"msyh")

# simhei_font_path = "C:\\Windows\\Fonts\\simhei.ttf"
# save_font_png(chinese_path,simhei_font_path,"simhei")