import os
import pygame as pg

def abs_path(name):
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    imgpath = os.path.join(file_dir, name)
    return imgpath

def save_font_png(path, fontPath, filename, word, size):
    if not os.path.exists(path):
        os.makedirs(path)

    pg.init()

    image = pg.image.load(abs_path('bg.png'))
    # size是生成汉字的字体大小
    font = pg.font.Font(fontPath, size)
    
    # 竖排汉字
    for i in range(len(word)):
        rtext = font.render(word[i], True, (0, 0, 0,255))
        x = (image.get_width()-size)/2
        
        y_offset = (image.get_height()-size*len(word))/2
        y = y_offset + i*size
        image.blit(rtext, (x, y))
    pg.image.save(image, os.path.join(path, filename + ".png"))


ouput_path = "D:\\GitHub\\learn_scratch\\map_puzzle\\sheng_mingzi"
simfang_font_path = "C:\\Windows\\Fonts\\simfang.ttf"
# save_font_png(ouput_path, simfang_font_path,"simfang")

# stsong_font_path = "C:\\Windows\\Fonts\\STSONG.TTF"
# save_font_png(chinese_path,stsong_font_path,"stsong")

# 当前目录下要有微软雅黑的字体文件msyh.ttc,或者去c:\Windows\Fonts目录下找
# msyh_font_path = "C:\\Windows\\Fonts\\msyh.ttc"
# save_font_png(chinese_path,msyh_font_path,"msyh")

# simhei_font_path = "C:\\Windows\\Fonts\\simhei.ttf"
# save_font_png(chinese_path,simhei_font_path,"simhei")


def write_txt(code, text,size):
    save_font_png(ouput_path, simfang_font_path, code, text,size)


names = [
    '台湾',
    '河北',
    '山西',
    '内蒙古',
    '辽宁',
    '吉林',
    '黑龙江',
    '江苏',
    '浙江',
    '安徽',
    '福建',
    '江西',
    '山东',
    '河南',
    '湖北',
    '湖南',
    '广东',
    '广西',
    '海南',
    '四川',
    '贵州',
    '云南',
    '西藏',
    '陕西',
    '甘肃',
    '青海',
    '宁夏',
    '新疆',
    '北京',
    '天津',
    '上海',
    '重庆',
    '香港',
    '澳门',
]
b_size = 250
s_size = 160
# write_txt('xinjiang', "内蒙古",s_size)
# write_txt('xizang', "西藏",b_size)

for i in range(len(names)):
    txt = names[i]
    if len(txt) == 2:
        size = b_size
    else:
        size = s_size
    write_txt(txt, txt, size)