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
        rtext = font.render(word[i], True, (0, 0, 0, 255))
        x = (image.get_width() - size) / 2

        y_offset = (image.get_height() - size * len(word)) / 2
        y = y_offset + i * size
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


def write_txt(code, text):
    one_size = 280
    two_size = 250
    three_size = 160
    four_size = 120
    if len(text) == 1:
        size = one_size
    elif len(text) == 2:
        size = two_size
    elif len(text) == 3:
        size = three_size
    elif len(text) == 4:
        size = four_size

    save_font_png(ouput_path, simfang_font_path, code, text, size)


names = [
    ['台湾', '台北', '台'],
    ['河北', '冀', '石家庄'],
    ['山西', '晋', '太原'],
    ['内蒙古', '内蒙古', '呼和浩特'],
    ['辽宁', '辽', '沈阳'],
    ['吉林', '吉', '长春'],
    ['黑龙江', '黑', '哈尔滨'],
    ['江苏', '苏', '南京'],
    ['浙江', '浙', '杭州'],
    ['安徽', '皖', '合肥'],
    ['福建', '闽', '福州'],
    ['江西', '赣', '南昌'],
    ['山东', '鲁', '济南'],
    ['河南', '豫', '郑州'],
    ['湖北', '鄂', '武汉'],
    ['湖南', '湘', '长沙'],
    ['广东', '粤', '广州'],
    ['广西', '桂', '南宁'],
    ['海南', '琼', '海口'],
    ['四川', '川', '成都'],
    ['贵州', '贵', '贵阳'],
    ['云南', '滇', '昆明'],
    ['西藏', '藏', '拉萨'],
    ['陕西', '秦', '西安'],
    ['甘肃', '陇', '兰州'],
    ['青海', '青', '西宁'],
    ['宁夏', '宁', '银川'],
    ['新疆', '新', '乌鲁木齐'],
    ['北京', '京', '北京'],
    ['天津', '津', '天津'],
    ['上海', '沪', '上海'],
    ['重庆', '渝', '重庆'],
    ['香港', '港', '香港'],
    ['澳门', '澳', '澳门'],
]

# write_txt('xinjiang', "内蒙古",s_size)
# write_txt('乌鲁木齐', "乌鲁木齐",four_size)

for i in range(len(names)):
    sheng = names[i][0]
    jian = names[i][1]
    hui = names[i][2]

    write_txt(sheng, sheng)
    write_txt(sheng + '_简', jian)
    write_txt(sheng + '_会', hui)
