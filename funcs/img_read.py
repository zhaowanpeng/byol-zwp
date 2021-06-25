#coding:utf-8
from PIL import Image, ImageFile
from io import BytesIO
import os, sys,cv2,warnings
import numpy as np
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__=["read_img","check_floder"]

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

#still exist part imgs can not be read
def read_img(imgpath,color=True):
    try:
        pic = Image.open(BytesIO(get_file_content(imgpath)))
        if color:
            pic = pic.convert("RGB")
        else:
            pic = pic.convert("L")
        return pic
    except:
        img=read_img_cv(imgpath,color)
        return img


def read_img_cv(imgpath,color=True):
    try:
        img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), 1)
        if color:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        return img
    except:
        return None

#floder is a path list type
def check_floder(floder):
    imgpaths = floder
    for path in imgpaths:
        imgs = os.listdir(path)
        img_num = len(imgs)
        print("checking path : {},      total img : {}".format(path, img_num))
        err_imgs = []
        for i, img_name in enumerate(imgs):
            img = read_img(path + img_name)
            if img == None:
                err_imgs.append(path + img_name)
            sys.stdout.write("\r checking: {} / {},    err : {}".format(i, img_num, len(err_imgs)))
            sys.stdout.flush()
        print("\ncomplete ! error number is {}".format(len(err_imgs)))
        print('\n'.join(str(i) for i in err_imgs))
        print("err img paths are saved to the err_read_img.txt")
        with open("err_read_img.txt", "w") as f:
            f.write('\n'.join(str(i) for i in err_imgs))
    return

def filter_cannot_read(imgs):
    read_success=[]
    read_fail=[]
    for imgpath in imgs:
        img = read_img(imgpath)
        if img == None:
            read_fail.append(imgpath)
            continue
        read_success.append(imgpath)
    print("total:{},  success:{},  fail:{}".format(len(imgs),len(read_success),len(read_fail)))
    with open("read_fail_img.txt", "w") as f:
        f.write('\n'.join(str(i) for i in read_fail))
    return read_success



