import numpy as np
import cv2 as cv

file = 'oo/biao'
def open_jpg(file):
    jpg = cv.imread(file+'.jpg')
    jz_jpg = np.array(cv.cvtColor(jpg, cv.COLOR_BGR2GRAY), dtype=int)
    shape = jz_jpg.shape
    print(shape)

    jz_jpg = jz_jpg.reshape(1 , jz_jpg.size)
    return jz_jpg[0], shape, jz_jpg.size

jz_jpg, shape, length = open_jpg(file)
print(jz_jpg, shape)
pgm_txt = 'P2\n'+str(shape[1])+' '+str(shape[0])+'\n'+'255'+'\n'
for i in range(int(length/10)+1):
    if length-i*10 > 10:
        for j in jz_jpg[i*10 : i*10+10]:
            pgm_txt += str(j)+' '
        pgm_txt += '\n'
    elif 10 >= length-i*10 > 0:
        for j in jz_jpg[i*10 :]:
            pgm_txt += str(j)+' '
        pgm_txt += '\n'
print(pgm_txt)

pgm_file= open(file+'.pgm','wb+')
pgm_file.write(pgm_txt.encode("ascii"))
pgm_file.close()
