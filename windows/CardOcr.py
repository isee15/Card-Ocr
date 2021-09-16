from ctypes import *
import os

def imageToJsonString(path):
    dllLib = CDLL("./CardOcrLib.dll")
    return dllLib.imageToJsonString(path)

if __name__=="__main__":
    os.system("chcp 65001")
    print(imageToJsonString("test0.png"))
