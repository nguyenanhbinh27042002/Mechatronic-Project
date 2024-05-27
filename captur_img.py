import cv2
from snap7.util import *
from snap7.types import *
import snap7.client as c
import snap7
import time

def WriteMemory(plc, byte, bit, datatype, value):
    result = plc.read_area(Areas.MK, 0, byte, datatype)
    if datatype == S7WLBit:
        set_bool(result, 0, bit, value)
    elif datatype == S7WLByte or datatype == S7WLWord:
        set_int(result, 0, value)
    elif datatype == S7WLReal:
        set_real(result, 0, value)
    elif datatype == S7WLDWord:
        set_dword(result, 0, value)
    plc.write_area(Areas.MK, 0, byte, result)


def ReadMemory(plc, byte, bit, datatype):
    result = plc.read_area(Areas.MK, 0, byte, datatype)
    if datatype == S7WLBit:
        return get_bool(result, 0, bit)
    elif datatype == S7WLByte or datatype == S7WLWord:
        return get_int(result, 0)
    elif datatype == S7WLReal:
        return get_real(result, 0)
    elif datatype == S7WLDWord:
        return get_dword(result, 0)
    else:
        return None

def capture(plc, byte, bit, datatype):
    folder_img = "/home/pi/Mechatronics_Project/Mechatronics-Project/Image"
    video=  cv2.VideoCapture(0)
    # success, img = video.read()
    count =0
    while(True):
        sensor=ReadMemory(plc,byte,bit,datatype)
        ret,frame = video.read()
        cv2.imshow("frame_video",frame)
        print(f"sensor:{sensor}")
        if sensor == True:
            count+=1
            cv2.imwrite(folder_img+"/"+"sample No."+str(count) +".JPG", frame)
            print('capture success......................')
            # time.sleep(1)
        if cv2.waitKey(1)==ord('q'):
            break

    
if __name__ == "__main__":
    plc = c.Client()
    plc.connect('192.168.0.1', 0, 1)
    print(plc.get_connected())
    byte =0
    bit =0
    datatype=S7WLBit
    capture(plc, byte,bit,datatype)

    
    #WriteMemory(plc, 0, 0, S7WLBit, 0)
    #print('Done')
