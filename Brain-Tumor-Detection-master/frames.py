import tkinter
from PIL import ImageTk
from PIL import Image

#set up đối tượng cửa sổ làm việc, bên trong bao gồm các thuộc tính, đặc trưng của đối tượng
class Frames:
    xAxis = 0 #Vị trí x của winframe
    yAxis = 0 #Vị trí y của winframe
    MainWindow = 0 #Cửa sổ hiển thị và làm việc hiển thị hình và các phím nhấn Close và View
    MainObj = 0 #Đối tượng chính(hình ảnh)
    winFrame = object() #Khung viền
    btnClose = object() #nút nhấn Close
    btnView = object() #nút nhấn View
    image = object() #Hình ảnh MRI
    method = object() #biến giá trị khi nhấn View Tumor Region
    callingObj = object() #Biến giá trị khi đã Browse ảnh
    labelImg = 0 #Hiển thị "Phát hiện khối U" và "Không Phát hiện khối U"

    def __init__(self, mainObj, MainWin, wWidth, wHeight, function, Object, xAxis=10, yAxis=10):
        self.xAxis = xAxis #gán x=10
        self.yAxis = yAxis #gán y=10
        self.MainWindow = MainWin 
        self.MainObj = mainObj
        self.MainWindow.title("Brain Tumor Detection") #Tiêu đề cửa sổ
        if (self.callingObj != 0): #gán giá trị cho biến callingObj khi đã Browser ảnh
            self.callingObj = Object

        if (function != 0): #gán giá trị cho biến function khi đã ấn View Tumor Region
            self.method = function

        global winFrame #cài đặt các giá trị của khung
        self.winFrame = tkinter.Frame(self.MainWindow, width=wWidth, height=wHeight) #nằm trong cửa sổ, chiều dài, chiều rộng
        self.winFrame['borderwidth'] = 5 #độ dày của đường viền
        self.winFrame['relief'] = 'ridge' #Kiểu đường viền "ridge" là kiểu đổ bóng
        self.winFrame.place(x=xAxis, y=yAxis) #Vị trí hiện khung

        #Cài đặt nút nhất Close
        self.btnClose = tkinter.Button(self.winFrame, text="Close", width=8, 
                                      command=lambda: self.quitProgram(self.MainWindow)) #Chức năng nút nhấn
        self.btnClose.place(x=1020, y=600)

        #Cài đặt nút nhất View
        self.btnView = tkinter.Button(self.winFrame, text="View", width=8, command=lambda: self.NextWindow(self.method))
        self.btnView.place(x=900, y=600)

    #Hàm con gán giá trị cho thuộc tính callingObj
    def setCallObject(self, obj):
        self.callingObj = obj

    #Hàm con gán giá trị cho thuộc tính method
    def setMethod(self, function):
        self.method = function

    #Hàm con đóng cửa sổ làm việc
    def quitProgram(self, window):
        global MainWindow #tạo Mainwindow là biến toàn cục
        self.MainWindow.destroy()

    #Hàm con 
    def getFrames(self):
        global winFrame #tạo winFrame là biến toàn cục
        return self.winFrame

    #Hàm con hiển thị khung
    def unhide(self):
        self.winFrame.place(x=self.xAxis, y=self.yAxis)

    #hàm con ẩn đối tượng
    def hide(self):
        self.winFrame.place_forget()

    #Hàm con sang  cửa sổ tiếp theo
    def NextWindow(self, methodToExecute):

        #list trang (0,1)
        listWF = list(self.MainObj.listOfWinFrame)

        #Kiểm tra xem có thêm ảnh và đã nhấn View Tumor Region hay chưa
        if (self.method == 0 or self.callingObj == 0):
            print("Calling Method or the Object from which Method is called is 0")
            return

        #Nếu đã nhấn thì gọi hàm con xử lý ảnh
        if (self.method != 1):
            methodToExecute()

        #Lấy ảnh đã thêm khi nhấn nút Browse
        if (self.callingObj == self.MainObj.DT):
            img = self.MainObj.DT.getImage()
        else:
            print("Error: No specified object for getImage() function")

        
        #lấy ảnh sau khi đã lọc nhiễu
        jpgImg = Image.fromarray(img)
        current = 0

        #Sang trang dùng vòng for
        for i in range(len(listWF)):
            listWF[i].hide() #ẩn hết nội dung ở cửa sổ trước
            if (listWF[i] == self):
                current = i #Gán giá trị trang

        if (current == len(listWF) - 1):  #Trang 2, current = 1
            listWF[current].unhide() #Hiện khung
            listWF[current].readImage(jpgImg)  #Đọc ảnh sau khi xử lý
            listWF[current].displayImage() #Hiện ảnh
            self.btnView['state'] = 'disable' #Vô hiệu nút View
        else:  #Trang 1, hiển thị 0
            listWF[current + 1].unhide() 
            listWF[current + 1].readImage(jpgImg)
            listWF[current + 1].displayImage()

        #in ra đang ở trang bao nhiêu 0 hoặc 1
        print("Step " + str(current) + " Extraction complete!")

    
    #Xóa bộ nhớ đã cấp phát cho 2 nút nhấn
    def removeComponent(self):
        self.btnClose.destroy()
        self.btnView.destroy()

    #Hàm con đọc ảnh
    def readImage(self, img):
        self.image = img

    #Hàm con hiện ảnh
    def displayImage(self):
        imgTk = self.image.resize((250, 250), Image.ANTIALIAS) #Chỉnh lại kích thước ảnh, sử dụng bộ lọc NTIALIAS
        imgTk = ImageTk.PhotoImage(image=imgTk) #mở ảnh sau khi xử lý
        self.image = imgTk #cho thuộc tính ảnh bằng ảnh imgTk
        self.labelImg = tkinter.Label(self.winFrame, image=self.image) #ẢNh hiện thị trong khung winFrame và ảnh hiển thị alf ảnh sau khi xử lý
        self.labelImg.place(x=700, y=150) #Chọn vị trí hiển thị ảnh
