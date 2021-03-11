from Tkinter import *
from tkFileDialog   import askopenfilename
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tkMessageBox
import math

#Function which does nothing
def onMouse(x):
    pass
#Common instruction for each features
def warning():
    tkMessageBox.showinfo("Instruction","Press ESC to exit\nPress S to save")

#Breif description of the features
def About():
    tkMessageBox.showinfo("About Functions","Black/White: Image becomes a binary image(0,1)\n"\
                          "Blend: Blends two images\n"\
                          "Blur: Blurs images with different methods\n"
                          "Brightness: Adjusts the brightness of the image\n"\
                          "Contrast: literally contrast the image\n"\
                          "Gray: Change the image to gray scale\n"\
                          "Logo: Add a logo\n"\
                          "Mask: Only certain colours are displayed in the image(red,green,blue)\n"\
                          "RGB: Image is divided into RGB components\n"\
                          "Rotate: Rotate the image\n"\
                          "Text: Add text on the image\n")
'''
#Shrink Image
def shrink():
    img_name = askopenfilename()
    while(img_name==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name = askopenfilename()
    size = 3
    img = cv2.imread(img_name)
    warning()
    rows,cols = img.shape[:2]
    new = np.zeros((rows/size,cols/size,3), np.uint8)
    prevR = 0
    afterR = 0
    while(prevR<rows-size):
        prevC = 0
        afterC = 0
        while(prevC<cols-size):
            prevX,prevY = img[prevC,prevR]
            img[afterC,afterR] = [prevX,prevY]
            prevC += size
            afterC += 1
        prevR = size
        afterR = 1
    cv2.imshow('Shrink image',img)
'''

#Function that contrasts the image
def contrast():
    img_name = askopenfilename()
    while(img_name==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name = askopenfilename()
    img = cv2.imread(img_name)
    warning()
    
    rows,cols = img.shape[:2]
    for i in range(rows):
        for j in range(cols):
            b,g,r = img[i,j]
            img[i,j] = 255-b,255-g,255-r
    cv2.namedWindow('Contrast', cv2.WINDOW_NORMAL)          
    cv2.imshow('Contrast',img)
    
    #ESC and Save features
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('images/Contrast.jpg',img)
        cv2.destroyAllWindows()
        
#Function that blurs the image
def blur():
    img_name = askopenfilename()
    while(img_name==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name = askopenfilename()
    img = cv2.imread(img_name)
    warning()
    
    #create Track Bar
    cv2.namedWindow('Blur Image',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Blur Mode','Blur Image',0,2,onMouse)
    cv2.createTrackbar('Blur','Blur Image',0,5,onMouse)

    mode = cv2.getTrackbarPos('Blur Mode','Blur Image')
    val = cv2.getTrackbarPos('Blur','Blur Image')

    while(1):
        val = val*2+1
        try:
            if mode == 0:
                #(val,val) is the kernal size
                blur = cv2.blur(img,(val,val))
            elif mode ==1:
                #Use Gaussian blur filter
                blur = cv2.GaussianBlur(img,(val,val),0)
            elif mode ==2:
                #Take all pixel of the given kernal and get median value into the middle pix
                blur = cv2.medianBlur(img,val)
            else:
                break
            #cv2.namedWindow('Blur Image', cv2.WINDOW_NORMAL)
            cv2.imshow('Blur Image',blur)
        except:
            break
        
        #ESC and Save features
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break
        elif k == ord('s'):
            cv2.imwrite('images/Blur.jpg',img)                        
        mode = cv2.getTrackbarPos('Blur Mode','Blur Image')
        val = cv2.getTrackbarPos('Blur','Blur Image')
        
    cv2.destroyAllWindows()

#Function to change the image into binary(black&white)
def blackWhite():
    img_name = askopenfilename()
    while(img_name==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name = askopenfilename()
    img = cv2.imread(img_name)
    warning()
    rows,cols = img.shape[:2]
    black_pix = [0,0,0]
    white_pix = [255,255,255]
    for i in range(rows):
        for j in range(cols):
            b,g,r = img[i,j]   
            avg_rgb = (int(r)+int(g)+int(b))/3
            if avg_rgb < 128:
                img[i,j] = black_pix
            else:
                img[i,j] = white_pix
    cv2.namedWindow('Black and White', cv2.WINDOW_NORMAL)
    cv2.imshow('Black and White',img)
    
    #ESC and Save features
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('images/blackWhite.jpg',img)
        cv2.destroyAllWindows()
        
'''
#Own blurring function but didn't go well
def own_blur():
    np.seterr(over='ignore')
    img_name = askopenfilename()
    img = cv2.imread(img_name)
    rows,cols = img.shape[:2]
    for i in range(0,rows,2):
        for j in range(0,cols,2):
            b,g,r = img[i,j]
            b1,g1,r1 = img[i+1,j]
            b2,g2,r2 = img[i,j+1]
            b3,g3,r3 = img[i+1,j+1]
            med_b = np.array([b,b1,b2,b3])
            med_g = np.array([g,g1,g2,g3])
            med_r = np.array([r,r1,r2,r3])
            nB = np.median(med_b)
            nG = np.median(med_g)
            nR = np.median(med_r)
            img[i,j] = nB,nG,nR
            img[i+1,j] = nB,nG,nR
            img[i,j+1] = nB,nG,nR
            img[i+1,j+1] = nB,nG,nR
    cv2.imshow('Blurred Image',img)
'''

#Function that outputs a gray-scaled image
def toGray():
    img_name = askopenfilename()
    while(img_name==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name = askopenfilename()
    img = cv2.imread(img_name)
    warning()
    
    rows,cols = img.shape[:2]
    for i in range(rows):
        for j in range(cols):
            b,g,r = img[i,j]
            r=int(r*0.299)
            g=int(g*0.587)
            b=int(b*0.114)
            y = r+g+b
            img[i,j] = [y,y,y]
    cv2.namedWindow('Gray Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Gray Image',img)
    #ESC and Save features
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('images/Rotate.jpg',img)
        cv2.destroyAllWindows()

#Function that outputs rotated images
def imgRotate():
    img_name = askopenfilename()
    while(img_name==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name = askopenfilename()
    img = cv2.imread(img_name)
    rows,cols = img.shape[:2]
    print "Rotate Image Feature:\n"
    degree =  input("Input the degree: ")
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    warning()
    img2 = cv2.warpAffine(img,M,(cols,rows))
    cv2.namedWindow('Rotated Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Rotated Image',img2)

    #ESC and Save features
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('images/Rotate.jpg',img2)
        cv2.destroyAllWindows()
        
#Function that outpus RGB images
def bgr():
    img_name = askopenfilename()
    while(img_name==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name = askopenfilename()
    img = cv2.imread(img_name)
    b,g,r = cv2.split(img)
    cv2.imshow('blue channel',b)
    cv2.imshow('green channel',g)
    cv2.imshow('red channel',r)
    warning()
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('images/R.jpg',r)
        cv2.imwrite('images/G.jpg',g)
        cv2.imwrite('images/B.jpg',b)
        cv2.destroyAllWindows()

#Function that adds texts on the image
def text():
    # Create the event handler first
    def inputFunc():
        global text
        text = eHello.get()
        F.quit()
        
    # Create the top level window/frame
    top = Tk()
    top.wm_title("Add Text")
    F = Frame(top)
    F.pack(expand="true")
    
    # Now the frame with text entry
    fEntry = Frame(F, border="1")
    eHello = Entry(fEntry)
    fEntry.pack(side="top", expand="true")
    eHello.pack(side="left", expand="true")

    # The frame with the buttons.
    fButtons = Frame(F, relief="sunken", border=1)
    bClear = Button(fButtons, text="OK", command=inputFunc)
    bClear.pack(side="left", padx=5, pady=2)
    fButtons.pack(side="top", expand="true")
    
    # Now run the eventloop
    F.mainloop()
    img_name = askopenfilename()
    img = cv2.imread(img_name)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print "\nText Feature:\n"
    pix_x = input("Input x-coordinate location for text area: ")
    pix_y = input("Input y-coordinate location for text area: ")
    size_font = input("Input size of the font: ")
    r,g,b = input("input R,G,B ex)255,255,255: ")
    cv2.putText(img, text, (pix_x,pix_y), font, size_font, (b,g,r), 2)
    cv2.namedWindow('Text', cv2.WINDOW_NORMAL)
    cv2.imshow('Text',img)

    #ESC and Save features
    warning()
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('images/Text.jpg',img)
        cv2.destroyAllWindows()

#Function that adds a logo on the image
def logo():
    img_name1 = askopenfilename()
    #Until user inputs an image
    while(img_name1==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name1 = askopenfilename()
    img1 = cv2.imread(img_name1)
    warning()
    
    r1,h1 = img1.shape[:2]
    tkMessageBox.showinfo("The Main Image",img_name1+" has been chosen")
    img_name2 = askopenfilename()
    
    #If second image is bigger than first one, choose again
    while(1):
        if(img_name2!=""):
            img2 = cv2.imread(img_name2)
            r2,h2 = img2.shape[:2]
            if((r1+h1) < (r2+h2)):
                tkMessageBox.showinfo("WARNING","Please choose an image that is smaller than the main image")
                img_name2 = askopenfilename()
            else:
                break
        else:
            tkMessageBox.showinfo("WARNING","Please choose an image")
            img_name2 = askopenfilename()
    img2 = cv2.imread(img_name2)
    tkMessageBox.showinfo("Image of Logo",img_name2+" has been chosen")
    
    #Values for the location of the logo
    print "\nLogo Feature:\n"
    hpos = input("Input x-coordinate value to add logo: ")
    vpos = input("Input y-coordinate value to add logo: ")
    rows,cols,channels = img2.shape
    roi = img1[vpos:rows+vpos, hpos:cols+hpos]
    
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #Using binary img, convert logo in to black
    ret,mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    #Where mask is not 0, AND operate for two given imgs
    img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2,img2,mask=mask)
    #Add two different images 
    dst = cv2.add(img1_bg,img2_fg)
    img1[vpos:rows+vpos, hpos:cols+hpos] = dst
    cv2.namedWindow('Logo', cv2.WINDOW_NORMAL)
    cv2.imshow('Logo',img1)
    
    #ESC and Save features
    warning()
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('images/Logo.jpg',img)
        cv2.destroyAllWindows()
    
#Converts HSV to RGB using given formula
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    
    #Automatically caps Lum.
    if(v>1):
        v = 1
    h_60 = h / 60.0
    f_h_60 = math.floor(h_60)
    h_int = int(f_h_60) % 6
    f = h_60 - f_h_60
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if h_int == 0: r, g, b = v, t, p
    elif h_int == 1: r, g, b = q, v, p
    elif h_int == 2: r, g, b = p, v, t
    elif h_int == 3: r, g, b = p, q, v
    elif h_int == 4: r, g, b = t, p, v
    elif h_int == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

#Converts RGB to HSV using given formula
def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val-min_val
    if max_val == min_val:
        h = 0
    elif max_val == r:
        h = (60 * ((g-b)/diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b-r)/diff) + 120) % 360
    elif max_val == b:
        h = (60 * ((r-g)/diff) + 240) % 360
    if max_val == 0:
        s = 0
    else:
        s = diff/max_val
    v = max_val
    return h, s, v

#Function that adjust brightness of the image
def brightness():
    img_name = askopenfilename()
    while(img_name==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name = askopenfilename()
    tkMessageBox.showinfo("The Main Image",img_name+" has been chosen")
    img = cv2.imread(img_name)
    height,width = img.shape[:2]
    v_in = input("Input brightness value(0 - 2): ")
    for i in range(height):
        for j in range(width):
            b,g,r = img[i,j]
            h,s,v = rgb2hsv(r,g,b)
            r,g,b = hsv2rgb(h,s,(v*v_in))
            img[i,j]=b,g,r
    cv2.namedWindow('Brightness', cv2.WINDOW_NORMAL)
    cv2.imshow('Brightness',img)
    
    #ESC and Save features
    warning()
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('images/Brightness.jpg',img)
        cv2.destroyAllWindows()

        
#Function that looks for specific colour using mask
def color_range():
    img_name = askopenfilename()
    while(img_name==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name = askopenfilename()
    tkMessageBox.showinfo("The Main Image",img_name+" has been chosen")
    img = cv2.imread(img_name)
    height = img.shape[0]
    width = img.shape[1]
    
    #Range of the defined color
    lower_blue = np.array([215,0.2,0.5])
    #lower_blue = np.array([120/239*360,119/240,152/240])
    upper_blue = np.array([250,1,1])
    #upper_blue = np.array([150/239*360,151/240,99/240])
    lower_green = np.array([60,0.4,0.4])
    upper_green = np.array([180,255,255])
    lower_red = np.array([245,0.4,0.4])
    upper_red = np.array([60,1,1])
    
    '''
    #Create the event handler first
    def inputFunc():
        global inColor
        if(eHello.get() == "blue"):
            inColor = "blue"
        elif(eHello.get() == "red"):
            inColor = "red"
        elif(eHello.get() == "green"):
            inColor = "green"
        F.quit()
    tkMessageBox.showinfo("WARNING","Only red/green/blue available at this moment")
    # create the top level window/frame
    top = Tk()
    F = Frame(top)
    F.pack(expand="true")
    
    # Now the frame with text entry
    fEntry = Frame(F, border="1")
    eHello = Entry(fEntry)
    fEntry.pack(side="top", expand="true")
    eHello.pack(side="left", expand="true")

    # The frame with the buttons.
    fButtons = Frame(F, relief="sunken", border=1)
    bClear = Button(fButtons, text="OK", command=inputFunc)
    bClear.pack(side="left", padx=5, pady=2)
    fButtons.pack(side="top", expand="true")
    '''
    # Now run the eventloop
    #F.mainloop()
        
    #Changes RGB to HSV then tries to find out the selected colour in a disignated range of colours
    #if(inColor == "blue"):
    if(1):
        flag1 = 0
        flag2 = 0
        flag3 = 0
        flag4 = 0
        print "Converting..."
        for i in range(height):
#            print "Loading...",float(i)/float(height),"%"
            load = round(float(i)/float(height)*100)
            if((load==25 and flag1==0) or (load==50 and flag2==0) or (load==75 and flag3==0) or (load ==99 and flag4==0)):
                if(load==25):
                    flag1 = 1
                if(load==50):
                    flag2 = 1
                if(load==75):
                    flag3 = 1
                if(load==99):
                    flag4 = 1
                print "Loading...",load,"%"
            for j in range(width):
                b,g,r = img[i,j]
                #print "b,g,r = ",b,g,r
                h,s,v = rgb2hsv(r,g,b)
                if(h<lower_blue[0] or s<lower_blue[1] or v<lower_blue[2] or h>upper_blue[0] or s>upper_blue[1] or v>upper_blue[2]):
                    continue                    
                else:
                    img[i,j] = [0,0,0]
        print "Completed!"
    elif(inColor == "red"):
        for i in range(height):
            for j in range(width):
                b,g,r = img[i,j]
                #print "b,g,r = ",b,g,r
                h,s,v = rgb2hsv(r,g,b)
                if(upper_red[0]<h<lower_red[0] or s<lower_red[1] or v<lower_red[2] or s>upper_red[1] or v>upper_red[2]):
                   img[i,j] = [0,0,0]
    else:
        for i in range(height):
            for j in range(width):
                b,g,r = img[i,j]
                #print "b,g,r = ",b,g,r
                h,s,v = rgb2hsv(r,g,b)
                if(h<lower_green[0] or s<lower_green[1] or v<lower_green[2] or h>upper_green[0] or s>upper_green[1] or v>upper_green[2]):
                   img[i,j] = [0,0,0]
    cv2.namedWindow('Color Range', cv2.WINDOW_NORMAL)
    cv2.imshow('Color Range',img)
    
    #ESC and Save features
    warning()
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('images/Mask.jpg',img)
        cv2.destroyAllWindows()

def blending():
    img_name1 = askopenfilename()
    while(img_name1==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name1 = askopenfilename()
    tkMessageBox.showinfo("First image to blend",img_name1+" has been chosen")
    img_name2 = askopenfilename()
    while(img_name2==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name2 = askopenfilename()
    tkMessageBox.showinfo("Second image to blend",img_name2+" has been chosen")
    
    #Load same size images
    img1 = cv2.imread(img_name1)
    img2 = cv2.imread(img_name2)


    #Create Track Bar range from 0 to 100
    cv2.namedWindow('Blended Image', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Mix Rate', 'Blended Image', 0, 100, onMouse)
    mix = cv2.getTrackbarPos('Mix Rate','Blended Image')
    warning()
    while True:
        img = cv2.addWeighted(img1, float(100-mix)/100,img2,float(mix)/100,0)
        cv2.imshow('Blended Image',img)
    
        #ESC and Save features
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('images/Blended.jpg',img)
            cv2.destroyAllWindows()
            break
        
        mix = cv2.getTrackbarPos('Mix Rate','Blended Image')

    cv2.destroyAllWindows()
    
'''
#For testing purposes of my own blending code (50:50 ratio)
def BlendImg():
    img_name1 = askopenfilename()
    while(img_name1==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name1 = askopenfilename()
    tkMessageBox.showinfo("First image to blend",img_name1+" has been chosen")
    img_name2 = askopenfilename()
    while(img_name2==""):
        tkMessageBox.showinfo("WARNING","Please choose an image")
        img_name2 = askopenfilename()
    tkMessageBox.showinfo("Second image to blend",img_name2+" has been chosen")
    #Load same size images
    img1 = cv2.imread(img_name1)
    img2 = cv2.imread(img_name2)
    
    height,width = img1.shape[:2]
    
    for i in range(height):
        for j in range(0,width,2):
            if(i%2==0):
                img1[i,j] = img2[i,j+1]
    cv2.imshow('Test Blend',img1)
'''

#############################
#TOP Level                  #
#Tkinter GUI implementd here#
#############################
if __name__ == '__main__':
    #Running Tkinter application 
    root = Tk()
    root.wm_title("Photo Editor")
    menu = Menu(root)
    root.config(menu=menu)
    
    #Not gonna use these options anymore
    """
    filemenu = Menu(menu)
    menu.add_cascade(label="File", menu=filemenu)
    filemenu.add_command(label="Open", command=OpenFile)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=root.quit)
    filemenu.add_command(label="", command=OpenFile)
    """
    helpmenu = Menu(menu)
    menu.add_cascade(label="Help", menu=helpmenu)
    helpmenu.add_command(label="About", command=About)

    F = Frame(root)
    F.pack()
    
    #Commands for buttons
    blend = Button(F,text="Blend",command=blending)
    rgb_img = Button(F,text="RGB",command=bgr)
    add_img = Button(F,text="Logo",command=logo)
    put_text = Button(F, text ="Text", command=text)
    color_range = Button(F, text="Mask", command=color_range)
    rotate_img = Button(F,text="Rotate",command=imgRotate)
    blur_img = Button(F,text="Blur",command=blur)
    black_white = Button(F,text="Black/White",command=blackWhite)
    bright = Button(F,text="Brightness",command=brightness)
    gray = Button(F,text="Gray",command=toGray)
    contrast = Button(F,text="Contrast",command=contrast)
    #shrink_img = Button(F,text="Shrink",command=shrink)
    
    #Display buttons
    black_white.pack(padx=90,pady=2)
    blend.pack( pady=2)
    blur_img.pack(pady=2)
    bright.pack(pady=2)
    contrast.pack(pady=2)
    gray.pack(pady=2)
    add_img.pack(pady=2)
    color_range.pack( pady=2)
    rgb_img.pack( pady=2)
    rotate_img.pack( pady=2)
    #shrink_img.pack(pady=2)
    put_text.pack( pady=2)

    mainloop()

