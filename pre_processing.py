import math
import numpy as np
import cv2
from scipy.ndimage import label

np.set_printoptions(threshold=np.nan)

letter_count=0
input_file_name = 'for_slides.jpg'
input_image = cv2.imread(input_file_name, 0)
image_copy = cv2.imread(input_file_name, 0)
input_image[input_image<128]=0#Black
input_image[input_image>=128]=255#White

[nrows, ncols]=input_image.shape

line_started=0;
lines_list=[];
for i in range(nrows):
    cont_in_row=0
    for j in range(ncols):
        if line_started==0 and input_image[i][j]==0:
            line_started=1
            line_top=i
            cont_in_row=1
        elif line_started==1 and input_image[i][j]==0:
            cont_in_row=1
    if cont_in_row==0 and line_started==1:
        line_bottom=i-1
        lines_list.append([line_top, line_bottom])
        line_started=0

for line in lines_list:
    print("line")
    input_image_seg=input_image[line[0]:line[1]+1,:]
    [nrows, ncols]=input_image_seg.shape
    letter_started=0;
    letters_list=[];
    for i in range(ncols):
        cont_in_col=0
        for j in range(nrows):
            if letter_started==0 and input_image_seg[j][i]==0:
                letter_started=1
                letter_left=i
                letter_top=line[0]+j
                letter_bottom=line[0]+j
                cont_in_col=1
            elif letter_started==1 and input_image_seg[j][i]==0:
                letter_top=min(letter_top, line[0]+j)
                letter_bottom=max(letter_bottom, line[0]+j)
                cont_in_col=1
        if cont_in_col==0 and letter_started==1:
            letter_right=i-1
            image_copy= cv2.rectangle(image_copy, (letter_left-2,letter_top-2), (letter_right+2,letter_bottom+2), (0,0,255), 1)
            letters_list.append([letter_left,letter_top,letter_right,letter_bottom])
            letter_started=0
            
    max_width=0
    for i in range(len(letters_list)):
        width_current=letters_list[i][2]-letters_list[i][0]
        max_width=max(width_current,max_width)
        
               
    for i in range(len(letters_list)):
        letter=letters_list[i]
        if i!=0:
            prev_letter=letters_list[i-1]
            space_between=letter[0]-prev_letter[2]
            if space_between>0.5*max_width:
                print("space")
        width=letter[2]-letter[0]+4
        height=letter[3]-letter[1]+10
        diff=math.floor(abs(width-height)/2)
        letter_image=input_image[letter[1]-5:letter[3]+5,letter[0]-2:letter[2]+2]
        if height>width:
            padding=255*np.ones((height,diff))            
            letter_image=np.concatenate((padding,letter_image,padding),axis=1)
        elif height<width:
            padding=255*np.ones((diff,width))
            letter_image=np.concatenate((padding,letter_image,padding),axis=0)
        if (input_image[letter[1]:letter[3],letter[0]:letter[2]].max()==0) and abs(letter[2]+letter[1]-letter[3]-letter[0])<5:
            print("period")
        else:   
            print("letter")
        letter_count+=1
        name="letter"+str(letter_count)+".jpg"
        cv2.imwrite( name, letter_image);
        cv2.imshow('image',letter_image);
        cv2.waitKey(0)
cv2.imwrite( "full_image.jpg", image_copy);
cv2.imshow('image_copy',image_copy);
cv2.waitKey(0)
