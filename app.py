import cv2
import numpy as np

#works only for images because this was a test


def make_line_points(y1,y2,line):
    slope,intercept=line
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])



def average_slope_intercept(lines):
    left_lines=[]
    right_lines=[]

    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2-x1==0:
                continue
            slope=(y2-y1)/(x2-x1)
            intercept=y1-slope*x1
            if slope<-0.5:
                left_lines.append((slope,intercept))
            elif slope>0.5:
                right_lines.append((slope,intercept))
    left_avg=np.mean(left_lines,axis=0) if left_lines else None
    right_avg=np.mean(right_lines,axis=0) if right_lines else None

    y1=height
    y2=int(height*0.6)

    left_line=make_line_points(y1,y2,left_avg) if left_avg is not None else None
    right_line=make_line_points(y1,y2,right_avg) if right_avg is not None else None

    return left_line, right_line










#change image here: 
image=cv2.imread("road1.jpg")

if image is None:
    print("image not loaded")
    exit()

else:
    print("image is loaded")
    print(image.shape)
    '''cv2.imshow("road image", image)'''

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    print(gray.shape)

    blur=cv2.GaussianBlur(gray,(5,5),0)

    edges=cv2.Canny(blur,50,150)


    
    mask=np.zeros_like(edges)
    height,width=edges.shape

    polygon=np.array([[
        (int(width*0.05),height),
        (int(width*0.95),height),
        (int(width*0.6), int(height*0.6)),
        (int(width*0.4),int(height*0.6))
    ]],np.int32)

    cv2.fillPoly(mask,polygon,255)

    masked_edges=cv2.bitwise_and(edges,mask)



    lines=cv2.HoughLinesP(masked_edges,1,np.pi/180,threshold=50,minLineLength=40,maxLineGap=100)
    
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)

    cv2.imshow("road image", image)




    line_image=np.zeros_like(image)

    left_line,right_line=average_slope_intercept(lines)

    if left_line is not None:
        cv2.line(line_image,(left_line[0],left_line[1]),(left_line[2],left_line[3]),(0,0,255),10)
    if right_line is not None:
        cv2.line(line_image, (right_line[0],right_line[1]),(right_line[2],right_line[3]),(255,0,0),10)
    
    combo=cv2.addWeighted(image,0.8,line_image,1,1)

    cv2.imshow("lane detection", combo)




    cv2.waitKey(0)
    cv2.destroyAllWindows()






