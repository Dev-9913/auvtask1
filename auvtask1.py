import cv2
import numpy as np
import math

def calculate_angle(pt1, pt2, pt3):
    
    vector1 = [pt1[0] - pt2[0], pt1[1] - pt2[1]]
    vector2 = [pt3[0] - pt2[0], pt3[1] - pt2[1]]
    
    
    norm1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    norm2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    vector1 = [vector1[0] / norm1, vector1[1] / norm1]
    vector2 = [vector2[0] / norm2, vector2[1] / norm2]
    
   
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    
    angle = math.acos(dot_product)
    
    angle_degrees = math.degrees(angle)
    
    return angle_degrees

path = 'haarcascades/arrow1.jpg'  
image = cv2.imread(path)


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])


mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = mask1 | mask2
cv2.imshow('..',red_mask)
cv2.waitKey(0)


contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)


image_with_contours = image.copy()

if contours:
    for contour in contours:
       
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        print(approx)
        
      
        cv2.drawContours(image_with_contours, [approx], -1, (0, 255, 0), 3)


cv2.imshow("Red Arrow with Approximated Contours", image_with_contours)
cv2.waitKey(0)
if contours:
    for c in contours:
   
     M = cv2.moments(c)
    #  print(M)
     if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
     else:
        center_x, center_y = 0, 0  
    # center point
    cv2.circle(image_with_contours, (center_x, center_y), 5, (255, 0, 0), -1)  # blue 
    #  vertical line
    height, width = image.shape[:2]
    cv2.line(image_with_contours, (center_x, 0), (center_x, height), (0, 0, 0), 2)  # black
    cv2.imshow('Image with Line', image_with_contours)
    cv2.waitKey(0)
    for point in approx:
        x, y = point[0]
            # Plot the corner points (red circles)
        cv2.circle(image_with_contours, (x, y), 5, (0, 0, 255), -1)  # red
    max_distance = 0
    best_point = None
    min_angle = float('inf')    
    for i, point in enumerate(approx):
            x, y = point[0]

            #distance from the centroid
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            
            #adjacent points
            prev_point = approx[i - 1][0]
            next_point = approx[(i + 1) % len(approx)][0]
            
            
            angle = calculate_angle(prev_point, point[0], next_point)
            
            #dvector condn
            if distance > max_distance and angle < min_angle:
                max_distance = distance
                min_angle = angle
                best_point = (x, y)

    if best_point:
            #dvector
            cv2.line(image_with_contours, (center_x, center_y), best_point, (0, 255, 255), 2)  # yellow
            cv2.imshow('Final Image', image_with_contours)
            cv2.waitKey(0)
            cv2.destroyAllWindows()   
            #as vertical from centroid
            vertical_angle = 90.0
            #dv slope
            delta_x = best_point[0] - center_x
            delta_y = best_point[1] - center_y
            if delta_x == 0: 
             angle = 0.0
            slope = delta_y / delta_x
            #wrt horizontal x
            line_angle = np.degrees(np.arctan(slope))
            angle = vertical_angle - line_angle
            if angle < 0:
              angle += 180.0
            print(f"Angle with vertical line: {angle} degrees")
            cv2.putText(image_with_contours, f"angle:{angle+3}", (center_x-10,center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 ,0, 0), 1, cv2.LINE_AA)
            cv2.imshow('Final Image', image_with_contours)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

 