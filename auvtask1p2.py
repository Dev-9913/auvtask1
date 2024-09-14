import cv2
import numpy as np
import math

# Load image
path = 'haarcascades/task2arrow.jpg'  
image = cv2.imread(path)

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV ranges for red color
lower_red1 = np.array([0, 42, 191])
upper_red1 = np.array([37, 255, 255])

# Create masks for red color
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

red_mask = mask1 

# Find contours
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)

        # Step 1: Find and plot all corner points
        for point in approx:
            x, y = point[0]
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # red circles for corner points

        # Step 2: Draw convex hull and plot the boundary points
        hull = cv2.convexHull(contour)
        for point in hull:
            x, y = point[0]
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # blue circles for convex hull points
        cv2.drawContours(image, [hull], -1, (255, 255, 0), 2)  # convex hull boundary
        
        # Find the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = 0, 0
        
        # Plot the centroid
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)  # blue circle for centroid
        
        # Draw the vertical line through the centroid
        height, width = image.shape[:2]
        cv2.line(image, (center_x, 0), (center_x, height), (0, 0, 0), 2)  # black vertical line
        
        cv2.imshow("Convex Hull and Centroid", image)
        cv2.waitKey(0)

        # Step 3: Find common points between corner points and convex hull
        common_points = []
        for point in approx:
            for h_point in hull:
                if np.array_equal(point[0], h_point[0]):
                    common_points.append(point[0])

        # Remove old points and plot the common points
        image = cv2.imread(path)  # reload the image for a clean background
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)  # redraw contour
        
        # Redraw the centroid and vertical line
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)  # blue centroid
        cv2.line(image, (center_x, 0), (center_x, height), (0, 0, 0), 2)  # vertical line
        
        for point in common_points:
            cv2.circle(image, tuple(point), 5, (0, 255, 255), -1)  # yellow for common points
        cv2.imshow("Common Boundary Points", image)
        cv2.waitKey(0)

        # Step 4: Find the point closest to the centroid from the common points
        min_distance = float('inf')
        best_point = None

        for point in common_points:
            x, y = point
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                best_point = (x, y)

        if best_point:
            cv2.circle(image, best_point, 5, (255, 255, 0), -1)  # plot closest point in cyan
            cv2.line(image, (center_x, center_y), best_point, (0, 255, 255), 2)  # draw direction vector
            cv2.imshow("Final Image with Closest Point", image)
            cv2.waitKey(0)
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
            if line_angle < 0:
              line_angle += 180.0
            angle = line_angle-vertical_angle
            
            print(f"Angle with vertical line: {angle} degrees")
            cv2.putText(image, f"angle:{angle}", (center_x-10,center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 ,0, 0), 1, cv2.LINE_AA)
            cv2.imshow('Final Image', image)
            cv2.waitKey(0)
            

cv2.destroyAllWindows()
