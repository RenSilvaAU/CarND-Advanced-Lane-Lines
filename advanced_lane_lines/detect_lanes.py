import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2

''' 
==============================================================================================================================
LaneDetector

Class used to detect lanes on a road with a forward facing camera

==============================================================================================================================
'''
class LaneDetector:


    def __init__(self,mtx=None,dist=None,hls_thresh=(80,255),colour_thresh=(150,255)):
        
        self.mtx = mtx
        self.dist = dist
        self.hls_thresh = hls_thresh
        self.colour_thresh = colour_thresh

        self.gray_shape = None
        
        self.colors = {'red':0,'green':1,'blue':2}
           
    ''' 
    ==========================================================================================================================
    calibrate_camera 
    ==========================================================================================================================

    Arguments:
    
        - train_files: list of objects to be used for training
        - test_files (optional): list of objects to demonstrate that the function calculates the correct coefficients
        - shape: number of corners expected to be found on each chessboard (row x col)
        - verbose (optional): if set ot non-zero, this function will display debug messages

    ==========================================================================================================================
    This method starts by preparing object points, which will be the (x, y, z) coordinates of the chessboard corners in the 
    world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for 
    each calibration image.

    Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I 
    successfully detect all chessboard corners in a test image. imgpoints will be appended with the (x, y) pixel position of each 
    of the corners in the image plane with each successful chessboard detection.
    ==========================================================================================================================
    '''
    def calibrate_camera(self,train_files,test_files=None,shape=(9,6),verbose=0):

        # initialise lists
        objpoints = []
        imgpoints = []

        # standard chessboard for a shape = shape
        objp = np.zeros((shape[0] * shape[1] ,3),np.float32)
        objp[:,:2] = np.mgrid[0:shape[0] ,0:shape[1] ].T.reshape(-1,2)

        # select a consistent shape.. and ensure all images have that
        self.gray_shape = None

        if verbose:
            print('\nCalibrating camera...')

            print('\nTraining...')

        for i,file in enumerate(train_files):

            img = mping.imread(file)

            # will use the shape of the first image, and make sure all ensure all shapes are the same
            if self.gray_shape is None:
                self.gray_shape = img.shape[1::-1]

            elif self.gray_shape != img.shape[1::-1]:
                img = cv2.resize(img,self.gray_shape)


            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            found, corners = cv2.findChessboardCorners(gray,shape,None)

            if found:
                imgpoints.append(corners)
                objpoints.append(objp)

                if verbose:
                    img = cv2.drawChessboardCorners(img,shape,corners,found)
                    plt.figure(figsize=(8,5))
                    plt.imshow(img)
                    plt.show()


        #now find the distortion coefficients
        if len(objpoints) == 0:
            return gray_shape, None, None

        # now calculate the coefficients
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.gray_shape, None, None)
        
        if len(test_files) > 0:
            if verbose:
                print('\nTesting...')
            
            self.test_undistort(test_files)
   
        if verbose:
            print('\nFinished Calculating calibration coefficients')
            
    '''
    ==========================================================================================================================
    test_undistort
    ==========================================================================================================================

    Arguments:
    
        - file_names: list of files that are to be undistorted for test

    ==========================================================================================================================
    This method navigates through a list of files, and applies the undistort method to each of them, displaying the original 
    and the undistorted image, side by side
    ==========================================================================================================================
    '''
    
    def test_undistort(self,file_names):

            
        # if test_images provided, display them 
        for file_name in file_names:
            
            img = mping.imread(file_name)

            dst = self.undistort(img)
            

            plt.figure(figsize=(10,6))

            plt.subplot(2,2,1)
            plt.imshow(img)

            plt.subplot(2,2,2)
            plt.imshow(dst)

            plt.show()            

    
    '''
    ==========================================================================================================================  
    Undistort
    ==========================================================================================================================

    Arguments:
    
        - img: image to be undistorted

    ==========================================================================================================================
    This method undistorts an image, based on the mtx and dist factors, created during calibration.
    
    NB -> the camera calibration must have been run before this method is effective
    ==========================================================================================================================
    '''
    
    def undistort(self,img):

        if self.gray_shape != img.shape[1::-1]:
            img = cv2.resize(img,self.gray_shape)

        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            
            

    '''
    ==========================================================================================================================
    hls_select
    ==========================================================================================================================

    Arguments:
    
        - img: image to be processed
        - thresh: threshold of values to be selected as white

    ==========================================================================================================================
    this method isolates the s channel
    and applies a binary (black or white), return white for those pixels that fall within a selected threshold
    ==========================================================================================================================

    '''
    def hls_select(self,img, thresh=None):
        
        if thresh == None:
            thresh = self.hls_thresh
        
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output

    '''
    ==========================================================================================================================
    colour_select
    ==========================================================================================================================

    Arguments:
    
        - img: image to be processed
        - color (optional): color to be selected: 'red','green', or 'blue'
        - thresh (optional): threshold of values to be selected as white

    ==========================================================================================================================
    this method isolates the s channel
    and applies a binary (black or white), return white for those pixels that fall within a selected threshold
    ==========================================================================================================================

    '''
    def colour_select(self,img, color='green', thresh=None):

        if thresh == None:
            thresh = self.colour_thresh
        
        sel_channel = img[:,:,self.colors[color]]
        binary_output = np.zeros_like(sel_channel)
        binary_output[(sel_channel > thresh[0]) & (sel_channel <= thresh[1])] = 1
        return binary_output
    
    '''
    ==========================================================================================================================
    abs_sobel_thresh
    ==========================================================================================================================

    Arguments:
    
        - img: image to be processed
        - orient (defaults to 'x'): orientation
        - thresh_min: minimum of values to be selected as white
        - thresh_max: maximum of values to be selected as white

    ==========================================================================================================================
    Applies sobel filter and returns a binary based on it
    ==========================================================================================================================

    '''    
    def abs_sobel_thresh(self, img, orient='x', thresh_min=0, thresh_max=255):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Return the result
        return binary_output
    
    
    
    
    '''
    ==========================================================================================================================
    mag_thresh
    ==========================================================================================================================

    Arguments:
    
        - img: image to be processed
        - sobel_kernel: (defaults to 3)
        - max_thresh: mag threshold

    ==========================================================================================================================
    Applies mag threshold
    ==========================================================================================================================
    '''
    def mag_thresh(self,img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    '''
    ==========================================================================================================================
    lane_binary
    =========================================================================================================================
    Arguments:
    
        - img: image to be processed

    ==========================================================================================================================
    Apply several a combinatio of colour and gradient transforms to detect the best lane binary
    ==========================================================================================================================

    '''  
    def lane_binary(self,img):
    
        # now do gradient a transform to get a binary
        hls_binary = self.hls_select(img)

        # now do colour gradient a transform to get a binary
        colour_binary = self.colour_select(img)

        sob_binary = self.abs_sobel_thresh(img, orient='x', thresh_min=30, thresh_max=100)

        # combining binaries ... this will be the content of method lane_binary()
        combined = np.zeros_like(hls_binary)
        combined[((hls_binary == 1) & (colour_binary == 1) | (sob_binary == 1))] = 1

        return combined
    
    
    '''
    ==========================================================================================================================
    warper
    ==========================================================================================================================

    Arguments:
    
        - img: image to be processed
        - src: four source points in the image to be preocessed (to be warped - normally a trapesoid)
        - dst: four source points in the image after preocessed (after warped - normally a rectangle)

    ==========================================================================================================================
    this function warps the image, to change the perspective
    ==========================================================================================================================

    '''
    def warper(self,img, src=None, dst=None):
        
        if src == None:
            src = self.src
            
        if dst == None:
            dst = self.dst

        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped

    '''
    ==========================================================================================================================
    set_src_dst
    ==========================================================================================================================

    Arguments:
    
        - img_size: the image size as reference

    ==========================================================================================================================
    adjusts hardcoded src and dst to the picture size 
    ==========================================================================================================================

    '''
    def set_src_dst(self,img_size):

        self.src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
            [((img_size[0] / 6) - 10), img_size[1]],
            [(img_size[0] * 5 / 6) + 60, img_size[1]],
            [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
        self.dst = np.float32(
            [[(img_size[0] / 4), 0],
            [(img_size[0] / 4), img_size[1]],
            [(img_size[0] * 3 / 4), img_size[1]],
            [(img_size[0] * 3 / 4), 0]])

        return self.src, self.dst

    '''
    ==========================================================================================================================
    find_lane_pixels
    ==========================================================================================================================

    Arguments:
    
        - binary_warped: an image that has been warped and binarised

    ==========================================================================================================================
    Applies histogram to find two lanes in an warped, binarised image
    ==========================================================================================================================

    '''

    def find_lane_pixels(self,binary_warped):
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, midpoint, out_img

    '''
    ==========================================================================================================================
    convert_to_metres
    ==========================================================================================================================

    Arguments:
    
        - left: x of left lane at the bottom of the picture
        - right:  x of right lane at the bottom of the picture
        - midpoint: midpoint of the bottom of the picture
        
    ==========================================================================================================================
    returns how many pictures off centre the middle of the lanes is (in pixels)
    ==========================================================================================================================

    '''
    def get_offcentre(self,left,right,midpoint):
        
        mid_lanes = (left + right) / 2
        
        if mid_lanes <= midpoint:
            return midpoint - mid_lanes , 'left'
        else:
            return mid_lanes - midpoint, 'right'
    

    '''
    ==========================================================================================================================
    convert_to_metres
    ==========================================================================================================================

    Arguments:
    
        - leftx: list of x values for left lane
        - rightx:  list of x values for right lane
        - ploty: list of y values for both left and right lane
        - moidpoint: midpoint at the bottom of the picture
        
    ==========================================================================================================================
    Converts xs, ys, and midpoint to metres (given in pixels)
    ==========================================================================================================================

    '''

    def convert_to_metres(self,leftx,rightx,ploty,midpoint):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension


        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Fit a second order polynomial to pixel positions in each  lane line
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        
        
        offcentre, position = self.get_offcentre(leftx[-1],rightx[-1],midpoint)
        
        offcentre_cr = offcentre*xm_per_pix

        return left_fit_cr, right_fit_cr, offcentre_cr, position

    '''
    ==========================================================================================================================
    fit_polynomial
    ==========================================================================================================================

    Arguments:
    
        - binary_warped: an image that has been warped and binarised
        
    ==========================================================================================================================
    Finds two lanes (left and white) on an image that has been warped, and binarised (identifying potential lanes), and fits
    to polynomials to it, containing two lanes
    ==========================================================================================================================

    '''
    def fit_polynomial(self,binary_warped,plot=False):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, midpoint, out_img = self.find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        if plot:

            # Plots the left and right polynomials on the lane lines
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')

        polig_x = np.concatenate((left_fitx[ploty.shape[0]//2:],right_fitx[ploty.shape[0]//2:][::-1]))
        polig_y = np.concatenate((ploty[ploty.shape[0]//2:],ploty[ploty.shape[0]//2:][::-1]))


        left_fit_cr, right_fit_cr, offcentre_cr, position = self.convert_to_metres(left_fitx, right_fitx, ploty, midpoint)

        return out_img, left_fit_cr, right_fit_cr, offcentre_cr, position, ploty, polig_x, polig_y



    '''
    ==========================================================================================================================
    measure_curvature_real
    ==========================================================================================================================

    Arguments:
    
        - left_fit_cr: xs of left polynomial, representing left lane
        - right_fit_cr: xs of rigth polynomial, representing right lane
        - ploty: ys of both polynomials (0..height of image)
        
    ==========================================================================================================================
    Measures curvature of both left and right polynomial (assuming they are both of second order)
    ==========================================================================================================================

    '''
    def measure_curvature_real(self,left_fit_cr, right_fit_cr, ploty):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension


        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curverad, right_curverad


    '''
    ==========================================================================================================================
    map_lane
    ==========================================================================================================================

    Arguments:
    
        - lmx: xs of corners
        - lmy: ys of corners
        - img: colour image that will received the lane
        
    ==========================================================================================================================
    draws the lane (in green) ahead of the car, on the provided image
    ==========================================================================================================================

    '''
    def map_lane(self,lmx,lmy,img,messages, src=None,dst=None):
        
        if src == None:
            src = self.src
            
        if dst == None:
            dst = self.dst

        # work on a copy
        canvas = img.copy()

        corners = np.array([[[x,y] for x,y in zip(lmx,lmy)]],dtype=np.int32 )

        # create a mask with white pixels
        mask = np.zeros(img.shape, dtype=np.uint8)


        # fill the ROI into the mask
        cv2.fillPoly(mask, corners, (0,255,0))

        mask = self.warper(mask,self.dst,self.src)

        # applying the mask to original image
        canvas =  cv2.addWeighted(canvas,1,mask,0.5,0)
        
        canvas = cv2.putText(canvas, messages[0], (50,60), 
                             cv2.FONT_HERSHEY_SIMPLEX, #font family
                             2, #font size
                             (255,255, 255), #font color
                             3) 
        
        canvas = cv2.putText(canvas, messages[1], (50,120), 
                             cv2.FONT_HERSHEY_SIMPLEX, #font family
                             2, #font size
                             (255,255, 255), #font color
                             3) 
        
        return canvas

    '''
    ==========================================================================================================================
    find_lanes
    ==========================================================================================================================

    Arguments:
    
        - img: colour image of the road, taken from a front-face camera
        - method (defaults to 'combo'): which gradient or method to use to find the lanes ('combo','lhs' or 'color')
        
    ==========================================================================================================================
    Full pipeline to find the lanes marked on the road, calculat its curvature and draw it on the image provided
    ==========================================================================================================================

    '''
    def find_lanes(self,img,method='combo',verbose=0):

        try:

            # undistort image
            imgu = self.undistort(img)

            # set src, and dst
            self.set_src_dst((imgu.shape[1], imgu.shape[0]))

            # warp image
            warped = self.warper(imgu)

            if method == 'lhs':
                # apply image thresholding and colour gradient
                imgu = self.hls_select(warped)
            elif method == 'colour':
                # apply image thresholding and colour gradient
                imgu = self.colour_select(warped)
            else:
                imgu = self.lane_binary(warped)
                
            # find lane lines
            with_lanes, left_fit_cr, right_fit_cr, offcentre_cr, position, ploty, polig_x, polig_y = self.fit_polynomial(imgu)

            # Calculate the radius of curvature in meters for both lane lines
            left_curverad, right_curverad = self.measure_curvature_real(left_fit_cr, right_fit_cr, ploty)

            messages = ['Radius of Curvature = {0:.0f}(m)'.format((left_curverad + right_curverad) / 2),
                        'Vehicle is {0:.2f}m {1:} of centre'.format(offcentre_cr,position) ]

            
            # map the lane ahead 
            result = self.map_lane(polig_x,polig_y,img, messages)

            # return picture with lane
            return result
        
        except:
            
            # if anything goes wrong, return original image
            return img
