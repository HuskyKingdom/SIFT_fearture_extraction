import cv2
import numpy as np
import math as mt
from matplotlib import pyplot as plt 
from functools import cmp_to_key
import os
from joblib import dump,load
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")


Num_WORDS = 500



def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
    """
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None

def computeGradientAtCenterPixel(pixel_array):
    """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return np.array([dx, dy, ds])

def computeHessianAtCenterPixel(pixel_array):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])



def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint
    """
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = int(gaussian_image[region_y, region_x + 1]) - int(gaussian_image[region_y, region_x - 1])
                    dy = int(gaussian_image[region_y - 1, region_x]) - int(gaussian_image[region_y + 1, region_x])
                    gradient_magnitude = mt.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) 
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < 1e-7:
                orientation = 0
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    """Find pixel positions of all scale-space extrema in the image pyramid
    """
    threshold = mt.floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints


def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise
    """
    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return np.all(center_pixel_value >= first_subimage) and \
                   np.all(center_pixel_value >= third_subimage) and \
                   np.all(center_pixel_value >= second_subimage[0, :]) and \
                   np.all(center_pixel_value >= second_subimage[2, :]) and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return np.all(center_pixel_value <= first_subimage) and \
                   np.all(center_pixel_value <= third_subimage) and \
                   np.all(center_pixel_value <= second_subimage[0, :]) and \
                   np.all(center_pixel_value <= second_subimage[2, :]) and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]
    return False


def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 /np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint
    """
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = mt.cos(np.deg2rad(angle))
        sin_angle = mt.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * mt.sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, mt.sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = int(gaussian_image[window_row, window_col + 1]) - int(gaussian_image[window_row, window_col - 1])
                        dy = int(gaussian_image[window_row - 1, window_col]) - int(gaussian_image[window_row + 1, window_col])
                        gradient_magnitude = mt.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = mt.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')

def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints


# SIFT Fearture Extraction
def SIFT(img):

    # Original Image - use original for now, but could double the size

    numOctave = int(mt.log(min(img.shape[0],img.shape[1]),2) - 3)
    # numLayers = n + 3 where n = num of layers we want to search for , stands for num of layers in an octave
    numLayers = 2 + 3
    baseSigma = 1.52 # From Low`s article, = sqrt(Sigma**2 - OriginSigma**2) where Sigma=1.6 OriginSigma = 0.5
    k = mt.pow(2,1/2) # Layer Sigma coefficient
    BottomImage = cv2.GaussianBlur(img,(0,0),baseSigma)

    # Genrate Kernels 
    gaussian_kernels = np.zeros(numLayers)
    gaussian_kernels[0] = baseSigma

    for index in range(1,numLayers): # Sigmas to apply within an octave
        last_sigma = (k ** (index - 1)) * baseSigma
        sigma_total = k * last_sigma
        gaussian_kernels[index] = mt.sqrt(sigma_total ** 2 - last_sigma ** 2)

    # Bluring images - Building Prymiad
    Guassian_Images_Prymid = []
    for Octaveindex in range(numOctave):
        CurrentOctaveImgs = []
        CurrentOctaveImgs.append(BottomImage)

        for LayerIndex in range(1,numLayers):
            Newimage = cv2.GaussianBlur(CurrentOctaveImgs[LayerIndex-1], (0, 0), sigmaX=gaussian_kernels[LayerIndex])
            CurrentOctaveImgs.append(Newimage)

        Guassian_Images_Prymid.append(CurrentOctaveImgs)
        # Create bottom image for next octave by choosing image with index -3 from previous octave
        BottomImage = CurrentOctaveImgs[-3]
        BottomImage = cv2.resize(BottomImage,(int(BottomImage.shape[1] / 2), int(BottomImage.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)

    """ PreviewImage = np.array(Guassian_Images_Prymid[0][0])
    PreviewImage = PreviewImage.astype(np.uint8)
    cv2.imshow("test",PreviewImage)
    cv2.waitKey(0) """

    # Building GoD by subtracting
    Dog_Prymiad = []
    for OctaveIndex in range(numOctave):
        Dog_Prymiad_Octave = []
        for LayerIndex in range(numLayers): 
            if(LayerIndex == 0): # ignore the bottom layer
                pass
            else:
                ResultImage = cv2.subtract(Guassian_Images_Prymid[OctaveIndex][LayerIndex],Guassian_Images_Prymid[OctaveIndex][LayerIndex-1])
                Dog_Prymiad_Octave.append(ResultImage)

        Dog_Prymiad.append(Dog_Prymiad_Octave)

    

    # Find Scale Spce Extrame where 0.04 = T
    keypoints = findScaleSpaceExtrema(Guassian_Images_Prymid,Dog_Prymiad,2,1.52,5)
    keypoints = removeDuplicateKeypoints(keypoints)
    descriptors = generateDescriptors(keypoints, np.array(Guassian_Images_Prymid))
    return descriptors,keypoints

def L2_distance(Fpoint,Spoint):
    acc_sum = 0
    for i in range(len(Fpoint)):
        acc_sum += (Fpoint[i] - Spoint[i])**2
    return mt.sqrt(acc_sum)







    
def clustering(descriptors,words=Num_WORDS,code="N"): # k-mean clustering
    
   
    if code == "S": #Trained features
        return load("clustering/Means"+str(16))

    Labels = []
    for i in range(len(descriptors)):
        Labels.append(-1)
    
    Means = []
    Converge_diff = 9999.999
    Round = 0
    if code == "N":
        for i in range(words):
            Means.append(np.random.random_integers(100,size=(128,)))
    else:
        Means = load("clustering/Means"+str(code))
        Round = int(code)
        

    while Converge_diff >= 1:
        print("---New Round")
        print("---K-means Clutering of Round " + str(Round) + " with Converge_diff " + str(Converge_diff))
        print("---Labelling...")
        # labelling
        for i_descriptor in range(len(descriptors)):
            if i_descriptor % 1000 == 0:
                print("-Dealing with Descriptor " + str(i_descriptor) + "/" + str(len(descriptors)))
            Min_distance = 9999.9
            for j_meanPoint in range(len(Means)):
                dis =  L2_distance(descriptors[i_descriptor],Means[j_meanPoint])
                if dis < Min_distance:
                    Min_distance = dis
                    Labels[i_descriptor] = j_meanPoint
        
        # recalculating mean
        print("---Recalculating Means")
        for i_mean in range(len(Means)):
            Total_value = np.zeros(128)
            Total_number = 0
            for j_descr in range(len(descriptors)):
                if Labels[j_descr] == i_mean:
                    for h_bit in range(128):
                        Total_value[h_bit] += descriptors[j_descr][h_bit]
                    Total_number += 1
            
            if (Total_number > 0): #update new center
                for h_bit in range(128):
                        Converge_diff += abs(Means[i_mean][h_bit] - (Total_value[h_bit] / Total_number))
                        Means[i_mean][h_bit] = Total_value[h_bit] / Total_number
                        
        dump(Means,"clustering/Means"+str(Round))
        Round += 1
    
    return Means
        


def labelling_to_mean(Means,Orin): # assigning Orin points to Means using L2 distance
    
    Labels = []
    for i in range(len(Orin)):
        Labels.append(-1)

    # labelling
    for i_descriptor in range(len(Orin)):
        if i_descriptor % 1000 == 0:
            print("-Labelling " + str(i_descriptor) + "/" + str(len(Orin)))
        Min_distance = 9999.9
        for j_meanPoint in range(len(Means)):
            dis =  L2_distance(Orin[i_descriptor],Means[j_meanPoint])
            if dis < Min_distance:
                Min_distance = dis
                Labels[i_descriptor] = j_meanPoint


    return Labels


def get_patch(img,sigma,x,y):

    size = int(sigma * 5 / 2)
    x = int(x) + 1
    y = int(y) + 1
    return img[x-size:x+size+1,y-size:y+size+1]


def find_imageIndex_with_index(index,points):

    return points[index].class_id

def get_class_label(image_index,num_of_images):
    # class 0-airplanes 1-cars 2-dog 3-faces 4-keyboard
    classLabel = -1
    if image_index < num_of_images[0]: # first class
        classLabel = 0
    elif image_index < num_of_images[0] + num_of_images[1]:
        classLabel = 1
    elif image_index < num_of_images[0] + num_of_images[1] +  num_of_images[2]:
        classLabel = 2
    elif image_index < num_of_images[0] + num_of_images[1] +  num_of_images[2] +  num_of_images[3]:
        classLabel = 3
    else:
        classLabel = 4

    return classLabel

def histogram_intersection(h1, h2):
    sm = 0
    for i in range(len(h1)):
        sm += min(h1[i], h2[i])
    return sm

def main_control():

    local_dir = "COMP338_Assignment1_Dataset/Training/"
    images = []
    Classes = ["airplanes","cars","dog","faces","keyboard"]
    Test_images = []
    num_of_train_images = [] # count num of train images in each class
    num_of_test_images = [] # count num of test images in each class
    
    # reading images
    for ipt in Classes:
        temp_count = 0
        print("\n---Reading from " + ipt + " dataset...")
        for i in range(1,81):
            
            if i <= 9:
                if os.path.exists(local_dir + ipt + "/000" + str(i) + ".jpg") != False:
                    print("Reading from " + local_dir + ipt + "/000" + str(i) + ".jpg") 
                    images.append(cv2.imread(local_dir + ipt + "/000" + str(i) + ".jpg",0))
                    temp_count += 1
            else:
                
                if os.path.exists(local_dir + ipt + "/00" + str(i) + ".jpg") != False:
                    print("Reading from " + local_dir + ipt + "/00" + str(i) + ".jpg") 
                    images.append(cv2.imread(local_dir + ipt + "/00" + str(i) + ".jpg",0))
                    temp_count += 1
            
        num_of_train_images.append(temp_count)

    print("Images Collected.")

    print("\n\n ________SIFT Image Classification________\n")
    print("1.Load existing training datas` pre-extracted features 2.Extract from images\n")
    input_val = input("Enter an option with its code: ")

    while True:

        if input_val == "2":
            
            # feature extraction step 1

            # Extracting SIFTs
            print("\n---Extracting SIFT features from training data...")
            SIFTs = [] # Discriptors and keypoints of training data
            SIFTs_Points = []
            SIFTs_test = [] # Discriptors and keypoints of test data
            SIFTs_Points_test = []

            count = 1
            keyPointsClass = 0

            for image in images:
                print("Extracting " + str(count) + "/" + str(len(images)) + "...")
                feature,point = SIFT(image)
                for i in range(len(feature)):
                    SIFTs.append(feature[i])
                    point[i].class_id = keyPointsClass
                    SIFTs_Points.append(point[i])
                count += 1
                keyPointsClass += 1

            print("Paking Points...")
            Paking = []
            for i in range(len(SIFTs_Points)):
                temp = (SIFTs_Points[i].pt, SIFTs_Points[i].size, SIFTs_Points[i].angle, SIFTs_Points[i].response, SIFTs_Points[i].octave, 
                SIFTs_Points[i].class_id)
                Paking.append(temp)

            print("Saving files.")
            dump(SIFTs,"features/SIFT_Features_Train")
            dump(Paking,"features/SIFT_Features_Train_Points")
            print("Features of training data extracted sucessfully. File saved.")

        elif input_val == "1":

            # feature extraction step 1
            print("\n---Loading SIFT features...")
            SIFTs = load("features/SIFT_Features_Train")
            SIFTs_Points = []
            Paking = load("features/SIFT_Features_Train_Points")
            for i in range(len(Paking)):
                temp_feature = cv2.KeyPoint(x=Paking[i][0][0],y=Paking[i][0][1],size=Paking[i][1], angle=Paking[i][2], 
                response=Paking[i][3], octave=Paking[i][4],class_id=Paking[i][5]) 
                SIFTs_Points.append(temp_feature)
            SIFTs_test = []
            SIFTs_Points_test = []
            print("Features extracted sucessfully.")

        else:
            print("Bye.")
            break

        # clastering step 2
        input_val_t = input("Do you want to load Clustered centers? Y or N :")
        if input_val_t == "Y":
            Centers = load("cluster")
        else:
            input_val_t = input("If you wish to continue with files saved, please enter its index, otherwith enter N :")
            Centers = KMeans(n_clusters=Num_WORDS,random_state=0,max_iter=50000).fit(SIFTs)
            dump(Centers,"cluster")
        
        # Image representation with histogram of code words - step 3

        # Feature extraction from test dataset
        
        print("\n---Reeading images from Test dataset...")
        for class_index in Classes:
            temp_count = 0
            for i in os.listdir("COMP338_Assignment1_Dataset/Test/" + class_index):
                print("Reading From COMP338_Assignment1_Dataset/Test/" + class_index + "/" + i)
                Test_images.append(cv2.imread("COMP338_Assignment1_Dataset/Test/" + class_index + "/" + i,0))
                temp_count += 1
            
            num_of_test_images.append(temp_count)

        print("Testing images collected!")

        print("\n---Extracting Features from Test dataset...")
        input_val = input("Do you wish to 1.load features from test images or  2.extract features from test images :")
        if input_val == "1":
            SIFTs_test = load("features/SIFT_Features_Test")
            SIFTs_Points_test = []
            Paking = load("features/SIFT_Features_Test_Points")
            for i in range(len(Paking)):
                temp_feature = cv2.KeyPoint(x=Paking[i][0][0],y=Paking[i][0][1],size=Paking[i][1], angle=Paking[i][2], 
                response=Paking[i][3], octave=Paking[i][4], class_id=Paking[i][5]) 
                SIFTs_Points_test.append(temp_feature)
            print("SIFT extracted for test images loaded successfully.")
        else:

            count = 1
            keyPointsClass = 0
            for image in Test_images:
                print("Extracting " + str(count) + "/" + str(len(Test_images)) + "...")
                feature,point = SIFT(image)
                for i in range(len(feature)):
                    SIFTs_test.append(feature[i])
                    point[i].class_id = keyPointsClass
                    SIFTs_Points_test.append(point[i])
                count += 1
                keyPointsClass += 1

            print("Saving file...")
            print("Paking Points...")
            Paking = []
            for i in range(len(SIFTs_Points_test)):
                temp = (SIFTs_Points_test[i].pt, SIFTs_Points_test[i].size, SIFTs_Points_test[i].angle, SIFTs_Points_test[i].response, SIFTs_Points_test[i].octave, 
                SIFTs_Points_test[i].class_id)
                Paking.append(temp)
            dump(SIFTs_test,"features/SIFT_Features_Test")
            dump(Paking,"features/SIFT_Features_Test_Points")
            print("SIFT extracted for test images, file saved.")

        # Labelling descriptors in datasets 
        print("\n---Image representing")
        input_val = input("- 1.Load labels for Train & Test 2.Label them :")

        if input_val == "2":
            # Training
            Traing_labels = Centers.predict(SIFTs)
            dump(Traing_labels,"features/Traing_labels")
            
            # Testing
            Test_labels = Centers.predict(SIFTs_test)
            dump(Test_labels,"features/Test_labels")
        else:
            Traing_labels = load("features/Traing_labels")
            Test_labels = load("features/Test_labels")

        print("Labels are ready.")

        # Visualize some image patches
        input_val = input("Enter a number to view some image patch from this word class(-1 if done): ")
        while(input_val  != "-1" ):
            print("Finding image patch with word class " + input_val + "...")
            plt.figure("Image patches preview - Word class " + input_val)
            num_images = 0
            for label_index in range(len(Traing_labels)):
                if num_images >= 4:
                    break
                if Traing_labels[label_index] == int(input_val):
                    plt.subplot(2,2,num_images+1)
                    plt.title("Image with index " + str(find_imageIndex_with_index(label_index,SIFTs_Points)) + " in the dataset")
                    plt.imshow(get_patch(images[find_imageIndex_with_index(label_index,SIFTs_Points)],SIFTs_Points[label_index].size,SIFTs_Points[label_index].pt[0],SIFTs_Points[label_index].pt[1]) ,cmap=plt.cm.gray)
                    num_images += 1
            if num_images != 0:
                plt.show()
                input_val = input("Enter a number to view some image patch from this word class(-1 if done): ")
            else:
                print("Info. Not found with word index " + input_val + "...")
                plt.close()
                input_val = input("Enter a number to view some image patch from this word class(-1 if done): ")

        # creating histograms
        print("\n---Creating BOW Histogrames for Training images...")
        Traing_hist = []
        Testing_hist = []

        print("Initiallizing arrays...")
        for i in range(len(images)): # that much images
            Traing_hist.append([])
            for k in range(Num_WORDS): # that many histogram elements in each image
                Traing_hist[i].append(0)

        for i in range(len(Test_images)): # that much images
            Testing_hist.append([])
            for k in range(Num_WORDS): # that many histogram elements in each image
                Testing_hist[i].append(0)

        print("Counting for training dataset...")       
        for descriptor_index in range(len(SIFTs)):
            Traing_hist[SIFTs_Points[descriptor_index].class_id][Traing_labels[descriptor_index]] += 1
        
        print("Counting for test dataset...")
        for descriptor_index in range(len(SIFTs_test)):
            Testing_hist[SIFTs_Points_test[descriptor_index].class_id][Test_labels[descriptor_index]] += 1

        # Normallization 
        for image_index in range(len(Traing_hist)):
            Nomalization_factor = 0
            for each in range(len(Traing_hist[image_index])):
                Nomalization_factor += Traing_hist[image_index][each]
            for each in range(len(Traing_hist[image_index])):
                Traing_hist[image_index][each] /= Nomalization_factor
        
        for image_index in range(len(Testing_hist)):
            Nomalization_factor = 0
            for each in range(len(Testing_hist[image_index])):
                Nomalization_factor += Testing_hist[image_index][each]
            for each in range(len(Testing_hist[image_index])):
                Testing_hist[image_index][each] /= Nomalization_factor

        print("Counting Done,file saved.")
        dump(Traing_hist,"features/Traing_hist")
        dump(Testing_hist,"features/Test_hist")

        # Classification - step4 (Nearest Neighbor)
        Classified_labels = []
        for test_index in range(len(Testing_hist)):
            minDist = 99999
            minDistIndex = -1
            for train_index in range(len(Traing_hist)):
                Dist = L2_distance(Testing_hist[test_index],Traing_hist[train_index])
                if Dist < minDist:
                    minDist = Dist
                    minDistIndex = train_index
            # label to its closest neighbor in training data
            Classified_labels.append(get_class_label(minDistIndex,num_of_train_images))
        
        # Evaluation - step5
        print("\n---Evaluating performance...")
        Grand_Turth = []
        for image_index in range(len(Test_images)):
            Grand_Turth.append(get_class_label(image_index,num_of_test_images))
        
        #overall errors
        num_Error = 0
        for i in range(len(Grand_Turth)):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1

        Overall_errors = num_Error / 50
        #class errors
        num_Error = 0 #airplane
        for i in range(10):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1
        A_errors = num_Error / 10
        num_Error = 0 #cars
        for i in range(10,20):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1
        C_errors = num_Error / 10
        num_Error = 0 #dog
        for i in range(20,30):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1
        D_errors = num_Error / 10
        num_Error = 0 #faces
        for i in range(30,40):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1
        F_errors = num_Error / 10
        num_Error = 0 #keyboard
        for i in range(40,50):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1
        K_errors = num_Error / 10

        print("===Performance Result===")
        print("[Overall Error rate :" + str(Overall_errors) +"]")
        print("Airplane class rate :" + str(A_errors))
        print("Cars Error rate :" + str(C_errors) )
        print("Dogs Error rate :" + str(D_errors) )
        print("Faces Error rate :" + str(F_errors) )
        print("Keyboard Error rate :" + str(K_errors) )


        input_val = input("\nEnter 0-4 to see images that are correctly classified(-1 to skip):")
        while input_val != "-1":
            if input_val == "0":
                for i in range(10):
                    if Grand_Turth[i] == Classified_labels[i]:
                        cv2.imshow("Correct classification of class airplane",Test_images[i])
                        cv2.waitKey(0)
            elif input_val == "1":
                for i in range(10,20):
                    if Grand_Turth[i] == Classified_labels[i]:
                        cv2.imshow("Correct classification of class cars",Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "2":
                for i in range(20,30):
                    if Grand_Turth[i] == Classified_labels[i]:
                        cv2.imshow("Correct classification of class dog",Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "3":
                for i in range(30,40):
                    if Grand_Turth[i] == Classified_labels[i]:
                        cv2.imshow("Correct classification of class faces",Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "4":
                for i in range(40,50):
                    if Grand_Turth[i] == Classified_labels[i]:
                        cv2.imshow("Correct classification of class keyboard",Test_images[i])
                        cv2.waitKey(0)
                        break
            else:
                break
            input_val = input("\nEnter 0-4 to see images that are correctly classified(-1 to skip):")
        

        
        input_val = input("\nEnter 0-4 to see images that are incorrectly classified:")
        while input_val != "-1":
            if input_val == "0":
                for i in range(10):
                    if Grand_Turth[i] != Classified_labels[i]:
                        cv2.imshow("MissClassified as " + Classes[Classified_labels[i]],Test_images[i])
                        cv2.waitKey(0)
            elif input_val == "1":
                for i in range(10,20):
                    if Grand_Turth[i] != Classified_labels[i]:
                        cv2.imshow("MissClassified as " + Classes[Classified_labels[i]],Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "2":
                for i in range(20,30):
                    if Grand_Turth[i] != Classified_labels[i]:
                        cv2.imshow("MissClassified as " + Classes[Classified_labels[i]],Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "3":
                for i in range(30,40):
                    if Grand_Turth[i] != Classified_labels[i]:
                        cv2.imshow("MissClassified as " + Classes[Classified_labels[i]],Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "4":
                for i in range(40,50):
                    if Grand_Turth[i] != Classified_labels[i]:
                        cv2.imshow("MissClassified as " + Classes[Classified_labels[i]],Test_images[i])
                        cv2.waitKey(0)
                        break
            else:
                break
                

            input_val = input("\nEnter 0-4 to see images that are incorrectly classified(-1 to skip):")

        print("\n---Calculating Confusion Metrics...")
        metrix = confusion_matrix(Grand_Turth,Classified_labels,labels=[0,1,2,3,4])
        # printing confusion matrix
        for i in range(5):
            print(metrix[i])
        print("Please Note that both row and col here is arranged in the order of airplanes, cars, dog, faces, keyboard.")

        

        # step 6



        print("\n---Using Histogram Intersection to comparing histograms...")
  
        # Classification - step6 (Nearest Neighbor -- Insersection)
        Classified_labels = []
        for test_index in range(len(Testing_hist)):
            max = 0.0001
            maxIndex = -1
            for train_index in range(len(Traing_hist)):
                Dist = histogram_intersection(Testing_hist[test_index],Traing_hist[train_index])  
                if Dist > max:
                    max = Dist
                    maxIndex = train_index
            # label to its closest neighbor in training data
            Classified_labels.append(get_class_label(maxIndex,num_of_train_images))

        # Evaluation - step5
        print("\n---Evaluating performance...")
        Grand_Turth = []
        for image_index in range(len(Test_images)):
            Grand_Turth.append(get_class_label(image_index,num_of_test_images))
        
        #overall errors
        num_Error = 0
        for i in range(len(Grand_Turth)):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1

        Overall_errors = num_Error / 50
        #class errors
        num_Error = 0 #airplane
        for i in range(10):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1
        A_errors = num_Error / 10
        num_Error = 0 #cars
        for i in range(10,20):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1
        C_errors = num_Error / 10
        num_Error = 0 #dog
        for i in range(20,30):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1
        D_errors = num_Error / 10
        num_Error = 0 #faces
        for i in range(30,40):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1
        F_errors = num_Error / 10
        num_Error = 0 #keyboard
        for i in range(40,50):
            if Grand_Turth[i] != Classified_labels[i]:
                num_Error += 1
        K_errors = num_Error / 10

        print("===Performance Result===")
        print("[Overall Error rate :" + str(Overall_errors) +"]")
        print("Airplane class rate :" + str(A_errors))
        print("Cars Error rate :" + str(C_errors) )
        print("Dogs Error rate :" + str(D_errors) )
        print("Faces Error rate :" + str(F_errors) )
        print("Keyboard Error rate :" + str(K_errors) )


        input_val = input("\nEnter 0-4 to see images that are correctly classified(-1 to skip):")
        while input_val != "-1":
            if input_val == "0":
                for i in range(10):
                    if Grand_Turth[i] == Classified_labels[i]:
                        cv2.imshow("Correct classification of class airplane",Test_images[i])
                        cv2.waitKey(0)
            elif input_val == "1":
                for i in range(10,20):
                    if Grand_Turth[i] == Classified_labels[i]:
                        cv2.imshow("Correct classification of class cars",Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "2":
                for i in range(20,30):
                    if Grand_Turth[i] == Classified_labels[i]:
                        cv2.imshow("Correct classification of class dog",Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "3":
                for i in range(30,40):
                    if Grand_Turth[i] == Classified_labels[i]:
                        cv2.imshow("Correct classification of class faces",Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "4":
                for i in range(40,50):
                    if Grand_Turth[i] == Classified_labels[i]:
                        cv2.imshow("Correct classification of class keyboard",Test_images[i])
                        cv2.waitKey(0)
                        break
            else:
                break
            input_val = input("\nEnter 0-4 to see images that are correctly classified(-1 to skip):")
        

        
        input_val = input("\nEnter 0-4 to see images that are incorrectly classified:")
        while input_val != "-1":
            if input_val == "0":
                for i in range(10):
                    if Grand_Turth[i] != Classified_labels[i]:
                        cv2.imshow("MissClassified as " + Classes[Classified_labels[i]],Test_images[i])
                        cv2.waitKey(0)
            elif input_val == "1":
                for i in range(10,20):
                    if Grand_Turth[i] != Classified_labels[i]:
                        cv2.imshow("MissClassified as " + Classes[Classified_labels[i]],Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "2":
                for i in range(20,30):
                    if Grand_Turth[i] != Classified_labels[i]:
                        cv2.imshow("MissClassified as " + Classes[Classified_labels[i]],Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "3":
                for i in range(30,40):
                    if Grand_Turth[i] != Classified_labels[i]:
                        cv2.imshow("MissClassified as " + Classes[Classified_labels[i]],Test_images[i])
                        cv2.waitKey(0)
                        break
            elif input_val == "4":
                for i in range(40,50):
                    if Grand_Turth[i] != Classified_labels[i]:
                        cv2.imshow("MissClassified as " + Classes[Classified_labels[i]],Test_images[i])
                        cv2.waitKey(0)
                        break
            else:
                break
                

            input_val = input("\nEnter 0-4 to see images that are incorrectly classified(-1 to skip):")

        print("\n---Calculating Confusion Metrics...")
        metrix = confusion_matrix(Grand_Turth,Classified_labels,labels=[0,1,2,3,4])
        # printing confusion matrix
        for i in range(5):
            print(metrix[i])
        print("Please Note that both row and col here is arranged in the order of airplanes, cars, dog, faces, keyboard.")
        


        break

    

main_control()


