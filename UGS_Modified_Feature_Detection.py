import math
import shutil
import numpy as np
import cv2 as cv
import requests
import ee
import os
import re
import exifread


MAX_ACCEPTABLE_LENGTH = 925
MIN_ACCEPTABLE_LENGTH = 775
MIN_GCPS_NEEDED = 3

RELATIVE_TOLERANCE_LENGTH = 0.1
ABSOLUTE_TOLERANCE_LENGTH = 15
RELATIVE_TOLERANCE_SLOPE = 0.1
ABSOLUTE_TOLERANCE_SLOPE = 0.1

MIN_MATCH_COUNT = 4
FLANN_INDEX_KDTREE = 1

# To find Photo Collections go to https://developers.google.com/earth-engine/datasets/catalog/landsat
LANDSAT_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
NAIP_COLLECTION = "USDA/NAIP/DOQQ"

PHOTO_LOCATION = 'ScrapingTest'
MATCHED_PHOTO_LOCATION = 'TestPotentialMatches\\test'
NOT_MATCHED_PHOTO_LOCATION = 'TestNotEnoughPoints\\test'

# This is the height and the width (since I use a square to get the photo) of a photo in degrees for the NAIP photos
HEIGHT_OF_PHOTO_IN_DEGREES = 0.08
WRITE_TO_FILE = 'Updated_Algorithm_Coords_Pixel_Location.txt'
NUM_PHOTOS_TO_PROCESS = 5


# Resizes the image
def resize(file_location, photo_trim_to_size):
    file_location = os.path.join(os.path.dirname(__file__), file_location)
    img = cv.imread(file_location, 0)
    scale_percent = getScalePercent(img.shape[0], photo_trim_to_size)
    new_width = int(img.shape[1] * scale_percent / 100)
    new_height = int(img.shape[0] * scale_percent / 100)
    dim = (new_width, new_height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


# Takes an img and crops it by 4.5% on each side
def crop(img):
    crop_size_width = round(0.045 * img.shape[1])
    crop_size_height = round(0.045 * img.shape[0])
    # Since height is associated with rows and width is associated with cols, we return the following
    return img[crop_size_height:img.shape[0] - crop_size_height, crop_size_width:img.shape[1] - crop_size_width]


# I arbitrarily wanted all photos to be around 800 pixels
def getScalePercent(height, photo_trim_size):
    return round((photo_trim_size / height) * 100)


# Gets the second image from earth engine
def getImageFromEE(crds):
    location = ee.Geometry.Point(crds[1], crds[0])

    # Below creates our region of interest from which we will pull the photo from ee
    roi = ee.Geometry.Rectangle(
        [crds[1] - 0.04, crds[0] - 0.04, crds[1] + 0.04, crds[0] + 0.04]).bounds()

    # This gets images from a specific collection. We need the image collection to look like the photos we are comparing to
    data = ee.ImageCollection(NAIP_COLLECTION).filter(ee.Filter.date('2017-01-01', '2018-12-31'))

    # Download the image from Earth Engine so we can read it in with cv.imread
    img = ee.Image(data.filterBounds(location).sort('CLOUD_COVER').first())

    # Get the bands of the photo
    bands = img.bandNames().getInfo()
    url = img.getThumbURL({'region': roi, 'scale': 10, 'bands': bands[0:3], 'format': 'png'})
    req = requests.get(url, stream=True)

    # Temporarily save the photo so we can read it in with cv.imread
    img_filename = 'temp_aerial_photo.png'
    with open(img_filename, 'wb') as out_file:
        shutil.copyfileobj(req.raw, out_file)

    # Now for the mosaic
    mosaic = data.mosaic()
    mos_bands = mosaic.bandNames().getInfo()
    mos_url = mosaic.getThumbURL({'region': roi, 'scale': 10, 'bands': mos_bands[0:3], 'format': 'png'})
    mos_req = requests.get(mos_url, stream=True)

    mos_img_filename = 'mos_aerial_photo.png'
    with open(mos_img_filename, 'wb') as mos_out_file:
        shutil.copyfileobj(mos_req.raw, mos_out_file)
    return cv.imread(img_filename, cv.IMREAD_GRAYSCALE), cv.imread(mos_img_filename, cv.IMREAD_GRAYSCALE)


# Pass in two images to do the feature detection on and two photos to draw the matches on
def runFeatureDetection(img1, img2, coords, to_add_width, to_add_height):
    # Create the SIFT detector
    sift = cv.SIFT_create()

    # Find the Keypoints and Descriptors using SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = tuple()
    if len(kp1) > 1 and len(kp2) > 1:
        matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # If we have enough matches, we will find the homography
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape

        matchesMask, was_updated = updateMaskDuplicates(src_pts, dst_pts, matchesMask)
        if checkLengthsAndSlope(src_pts, dst_pts, matchesMask, w):
            saveCoordAndPixelLocations(src_pts, dst_pts, matchesMask, img2.shape, coords, to_add_width, to_add_height)
            case_val = 2
        else:
            case_val = 1
    else:
        case_val = 1
        matchesMask = None

    # Draws the matches in green
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    return img3, case_val


# Takes the indices of the mode and updates the mask so all the points going to the same spot aren't drawn
def updateMaskDuplicates(start_pts, end_pts, match_mask):
    mode_indices = getDuplicateIndices(start_pts, end_pts, match_mask)
    was_updated = False
    for index in mode_indices:
        # if we are supposed to draw a redundant point, we set it so we don't. Note, we don't check if matchMask at index
        # is 1 because we do that previously
        match_mask[index] = 0
        was_updated = True
    return match_mask, was_updated


# Finds duplicates in a list and returns the indices that contain the duplicates
def getDuplicateIndices(start_pts, end_pts, mask):
    found_points = list()
    duplicate_indices = set()
    for i in range(len(end_pts)):
        # If we are going to use this point
        if mask[i] == 1:
            # If we have already found this point, add the index of the first index of the point and the current index
            if isDuplicate(end_pts[i], found_points):
                # This gets the index of the first occurrence of the end point
                first_occurrence = getFirstIndex(end_pts, end_pts[i], mask)
                # If these are different lines, then we have a bad end point and will delete both lines
                if (start_pts[first_occurrence][0][0] != start_pts[i][0][0]) or \
                        (start_pts[first_occurrence][0][1] != start_pts[i][0][1]):
                    duplicate_indices.add(first_occurrence)
                    duplicate_indices.add(i)
                # If these are the same line, we will just delete the newly found line
                elif (start_pts[first_occurrence][0][0] == start_pts[i][0][0]) or \
                        (start_pts[first_occurrence][0][1] == start_pts[i][0][1]):
                    duplicate_indices.add(i)
            else:
                found_points.append(end_pts[i])
    return duplicate_indices


# Goes through the list to see if the point is found
def isDuplicate(point, list_to_search):
    for val in list_to_search:
        # if the points match, we can immediately return
        if point[0][0] == val[0][0] and point[0][1] == val[0][1]:
            return True
    return False


# This will return the index of the first duplicate value of a point
def getFirstIndex(end_pts, point, mask):
    for i in range(len(end_pts)):
        if mask[i] == 1:
            if end_pts[i][0][0] == point[0][0] and end_pts[i][0][1] == point[0][1]:
                return i


# Finds the lengths of points that are acceptable (a 1 in the mask) and checks if the length is considered "good". Note,
# the mask we are passing in has no more duplicate end points
def checkLengthsAndSlope(start_pts, end_pts, mask, width):
    lengths = list()
    slopes = list()
    len_indices_for_mask = list()
    similar_lengths = list()
    indices = list()
    similar_slopes = list()
    for i in range(len(start_pts)):
        if mask[i] == 1:
            lengths.append(math.hypot(start_pts[i, 0, 0] - (end_pts[i, 0, 0] + width), start_pts[i, 0, 1] - end_pts[i, 0, 1]))
            len_indices_for_mask.append(i)
        slopes.append((start_pts[i, 0, 1] - end_pts[i, 0, 1]) / (start_pts[i, 0, 0] - (end_pts[i, 0, 0] + width)))

    # This will give us all the lengths that are similar as well as the indices of those lengths. We need to make sure
    # the lengths list is not empty
    if len(lengths) > 0:
        similar_lengths, indices = groupLikeItems(lengths, len_indices_for_mask, mask, RELATIVE_TOLERANCE_LENGTH, ABSOLUTE_TOLERANCE_LENGTH)

    # If we have enough good lengths, we will check the slopes
    if len(similar_lengths) >= MIN_GCPS_NEEDED:
        updated_slopes = list()
        for i in indices:
            updated_slopes.append(slopes[i])
        # if there are still slopes after filtering the length, we will then filter the slopes
        if len(updated_slopes) > 0:
            similar_slopes, useless_indices = groupLikeItems(updated_slopes, indices, mask, RELATIVE_TOLERANCE_SLOPE, ABSOLUTE_TOLERANCE_SLOPE)

        if len(similar_slopes) >= MIN_GCPS_NEEDED:
            return True
        else:
            return False
    # If we don't have enough good lengths, then we don't know if we have a good photo match
    else:
        return False


# Groups similar items together (slope and length)
def groupLikeItems(to_group_list, indices, mask, rel_tol, abs_tol):
    to_sort = list(list())
    associated_indices = list(list())
    averages = list()
    # Below is the index we will use to index into indices to keep track of what list has what indices associated with it
    cur_index = 0
    for to_group in to_group_list:
        updated = False
        # if we are just starting
        if len(to_sort) == 0:
            to_sort.append(list())
            associated_indices.append(list())
            to_sort[0].append(to_group)
            associated_indices[0].append(indices[cur_index])
            averages.append(to_group)
        else:
            for i in range(len(to_sort)):
                # Just grab the first slope we added
                # if current slope is close enough to another slope, we will add group the slopes together
                if math.isclose(averages[i], to_group, rel_tol=rel_tol, abs_tol=abs_tol):
                    averages[i] = ((averages[i] * len(to_sort[i])) + to_group) / (len(to_sort[i]) + 1)
                    to_sort[i].append(to_group)
                    associated_indices[i].append(indices[cur_index])
                    updated = True
                    break
            # If we haven't updated, that means there are no other similar slopes and we must append
            # a new list with the slope in it
            if not updated:
                to_add = list()
                to_add.append(to_group)
                to_sort.append(to_add)
                averages.append(to_group)

                # Now we update the associated indices list of lists
                index_to_add = list()
                index_to_add.append(indices[cur_index])
                associated_indices.append(index_to_add)
        cur_index += 1
    largest_group, index = getLargestGroup(to_sort)
    updateMask(associated_indices[index], mask)
    return largest_group, associated_indices[index]


# Take in a list of lists, find the longest list and return it as well as the index of the list we just found
# in the outermost list
def getLargestGroup(to_sort):
    max_list = list()
    max_index = 0
    cur_index = 0
    for inner_list in to_sort:
        if len(inner_list) > len(max_list):
            max_list = inner_list
            max_index = cur_index
        cur_index += 1
    return max_list, max_index


# The final update to the mask. We will only keep the points that have similar slopes. Indices should be a list
def updateMask(indices, mask):
    for i in range(len(mask)):
        if i not in indices:
            mask[i] = 0


# This will take in the start points, end points, mask, dimensions of the second image, the coordinate we used to
# get the NAIP image, and the width and height we need to add. The saved pixels will be in reference to the full sized UGS
# image, that is why we add the width or height depending on the photo. Using these variables, we will save the pixel
# location of the start points in the UGS photo associated with their coordinates values which will be found using the NAIP photo
def saveCoordAndPixelLocations(start_pts, end_pts, mask, img_dim, coords, to_add_width, to_add_height):
    # Note, img_dim gives the number of rows first (the height) and then the number of columns second (the width)
    # With the above note in mind, we save the coords in pixel form as (x,y).
    # Since coordinate is in the exact middle of the photo, I just divide by 2
    coords_pixel_location = (img_dim[1] / 2, img_dim[0] / 2)
    degree_per_pixel_height = HEIGHT_OF_PHOTO_IN_DEGREES / img_dim[0]
    degree_per_pixel_width = HEIGHT_OF_PHOTO_IN_DEGREES / img_dim[1]
    to_write = "(" + str(coords[0]) + ", " + str(coords[1]) + "), (" + str(degree_per_pixel_width) + ", " + str(degree_per_pixel_height) + "),"

    for i in range(len(mask)):
        # If this is a valid endpoint
        if mask[i] == 1:
            to_write += " [(" + str(start_pts[i][0][0] + to_add_width) + ", " + str(start_pts[i][0][1] + to_add_height) + "), "
            diff_in_x = coords_pixel_location[0] - end_pts[i][0][0]
            diff_in_y = coords_pixel_location[1] - end_pts[i][0][1]
            # coords is in form (lat, long) so we add the difference in y to the lat and subtract the difference in x to long
            to_write += "(" + str(coords[0] + (diff_in_y * degree_per_pixel_height)) + ", " + str(coords[1] - (diff_in_x * degree_per_pixel_width)) + ")]"
    to_write += '\n\n'
    with open(WRITE_TO_FILE, "a") as f:
        f.write(to_write)


# Get the coords from the metadata of an image
def getCoordsFromMetadata(file_name):
    # We just need to read from the file
    with open(file_name, 'rb') as img_file:
        tags = exifread.process_file(img_file, details=False)
        if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
            lat_dms = tags["GPS GPSLatitude"]
            long_dms = tags["GPS GPSLongitude"]
        return getDecimalCoords(lat_dms, long_dms)


# Convert from DMS to decimal (lat, long)
def getDecimalCoords(lat_dms, long_dms):
    lat = lat_dms.values[0] + (lat_dms.values[1] / 60) + (lat_dms.values[2] / 3600)
    # TODO See if there is a better way to find out if something is negative
    long = -(long_dms.values[0] + (long_dms.values[1] / 60) + (long_dms.values[2] / 3600))
    return float(lat), float(long)


# Grab a photo and compare it to the stuff we downloaded
def main():
    with open(WRITE_TO_FILE, 'w') as f:
        f.write(
            "(starting coords), (width conversion, height conversion), [(start pixel location), (associated coordinate value)]\n\n")

    try:
        ee.Initialize()
    except:
        ee.Authenticate(auth_mode='paste')
        ee.Initialize()

    cwd = os.getcwd()
    for filename in os.listdir(PHOTO_LOCATION):
        image_file_location = os.path.join(PHOTO_LOCATION, filename)
        # use the below image id for downloaded files from the UGS website
        image_id = filename[3:9]

        # Use the line below to get the coords in (lat, long) form from the metadata of an image.
        coordinates = getCoordsFromMetadata(image_file_location)

        # Get the image from earth engine, 0 means use LandSat. 1 uses NAIP
        img2, mosaic = getImageFromEE(coordinates)

        # We need to resize the image because a .tif takes too long to work with.
        # We will resize it to be about the same size height as the EE image.
        img1 = resize(image_file_location, mosaic.shape[0])

        photo_to_process = 0
        write_to_file = None
        case_val = 0
        img3 = None
        while photo_to_process < NUM_PHOTOS_TO_PROCESS:
            write_to_file = image_id + '_' + coordinates[0].__str__() + '_' + coordinates[1].__str__() + '.jpg'
            # Now we do feature detection using the mosaicked NAIP img
            if photo_to_process == 0:
                # Normal feature detection with the mosaic. No need to add anything to the height or width
                img3, case_val = runFeatureDetection(img1, mosaic, coordinates, 0, 0)
            # Upper half. No need to add any height or width
            elif photo_to_process == 1:
                img3, case_val = runFeatureDetection(img1[0:round(img1.shape[0] * 0.5), 0:], mosaic, coordinates, 0, 0)
                write_to_file = write_to_file[:-4] + '_upper_half' + write_to_file[-4:]
            # Lower half. Need to add half of img1's height
            elif photo_to_process == 2:
                img3, case_val = runFeatureDetection(img1[round(img1.shape[0] * 0.5):, 0:], mosaic, coordinates, 0, round(img1.shape[0] * 0.5))
                write_to_file = write_to_file[:-4] + '_lower_half' + write_to_file[-4:]
            # Left half. No need to add height or width
            elif photo_to_process == 3:
                img3, case_val = runFeatureDetection(img1[0:, 0:round(img1.shape[1] * 0.5)], mosaic, coordinates, 0, 0)
                write_to_file = write_to_file[:-4] + '_left_half' + write_to_file[-4:]
            # Right half. Need to add half of img1's width
            elif photo_to_process == 4:
                img3, case_val = runFeatureDetection(img1[0:, round(img1.shape[1] * 0.5):img1.shape[1]], mosaic, coordinates, round(img1.shape[1] * 0.5), 0)
                write_to_file = write_to_file[:-4] + '_right_half' + write_to_file[-4:]

            # If one of the above if statements is a good match, we will break out of the loop and go to the next photo.
            # If we don't find any good matches, the right half image will be saved to the Bad Matches file location
            if case_val == 2:
                break
            photo_to_process += 1

        # If we don't have enough matches, add the image to the right folder
        if case_val == 1:
            os.chdir(NOT_MATCHED_PHOTO_LOCATION)
        elif case_val == 2:
            os.chdir(MATCHED_PHOTO_LOCATION)

        cv.imwrite(write_to_file, img3)
        # Change the directory back to the original
        os.chdir(cwd)


if __name__ == '__main__':
    main()
