# Modified-SIFT-Algorithm

The UGS_Modified_Feature_Detection implements a length and slope filter on the OpenCV SIFT algorithm. The filter increases the accuracy of the matches SIFT finds. This algorithm was designed to automatically georeference images. The image set used for testing can be found at: 

https://imagery.geology.utah.gov/pages/search.php?search=%21collection21441+&k=&modal=&display=thumbs&order_by=date&offset=0&per_page=350&archive=&sort=DESC&restypes=&recentdaylimit=&foredit=&noreload=true&access=

The algorithm has a MSE of 5.14*10^-6 and 1.97*10^-6 for latitude and longitude respectively. 
