import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
from PIL import ImageDraw
import contextlib
import sys
import infer
import shutil
from scipy.ndimage.measurements import label
import scipy
# from roi import inference
def get_rect(x, y, width, height, angle):
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect
def find_ROI(image):
    # blue = image[:, :, 2]
    # image = np.where(blue>254,1,0).astype('uint8')
    s = [[1,1,1],
         [1,1,1],
         [1,1,1]]
    labels = label(image, structure=s)
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        x = np.min(nonzerox)
        y = np.min(nonzeroy)
        width = (np.max(nonzerox) - np.min(nonzerox)) * 2
        height = (np.max(nonzeroy) - np.min(nonzeroy)) * 2
        x -= (width / 4)
        y -= (height / 4)

        img = Image.fromarray(image)

        # Draw a rotated rectangle on the image.
        draw = ImageDraw.Draw(img)
        rect = get_rect(x=x, y=y, width=width, height=height, angle=0.0)
        draw.polygon([tuple(p) for p in rect], fill=1)
        # Convert the Image data to a numpy array.
        image = np.asarray(img)


    return image
class DummyFile(object):
    def write(self, x): pass
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    yield
    sys.stdout = save_stdout
with nostdout():
    import main
file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")
video = skvideo.io.vread(file)
roads = main.predict_images('roads', video)
vehicles = main.predict_images('vehicles', video)
# i = 0
# for vehicle in vehicles:
#     vehicle = find_ROI(vehicle)
#     scipy.misc.imsave('masks/'+str(i)+'.png', vehicle)
#     i+=1
# s_vehicles = infer.main()
# vehicles = vehicles + s_vehicles
# for vehicle in vehicles:
#     vehicle[vehicle > 0] = 1
# print ('vehicles[0]', vehicles[0])
# vehicles = inference.main(vehicles, video)
# print ('roadsshape', roads[0].shape)
answer_key = {}
frame = 1
for road, vehicle in zip(roads, vehicles):
    answer_key[frame] = [encode(vehicle), encode(road)] # [encode(binary_car_result), encode(binary_road_result)]
    frame += 1

# Print output in proper json format
sys.stdout = sys.__stdout__
print (json.dumps(answer_key))