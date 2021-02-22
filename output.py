import zipfile, os, cv2, csv, shutil
from os.path import abspath

if not os.path.exists('output'):
    os.makedirs('output')
absolute_path = abspath('output')

with open("test.preds.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
      if i > 0:
        image = cv2.imread(os.path.join('./test', line[0]))
          
        if line[1] == 'rotated_left':
          image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
          
        elif line[1] == 'rotated_right':
          image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
          
        elif line[1] == 'upside_down':
          image = cv2.rotate(image, cv2.cv2.ROTATE_180)
        image_path = os.path.join(absolute_path,line[0])
        cv2.imwrite(image_path, image)

zf = zipfile.ZipFile("output.zip", "w")
for dirname, subdirs, files in os.walk("output"):
    zf.write(dirname)
    for filename in files:
        zf.write(os.path.join(dirname, filename))
zf.close()

#shutil.rmtree('output', ignore_errors=True)