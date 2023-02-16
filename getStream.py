import cv2
import numpy as np
from urllib.request import urlopen

url = "http://192.168.0.218:8080/?action=stream" # Your video streaming url. don't try with this url!!
stream = urlopen(url)
buffer = b''

while True:
    buffer += stream.read(4096) # read by buffer 
    head = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')
    if head > -1 and end > -1:
        jpg = buffer[head:end+2]
        buffer = buffer[end+2:]
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imshow("stream", img)
    key = cv2.waitKey(1)
    if key == 27:
        # if you push the ESC key, 
        break

cv2.destroyAllWindows()