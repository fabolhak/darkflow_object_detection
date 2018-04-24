#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from darkflow.net.build import TFNet


# image detection class
class image_detection:

  def __init__(self):
    options = {"model": "../darkflow/cfg/yolo.cfg", "load": "../nets/yolo_50.weights", "threshold": 0.5922, "batch": 1, "gpu": 0.9, "summary": None, "config": "../darkflow/cfg/"}
    self.model = TFNet(options)
    self.bridge = CvBridge()
    self.image_pub = rospy.Publisher("/left/detected",Image,queue_size=1)
    self.image_sub = rospy.Subscriber("/left/image_rect_color", Image, self.callback, queue_size=1)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    result = self.model.return_predict(cv_image)

    for res in result:
      if 'car' in res['label']:
        cv2.rectangle(cv_image, (res['topleft']['x'], res['topleft']['y']), (res['bottomright']['x'], res['bottomright']['y']), (0, 255, 0), 3)


    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)


# ros node
def node():
    rospy.init_node('detection_node')
    image_detection()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == "__main__":
    node()

