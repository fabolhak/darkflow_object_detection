#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon
from geometry_msgs.msg import Point32
from darkflow_vehicle_detection.msg import Objects
from cv_bridge import CvBridge, CvBridgeError
from darkflow.net.build import TFNet


# image detection class
class image_detection:

  def __init__(self, model_file, weights_file, labels_file, threshold, config_folder, gpu_usage):

    # initialize darkflow
    options = {"model": model_file, "load": weights_file, "threshold": threshold,
               "batch": 1, "gpu": gpu_usage, "summary": None, "config": config_folder,
               "labels": labels_file}
    self.model = TFNet(options)

    # create ros publisher and subscriber
    self.bridge = CvBridge()
    self.image_pub = rospy.Publisher("detected_img",Image,queue_size=1)
    self.obj_pub = rospy.Publisher("detected_obj", Objects, queue_size=1)
    self.image_sub = rospy.Subscriber("input_img", Image, self.callback, queue_size=1)

  def callback(self,data):

    # convert message data to image
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
      return

    # run the neural network and get outputs
    result = self.model.return_predict(cv_image)

    # create new objects message
    pub_msg = Objects()
    pub_msg.header.stamp = data.header.stamp
    pub_msg.header.frame_id = data.header.frame_id

    # save results of neural network in objects message
    for res in result:
      if 'car' in res['label']:
        poly = Polygon()
        # top left
        poly.points.append(Point32(res['topleft']['x'], res['topleft']['y'], 0))
        # top right
        poly.points.append(Point32(res['bottomright']['x'], res['topleft']['y'], 0))
        # bottom left
        poly.points.append(Point32(res['topleft']['x'], res['bottomright']['y'], 0))
        # bottom right
        poly.points.append(Point32(res['bottomright']['x'], res['bottomright']['y'], 0))

        pub_msg.obj.append(poly)
        pub_msg.confidence.append(float(res['confidence']))

    # publish outputs message
    self.obj_pub.publish(pub_msg)

    # if someone subscribed to debug image -> publish it
    if 0 < self.image_pub.get_num_connections():

      # draw rectangles in image
      for res in result:
        if 'car' in res['label']:
          cv2.rectangle(cv_image, (res['topleft']['x'], res['topleft']['y']),
                                  (res['bottomright']['x'], res['bottomright']['y']),
                                  (0, 255, 0), 3)

      # convert to image message and publish
      try:
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
      except CvBridgeError as e:
        print(e)


# ros node
def node():
    rospy.init_node('detection_node')

    # get parameters
    model_file =    rospy.get_param('~model_file')
    weights_file =  rospy.get_param('~weights_file')
    labels_file =   rospy.get_param('~labels_file')
    threshold =     rospy.get_param('~threshold')
    config_folder = rospy.get_param('~config_folder')
    gpu_usage =     rospy.get_param('~gpu_usage')

    # initialize image detection class
    image_detection(model_file, weights_file, labels_file, threshold, config_folder, gpu_usage)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == "__main__":
    node()

