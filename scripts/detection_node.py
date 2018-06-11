#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from drive_ros_msgs.msg import BoundingBox
from drive_ros_msgs.msg import BoundingBoxArray
from cv_bridge import CvBridge, CvBridgeError
from darkflow.net.build import TFNet


# image detection class
class image_detection:

  def __init__(self, model_file, weights_file, labels_file, threshold, config_folder, gpu_usage, gpu_name):

    # initialize darkflow
    options = {"model": model_file, "load": weights_file, "threshold": threshold,
               "batch": 1, "gpu": gpu_usage, "gpuName": gpu_name, "summary": None,
               "config": config_folder, "labels": labels_file}
    self.model = TFNet(options)

    # create ros publisher and subscriber
    self.bridge = CvBridge()
    self.image_pub = rospy.Publisher("detected_img",Image,queue_size=1)
    self.obj_pub =   rospy.Publisher("detected_bb", BoundingBoxArray, queue_size=1)
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

    # create new output message
    pub_msg = BoundingBoxArray()

    # save results of neural network in output message
    for res in result:
      if 'car' in res['label']:

        bb = BoundingBox()
        bb.header.stamp    = data.header.stamp
        bb.header.frame_id = data.header.frame_id

        bb.x1 = res['topleft']['x']
        bb.y1 = res['topleft']['y']
        bb.x2 = res['bottomright']['x']
        bb.y2 = res['bottomright']['y']

        bb.confidence = float(res['confidence'])
        bb.classID = 0
        pub_msg.boxes.append(bb)


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
    gpu_name =      rospy.get_param('~gpu_name')

    # initialize image detection class
    image_detection(model_file, weights_file, labels_file, threshold, config_folder, gpu_usage, gpu_name)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == "__main__":
    node()

