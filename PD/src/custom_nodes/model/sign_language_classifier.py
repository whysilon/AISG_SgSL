"""
Node template for creating custom nodes.
"""

from typing import Any, Dict
from time import sleep
from peekingduck.pipeline.nodes.abstract_node import AbstractNode

from typing import Any, Dict

import cv2
import numpy as np
import tensorflow as tf

from peekingduck.pipeline.nodes.node import AbstractNode

IMG_HEIGHT = 60
IMG_WIDTH = 60
SEQUENCE_LENGTH = 24

class Node(AbstractNode):
   """Initializes and uses a RNN to predict if a video frame of length 24 is a certain sign langauge
   """
   def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
      super().__init__(config, node_path=__name__, **kwargs)
      self.model = tf.keras.models.load_model('LRCN_SGSL_model.h5')

   def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
      """ Reads the image input and returns the predicted class label and
      confidence score.

      Args:
            inputs (dict): Dictionary with key "img".

      Returns:
            outputs (dict): Dictionary with keys "pred_label" and "pred_score".
      """
      # Take input out from the custom dabble node which processes
      # the previous 24 frames
      imgs = inputs["img"]
      imgs = np.expand_dims(imgs, axis=0)
      
      # Check if the frames length is correct
      if(imgs.shape[1] == 24):
            predictions = self.model.predict(imgs)
            #print(predictions)
            score = tf.nn.softmax(predictions[0])
            #print(score)
            return {
                  "pred_label": self.class_label_map[np.argmax(score)],
                  "pred_score": 100.0 * np.max(score),
            }
      # Else return empty output
      else:
            return {
                  "pred_label": "None",
                  "pred_score": 0
            }
