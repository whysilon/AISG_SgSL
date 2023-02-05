"""
Node template for creating custom nodes.
"""

from typing import Any, Dict
import cv2
import numpy as np
from peekingduck.pipeline.nodes.abstract_node import AbstractNode
import os

IMG_HEIGHT = 60
IMG_WIDTH = 60
SEQUENCE_LENGTH = 24

class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does .

        Args:
            inputs (dict): Dictionary with keys "img".

        Returns:
            outputs (dict): Dictionary with keys "img".
        """
        img = inputs["img"]
        img = cv2.cvtColor(inputs["img"], cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = np.expand_dims(img, axis=0)
        print(img.shape)
        try:
            imgs = np.load('frames.npy')
            imgs = np.append(imgs,img,axis = 0)
        except:
            print("Load Failed! Saving new frame")
            np.save("frames.npy",img)
            imgs = np.load('frames.npy')
        imgs = imgs[-SEQUENCE_LENGTH:]
        print(imgs.shape)
        np.save("frames.npy", imgs)
        if(imgs.shape[0] == 24):
            return {"img" : imgs}
        else:
            return {"img" : img}