# flaskFakeNewsDetectorapp
it detects fake news using machine leaning.  
  
 # *steps to set up*      

      1. flaskFakeNewsApp.py 
            flask file  
      2. grub3.py 
            ml file
      3. _static:_ 
            1.css file
      4. _templates:_
            1.index.html
            2.update.html
            3.ml.html
            * (_name_) = directory *
      
# *libraries required*  

      import matplotlib.pyplot as plt
      import tensorflow.compat.v1 as tf
      tf.disable_v2_behavior()
      import re
      import random
      import numpy as np
      import pandas as pd
      from sklearn.naive_bayes import BernoulliNB
      from sklearn.feature_extraction.text import TfidfVectorizer
      from sklearn.preprocessing import LabelEncoder
      from sklearn.utils import shuffle
      from sklearn.model_selection import train_test_split
