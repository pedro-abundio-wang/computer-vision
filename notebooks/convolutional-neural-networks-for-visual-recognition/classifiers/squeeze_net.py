import tensorflow as tf

NUM_CLASSES = 1000


class SqueezeNet(object):
    
    
    def fire_module(self, x, inp, sp, e11p, e33p):
        """fire module
        """
        with tf.variable_scope("fire"):
            with tf.variable_scope("squeeze"):
                W = tf.get_variable(name="weights", shape=[1,1,inp,sp])
                b = tf.get_variable(name="bias", shape=[sp])
                s = tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding="VALID") + b
                s = tf.nn.relu(s)
            with tf.variable_scope("e11"):
                W = tf.get_variable(name="weights", shape=[1,1,sp,e11p])
                b = tf.get_variable(name="bias", shape=[e11p])
                e11 = tf.nn.conv2d(input=s, filter=W, strides=[1,1,1,1], padding="VALID") + b
                e11 = tf.nn.relu(e11)
            with tf.variable_scope("e33"):
                W = tf.get_variable(name="weights", shape=[3,3,sp,e33p])
                b = tf.get_variable(name="bias", shape=[e33p])
                e33 = tf.nn.conv2d(input=s, filter=W, strides=[1,1,1,1], padding="SAME") + b
                e33 = tf.nn.relu(e33)
                
            return tf.concat(values=[e11, e33], axis=-1, name='concat')
    
    
    def extract_features(self, input=None, reuse=True):
        """Create a SqueezeNet model.
        Inputs:
        - input: optional input to the model. If None, will use placeholder for input.
        """
        if input is None:
            input = self.image
        x = input
        
        layers = []
        
        with tf.variable_scope('features', reuse=reuse):
            with tf.variable_scope('layer0'):
                W = tf.get_variable(name="weights", shape=[3,3,3,64])
                b = tf.get_variable(name="bias", shape=[64])
                x = tf.nn.conv2d(input=x, filter=W, strides=[1,2,2,1], padding="VALID")
                x = tf.nn.bias_add(x, b)
                layers.append(x)
            with tf.variable_scope('layer1'):
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('layer2'):
                x = tf.nn.max_pool(value=x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer3'):
                x = self.fire_module(x, inp=64, sp=16, e11p=64, e33p=64)
                layers.append(x)
            with tf.variable_scope('layer4'):
                x = self.fire_module(x, inp=128, sp=16, e11p=64, e33p=64)
                layers.append(x)
            with tf.variable_scope('layer5'):
                x = tf.nn.max_pool(value=x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer6'):
                x = self.fire_module(x, inp=128, sp=32, e11p=128, e33p=128)
                layers.append(x)
            with tf.variable_scope('layer7'):
                x = self.fire_module(x, inp=256, sp=32, e11p=128, e33p=128)
                layers.append(x)
            with tf.variable_scope('layer8'):
                x = tf.nn.max_pool(value=x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer9'):
                x = self.fire_module(x, inp=256, sp=48, e11p=192, e33p=192)
                layers.append(x)
            with tf.variable_scope('layer10'):
                x = self.fire_module(x, inp=384, sp=48, e11p=192, e33p=192)
                layers.append(x)
            with tf.variable_scope('layer11'):
                x = self.fire_module(x, inp=384, sp=64, e11p=256, e33p=256)
                layers.append(x)
            with tf.variable_scope('layer12'):
                x = self.fire_module(x, inp=512, sp=64, e11p=256, e33p=256)
                layers.append(x)
                
        return layers

    
    def __init__(self, save_path=None, sess=None):
        """Create a SqueezeNet model.
        Inputs:
        - save_path: path to TensorFlow checkpoint
        - sess: TensorFlow session
        - input: optional input to the model. If None, will use placeholder for input.
        """
        self.image = tf.placeholder(dtype='float', shape=[None,None,None,3], name='input_image')
        self.labels = tf.placeholder(dtype='int32', shape=[None], name='labels')
        x = self.image
        
        self.layers = []
        self.layers = self.extract_features(x, reuse=False)
        self.features = self.layers[-1]
        
        with tf.variable_scope('classifier'):
            with tf.variable_scope('layer0'):
                x = self.features
                self.layers.append(x)
            with tf.variable_scope('layer1'):
                W = tf.get_variable(name="weights", shape=[1,1,512,1000])
                b = tf.get_variable(name="bias", shape=[1000])
                x = tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding="VALID")
                x = tf.nn.bias_add(x, b)
                self.layers.append(x)
            with tf.variable_scope('layer2'):
                x = tf.nn.relu(x)
                self.layers.append(x)
            with tf.variable_scope('layer3'):
                x = tf.nn.avg_pool(value=x, ksize=[1,13,13,1], strides=[1,13,13,1], padding='VALID')
                self.layers.append(x)
                
        self.classifier = tf.reshape(x, [-1, NUM_CLASSES])

        if save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
            
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, NUM_CLASSES), logits=self.classifier))
