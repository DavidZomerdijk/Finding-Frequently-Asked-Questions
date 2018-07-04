import os
from glob import glob
import tensorflow as tf

class Model(object):
  """Abstract object representing an Reader model."""
  def __init__(self):
    pass

  def get_model_dir(self):
    model_dir = self.dataset
    for attr in self._attrs:
      if hasattr(self, attr):
        model_dir += "/%s=%s" % (attr, getattr(self, attr))
    return model_dir

  def save(self, checkpoint_dir, global_step=None):
    self.saver = tf.train.Saver()

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__
    model_dir = self.get_model_dir()

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(self.sess, 
        os.path.join(checkpoint_dir, model_name), global_step=global_step)

  def initialize(self, log_dir="./logs"):
    self.merged_sum = tf.merge_all_summaries()
    self.writer = tf.train.SummaryWriter(log_dir, self.sess.graph_def)

    tf.initialize_all_variables().run()
    self.load(self.checkpoint_dir)

    start_iter = self.step.eval()

  def load(self, checkpoint_dir):
    self.saver = tf.train.Saver()

    print(" [*] Loading checkpoints...")
    model_dir = self.get_model_dir()
    print(checkpoint_dir)
    # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    checkpoint_dir = checkpoint_dir + "/" + model_dir
    print(checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state( checkpoint_dir )
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Load SUCCESS")
      return True
    else:
      print(" [!] Load failed...")
      return False

  def linear(self, input_, output_size, scope=None, bias=True):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
      matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
      bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

  def recall(self, x, y, n=3):
    ind = y.argsort()[-n:][::-1]
    l = []
    for i in ind:
      l.append( 1 if  x[i] > 0 else 0 )
    return sum(l) / len(l)