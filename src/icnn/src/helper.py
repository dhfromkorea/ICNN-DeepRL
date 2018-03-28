import tensorflow as tf

def variable_summaries(var, name=None, suffix=None):
    if name is None:
        if suffix is None:
            name = var.name
        else:
            name = '/'.join(var.name.split('/')[:-1])+'/'+suffix
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stdev'):
            stdev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stdev/' + name, stdev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)
