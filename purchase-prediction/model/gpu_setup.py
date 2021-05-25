"""
    Name: main.py
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
import tensorflow as tf


def limit_gpu_usage():
    # limit GPU usage
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # don't pre-allocate memory; allocate as-needed
    config.gpu_options.per_process_gpu_memory_fraction = 0.95  # limit memory to be allocated
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))  # create sess w/ above settings
