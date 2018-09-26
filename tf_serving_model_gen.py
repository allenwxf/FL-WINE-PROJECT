import numpy as np
import tensorflow as tf

# The export path contains the name and the version of the model

tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference

model = tf.keras.models.load_model('./model/variety_prediction_zh.h5')

print(model.input)
print(model.outputs)


export_path = './model/VarietyPredictionZh/1'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key

with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_bow': model.input[0], 'input_price': model.input[1], 'input_embed': model.input[2]},
        outputs={'outputs': model.outputs[0]}
    )
