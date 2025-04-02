import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('C:/Users/ABIJITH RAJA B/Desktop/Next gen connect/sign_language_interpreter_model.h5')

# Save the model in TensorFlow SavedModel format
model.save('C:/Users/ABIJITH RAJA B/Desktop/Next gen connect/saved_model')
