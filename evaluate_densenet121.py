# =========================================================
# FINAL DENSENET121 TESTING + GRADCAM CODE
# =========================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# =========================================================
# LOAD MODEL
# =========================================================

model = load_model(
    'best_densenet121_model.h5',
    compile=False
)

print("Model Loaded Successfully!")

# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    'glioma',
    'meningioma',
    'no_tumor',
    'pituitary'
]

# =========================================================
# IMAGE PATH
# =========================================================

img_path = r"D:\Personal Data\Masters Data\Smester 2\Deep Learning\CA 1\Task 1\Dataset\Brain tumor\brisc2025\classification_task\test\glioma\brisc2025_test_00039_gl_ax_t1.jpg"

# =========================================================
# LOAD IMAGE
# =========================================================

img = image.load_img(
    img_path,
    target_size=(224,224)
)

img_array = image.img_to_array(img)

# Keep original copy
display_img = img_array.astype("uint8")

# Normalize
img_array = img_array / 255.0

# Expand dimensions
img_array = np.expand_dims(
    img_array,
    axis=0
)

# =========================================================
# PREDICTION
# =========================================================

predictions = model.predict(
    img_array,
    verbose=0
)

pred_index = np.argmax(
    predictions[0]
)

predicted_class = class_names[
    pred_index
]

confidence = np.max(
    predictions[0]
) * 100

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n===================================")
print("Predicted Disease :", predicted_class)
print("Confidence : {:.2f}%".format(confidence))
print("===================================")

print("\nClass Probabilities:\n")

for i, prob in enumerate(predictions[0]):

    print(
        f"{class_names[i]} : {prob*100:.2f}%"
    )

# =========================================================
# AI ANALYSIS
# =========================================================

if predicted_class == 'glioma':

    print(
        "\nAI Analysis : MRI scan indicates features associated with Glioma tumor."
    )

elif predicted_class == 'meningioma':

    print(
        "\nAI Analysis : MRI scan suggests characteristics of Meningioma."
    )

elif predicted_class == 'pituitary':

    print(
        "\nAI Analysis : MRI scan indicates possible Pituitary tumor."
    )

else:

    print(
        "\nAI Analysis : No tumor detected in MRI scan."
    )

# =========================================================
# BEST DENSENET GRADCAM LAYER
# =========================================================

last_conv_layer_name = 'conv5_block16_concat'

last_conv_layer = model.get_layer(
    last_conv_layer_name
)

print("\nUsing GradCAM Layer :", last_conv_layer_name)

# =========================================================
# CREATE GRADCAM MODEL
# =========================================================

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        last_conv_layer.output,
        model.output
    ]
)

# =========================================================
# COMPUTE GRADIENTS
# =========================================================

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        img_array
    )

    pred_index = tf.argmax(
        predictions[0]
    )

    loss = predictions[:, pred_index]

# Gradients
grads = tape.gradient(
    loss,
    conv_outputs
)

# Better localization
pooled_grads = tf.reduce_mean(
    tf.abs(grads),
    axis=(0,1,2)
)

# =========================================================
# HEATMAP GENERATION
# =========================================================

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(
    heatmap
)

heatmap = heatmap.numpy()

# ReLU
heatmap = np.maximum(
    heatmap,
    0
)

# Safe normalization
if np.max(heatmap) != 0:

    heatmap = heatmap / np.max(heatmap)

# =========================================================
# LOAD ORIGINAL IMAGE
# =========================================================

original_img = cv2.imread(
    img_path
)

original_img = cv2.cvtColor(
    original_img,
    cv2.COLOR_BGR2RGB
)

# =========================================================
# RESIZE HEATMAP
# =========================================================

heatmap = cv2.resize(
    heatmap,
    (
        original_img.shape[1],
        original_img.shape[0]
    )
)

# =========================================================
# APPLY COLORMAP
# =========================================================

heatmap = np.uint8(
    255 * heatmap
)

heatmap_color = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

heatmap_color = cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)

# =========================================================
# OVERLAY HEATMAP
# =========================================================

superimposed_img = cv2.addWeighted(
    original_img,
    0.6,
    heatmap_color,
    0.4,
    0
)

# =========================================================
# SAVE OUTPUT
# =========================================================

cv2.imwrite(
    'gradcam_result.jpg',
    cv2.cvtColor(
        superimposed_img,
        cv2.COLOR_RGB2BGR
    )
)

print("\nGradCAM image saved as: gradcam_result.jpg")

# =========================================================
# DISPLAY RESULTS
# =========================================================

plt.figure(figsize=(15,5))

# ---------------------------------------------------------
# ORIGINAL MRI
# ---------------------------------------------------------

plt.subplot(1,3,1)

plt.imshow(original_img)

plt.title("Original MRI")

plt.axis('off')

# ---------------------------------------------------------
# HEATMAP
# ---------------------------------------------------------

plt.subplot(1,3,2)

plt.imshow(heatmap_color)

plt.title("GradCAM Heatmap")

plt.axis('off')

# ---------------------------------------------------------
# FINAL OVERLAY
# ---------------------------------------------------------

plt.subplot(1,3,3)

plt.imshow(superimposed_img)

plt.title(
    f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
)

plt.axis('off')

# =========================================================
# SHOW
# =========================================================

plt.tight_layout()

plt.show()