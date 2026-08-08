import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def generate_gradcam(
    model,
    image_path,
    output_name
):

    img = image.load_img(
        image_path,
        target_size=(224,224)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # =====================================
    # FIND LAST CONV LAYER
    # =====================================

    last_conv_layer = None

    for layer in reversed(model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            last_conv_layer = layer.name

            break

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(
                last_conv_layer
            ).output,
            model.output
        ]
    )

    # =====================================
    # GRADIENTS
    # =====================================

    with tf.GradientTape() as tape:

        conv_outputs,preds = grad_model(
            img_array
        )

        pred_index = tf.argmax(
            preds[0]
        )

        class_channel = preds[
            :,
            pred_index
        ]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs *
        pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap.numpy()

    if np.max(heatmap) != 0:

        heatmap = (
            heatmap /
            np.max(heatmap)
        )

    # =====================================
    # ORIGINAL MRI
    # =====================================

    original_img = cv2.imread(
        image_path
    )

    original_img = cv2.cvtColor(
        original_img,
        cv2.COLOR_BGR2RGB
    )

    heatmap = cv2.resize(
        heatmap,
        (
            original_img.shape[1],
            original_img.shape[0]
        )
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # =====================================
    # SAVE HEATMAP
    # =====================================

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_file = (
        output_name +
        "_heatmap.jpg"
    )

    cv2.imwrite(
        "heatmaps/" +
        heatmap_file,
        heatmap_color
    )

    # =====================================
    # SAVE GRADCAM
    # =====================================

    heatmap_rgb = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    gradcam = cv2.addWeighted(
        original_img,
        0.65,
        heatmap_rgb,
        0.35,
        0
    )

    gradcam_file = (
        output_name +
        "_gradcam.jpg"
    )

    cv2.imwrite(
        "gradcams/" +
        gradcam_file,
        cv2.cvtColor(
            gradcam,
            cv2.COLOR_RGB2BGR
        )
    )

    return (
        heatmap_file,
        gradcam_file
    )