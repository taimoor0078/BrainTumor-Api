from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

from gradcam import generate_gradcam

app = Flask(__name__)

os.makedirs("uploads", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)
os.makedirs("gradcams", exist_ok=True)

dense_model = load_model(
    "best_densenet121_model.h5",
    compile=False
)

custom_model = load_model(
    "best_custom_densenet121_model.h5",
    compile=False
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


def preprocess_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = img_array.astype(
        np.float32
    ) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


def predict_model(model, img_array):

    preds = model.predict(
        img_array,
        verbose=0
    )

    idx = np.argmax(preds[0])

    confidence = float(
        preds[0][idx]
    ) * 100

    probs = {}

    for i, name in enumerate(CLASS_NAMES):

        probs[name] = round(
            float(preds[0][i] * 100),
            2
        )

    return {
        "prediction": CLASS_NAMES[idx],
        "confidence": round(confidence, 2),
        "probabilities": probs
    }


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        unique_id = str(
            uuid.uuid4()
        )

        upload_path = os.path.join(
            "uploads",
            unique_id + ".jpg"
        )

        file.save(upload_path)

        img_array = preprocess_image(
            upload_path
        )

        dense_result = predict_model(
            dense_model,
            img_array
        )

        custom_result = predict_model(
            custom_model,
            img_array
        )

        dense_conf = dense_result[
            "confidence"
        ]

        custom_conf = custom_result[
            "confidence"
        ]

        dense_pred = dense_result[
            "prediction"
        ]

        custom_pred = custom_result[
            "prediction"
        ]

        agreement = (
            dense_pred ==
            custom_pred
        )

        confidence_gap = abs(
            dense_conf -
            custom_conf
        )

        if dense_conf >= custom_conf:

            final_result = dense_result

            selected_model = (
                "DenseNet121"
            )

            selected_tf_model = (
                dense_model
            )

        else:

            final_result = custom_result

            selected_model = (
                "CustomDenseNet121"
            )

            selected_tf_model = (
                custom_model
            )

        doctor_review = False

        if (
            agreement == False
            and
            confidence_gap < 5
        ):

            doctor_review = True

        heatmap_file, gradcam_file = (
            generate_gradcam(
                model=selected_tf_model,
                image_path=upload_path,
                output_name=unique_id
            )
        )

        base_url = (
            request.host_url.rstrip("/")
        )

        response_data = {

            "final_prediction":
                final_result[
                    "prediction"
                ],

            "final_confidence":
                final_result[
                    "confidence"
                ],

            "selected_model":
                selected_model,

            "agreement":
                agreement,

            "requires_doctor_review":
                doctor_review,

            "dense_result":
                dense_result,

            "custom_result":
                custom_result,

            "heatmap_url":
                f"{base_url}/heatmaps/{heatmap_file}",

            "gradcam_url":
                f"{base_url}/gradcams/{gradcam_file}"
        }

        print(
            "\n========== API RESPONSE =========="
        )

        print(
            response_data
        )

        print(
            "==================================\n"
        )

        return jsonify(
            response_data
        )

    except Exception as ex:

        return jsonify({
            "error": str(ex)
        }), 500


@app.route(
    "/heatmaps/<filename>"
)
def heatmap(filename):

    return send_from_directory(
        "heatmaps",
        filename
    )


@app.route(
    "/gradcams/<filename>"
)
def gradcam(filename):

    return send_from_directory(
        "gradcams",
        filename
    )


@app.route("/")
def home():

    return jsonify({
        "status": "running",
        "message": "Brain Tumor API Running"
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8001,
        debug=True
    )