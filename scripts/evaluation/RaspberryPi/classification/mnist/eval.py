import argparse
import csv
import json
import os

import cv2
import numpy as np
import ncnn


# ============================================================
# Load NCNN model
# ============================================================

def load_model(param_path, bin_path):
    net = ncnn.Net()
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_arithmetic = False
    ret = net.load_param(param_path)
    if ret != 0:
        raise RuntimeError(
            f"Failed to load param file: {param_path}"
        )

    ret = net.load_model(bin_path)
    if ret != 0:
        raise RuntimeError(
            f"Failed to load model file: {bin_path}"
        )

    return net


# ============================================================
# Preprocess image
#
# Same as:
#
# transforms.ToTensor()
# transforms.Normalize((0.1307,), (0.3081,))
# ============================================================

def preprocess_image(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    # Resize to MNIST size
    image = cv2.resize(
        image,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # uint8 [0,255] -> float32 [0,1]
    image = image.astype(np.float32) / 255.0

    # MNIST normalization
    image = (image - 0.1307) / 0.3081

    # Check for invalid values
    if not np.all(np.isfinite(image)):
        raise RuntimeError(
            f"Image contains NaN/Inf: {image_path}"
        )

    return image


# ============================================================
# Run inference
# ============================================================

def predict(net, image):

    # Create NCNN input
    mat = ncnn.Mat(image)

    extractor = net.create_extractor()

    # Input
    ret = extractor.input("in0", mat)

    if ret != 0:
        raise RuntimeError(
            f"NCNN input failed with return code {ret}"
        )

    # Inference
    ret, output = extractor.extract("out0")

    if ret != 0:
        raise RuntimeError(
            f"NCNN extraction failed with return code {ret}"
        )

    # Convert output to NumPy
    output = np.array(
        output,
        dtype=np.float32
    ).flatten()

    if len(output) != 10:
        raise RuntimeError(
            f"Expected 10 outputs, got {len(output)}"
        )

    return output


# ============================================================
# Main evaluation
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate an NCNN MNIST model on a test dataset."
    )

    parser.add_argument(
        "--param",
        required=True,
        help="Path to NCNN .param file"
    )

    parser.add_argument(
        "--bin",
        required=True,
        help="Path to NCNN .bin file"
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing test images"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV file containing filename,label"
    )

    parser.add_argument(
        "--output",
        default="evaluation_results.csv",
        help="Output CSV containing predictions"
    )

    parser.add_argument(
        "--json",
        default="evaluation_summary.json",
        help="Output JSON summary"
    )

    args = parser.parse_args()

    # ========================================================
    # Load model
    # ========================================================

    print("Loading NCNN model...")

    net = load_model(
        args.param,
        args.bin
    )

    print("Model loaded successfully.")

    # ========================================================
    # Read CSV
    # ========================================================

    samples = []

    with open(
        args.csv,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if "filename" not in reader.fieldnames:
            raise RuntimeError(
                "CSV must contain a 'filename' column."
            )

        if "label" not in reader.fieldnames:
            raise RuntimeError(
                "CSV must contain a 'label' column."
            )

        for row in reader:

            filename = row["filename"].strip()
            label = int(row["label"])

            samples.append(
                {
                    "filename": filename,
                    "label": label
                }
            )

    print(f"Samples in CSV: {len(samples)}")

    # ========================================================
    # Evaluation
    # ========================================================

    total = 0
    correct = 0
    skipped = 0

    confusion_matrix = np.zeros(
        (10, 10),
        dtype=np.int64
    )

    results = []

    print("\nStarting evaluation...\n")

    for index, sample in enumerate(samples):

        csv_filename = sample["filename"]
        true_label = sample["label"]

        # ----------------------------------------------------
        # Match filename
        #
        # CSV:
        #     821.png
        #
        # Image folder:
        #     821.jpg
        # ----------------------------------------------------

        stem = os.path.splitext(csv_filename)[0]

        image_path = os.path.join(
            args.images,
            stem + ".jpg"
        )

        # If exact .jpg does not exist, try original filename
        if not os.path.exists(image_path):

            alternative_path = os.path.join(
                args.images,
                csv_filename
            )

            if os.path.exists(alternative_path):
                image_path = alternative_path

            else:
                print(
                    f"[WARNING] Image not found: "
                    f"{csv_filename}"
                )

                skipped += 1
                continue

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        try:

            image = preprocess_image(
                image_path
            )

            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

            scores = predict(
                net,
                image
            )

            # ------------------------------------------------
            # Check output
            # ------------------------------------------------

            if not np.all(np.isfinite(scores)):

                print(
                    f"[WARNING] Non-finite output: "
                    f"{csv_filename}"
                )

                predicted_label = -1

            else:

                predicted_label = int(
                    np.argmax(scores)
                )

        except Exception as e:

            print(
                f"[ERROR] {csv_filename}: {e}"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Update statistics
        # ----------------------------------------------------

        total += 1

        if predicted_label == true_label:

            correct += 1

        confusion_matrix[
            true_label,
            predicted_label
        ] += 1

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "filename": csv_filename,
            "image_path": image_path,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": predicted_label == true_label
        }

        # Store scores
        for digit in range(10):
            result[f"score_{digit}"] = float(
                scores[digit]
            )

        results.append(result)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (index + 1) % 100 == 0:

            accuracy = (
                correct / total * 100
                if total > 0
                else 0
            )

            print(
                f"Processed {index + 1}/{len(samples)} | "
                f"Accuracy: {accuracy:.2f}%"
            )

    # ========================================================
    # Overall metrics
    # ========================================================

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    accuracy_percent = accuracy * 100

    # ========================================================
    # Per-class accuracy
    # ========================================================

    per_class_accuracy = {}

    for digit in range(10):

        class_total = np.sum(
            confusion_matrix[digit]
        )

        class_correct = (
            confusion_matrix[digit, digit]
        )

        if class_total > 0:

            class_accuracy = (
                class_correct / class_total
            )

        else:

            class_accuracy = 0.0

        per_class_accuracy[str(digit)] = (
            class_accuracy
        )

    # ========================================================
    # Print results
    # ========================================================

    print("\n")
    print("=" * 60)
    print("MNIST NCNN EVALUATION")
    print("=" * 60)

    print(f"Total CSV samples : {len(samples)}")
    print(f"Evaluated         : {total}")
    print(f"Skipped           : {skipped}")
    print(f"Correct           : {correct}")
    print(f"Incorrect         : {total - correct}")
    print(
        f"Accuracy          : "
        f"{accuracy_percent:.2f}%"
    )

    # ========================================================
    # Per-class accuracy
    # ========================================================

    print("\nPer-class accuracy:")

    for digit in range(10):

        class_total = np.sum(
            confusion_matrix[digit]
        )

        class_correct = (
            confusion_matrix[digit, digit]
        )

        class_accuracy = (
            class_correct / class_total * 100
            if class_total > 0
            else 0
        )

        print(
            f"  Digit {digit}: "
            f"{class_correct}/{class_total} "
            f"({class_accuracy:.2f}%)"
        )

    # ========================================================
    # Confusion matrix
    # ========================================================

    print("\nConfusion Matrix:")
    print(
        "Rows = True label, "
        "Columns = Predicted label"
    )

    print()

    print(
        "      " +
        " ".join(
            f"{i:5d}" for i in range(10)
        )
    )

    for digit in range(10):

        print(
            f"{digit:3d} | " +
            " ".join(
                f"{confusion_matrix[digit, j]:5d}"
                for j in range(10)
            )
        )

    # ========================================================
    # Save detailed predictions CSV
    # ========================================================

    if results:

        fieldnames = list(
            results[0].keys()
        )

        with open(
            args.output,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(results)

        print(
            f"\nDetailed predictions saved to: "
            f"{args.output}"
        )

    # ========================================================
    # Save JSON summary
    # ========================================================

    summary = {

        "total_csv_samples": len(samples),

        "evaluated_samples": total,

        "skipped_samples": skipped,

        "correct_predictions": correct,

        "incorrect_predictions": total - correct,

        "accuracy": accuracy,

        "accuracy_percent": accuracy_percent,

        "per_class_accuracy":
            per_class_accuracy,

        "confusion_matrix":
            confusion_matrix.tolist()
    }

    with open(
        args.json,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        f"Evaluation summary saved to: "
        f"{args.json}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
