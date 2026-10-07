import argparse
import cv2
import numpy as np
import ncnn


# ============================================================
# MNIST preprocessing
# ============================================================
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    print(f"Original image shape: {image.shape}")

    # MNIST images are 28x28
    image = cv2.resize(
        image,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # Convert uint8 [0,255] -> float32 [0,1]
    image = image.astype(np.float32) / 255.0

    # Same normalization used during PyTorch testing
    image = (image - 0.1307) / 0.3081

    print(f"Preprocessed image shape: {image.shape}")
    print(
        f"Input range: {image.min():.4f} to {image.max():.4f}"
    )
    print(f"Input mean: {image.mean():.6f}")
    print(f"Input std: {image.std():.6f}")

    return image


# ============================================================
# Create NCNN network
# ============================================================
def load_ncnn_model(param_path, bin_path):

    net = ncnn.Net()
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_arithmetic = False
    # Keep model files unchanged.
    net.load_param(param_path)
    net.load_model(bin_path)

    return net


# ============================================================
# Run NCNN inference
# ============================================================
def run_inference(net, image):

    # Convert NumPy array to NCNN Mat
    mat = ncnn.Mat(image)

    print("\nNCNN input:")
    print(f"  width    : {mat.w}")
    print(f"  height   : {mat.h}")
    print(f"  channels : {mat.c}")

    # Create extractor
    ex = net.create_extractor()

    # Reduce memory usage
    ex.set_light_mode(True)

    # Input
    ret = ex.input("in0", mat)

    if ret != 0:
        raise RuntimeError(
            f"NCNN input failed with return code: {ret}"
        )

    # Output
    ret, out = ex.extract("out0")

    if ret != 0:
        raise RuntimeError(
            f"NCNN extraction failed with return code: {ret}"
        )

    # Convert NCNN output to NumPy
    output = np.array(out, dtype=np.float32).flatten()

    return output


# ============================================================
# Analyze model output
# ============================================================
def analyze_output(output):

    print("\nOutput shape:", output.shape)

    print("\nRaw output:")

    for i, value in enumerate(output):
        print(f"  {i}: {value}")

    # --------------------------------------------------------
    # Check NaN
    # --------------------------------------------------------
    has_nan = np.any(np.isnan(output))

    # --------------------------------------------------------
    # Check +/- infinity
    # --------------------------------------------------------
    has_inf = np.any(np.isinf(output))

    # --------------------------------------------------------
    # Finite mask
    # --------------------------------------------------------
    finite_mask = np.isfinite(output)

    print("\nFinite values:")
    print(finite_mask)

    # --------------------------------------------------------
    # Convert log probabilities back to probabilities
    # --------------------------------------------------------
    probabilities = np.exp(output)

    # Convert any numerical -inf -> 0
    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Normalize probabilities
    # --------------------------------------------------------
    probability_sum = probabilities.sum()

    if probability_sum <= 0 or not np.isfinite(probability_sum):

        print("\nERROR: Invalid probability distribution.")

        return {
            "valid": False,
            "prediction": -1,
            "probabilities": None,
            "reason": "Invalid probability distribution"
        }

    probabilities = probabilities / probability_sum

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    predicted_class = int(np.argmax(probabilities))

    confidence = float(probabilities[predicted_class])

    # --------------------------------------------------------
    # Print probabilities
    # --------------------------------------------------------
    print("\nProbabilities:")

    for i, probability in enumerate(probabilities):

        print(
            f"  {i}: {probability:.8f}"
        )

    print("\n" + "=" * 60)
    print("MNIST Prediction")
    print("=" * 60)

    print(f"Predicted digit : {predicted_class}")
    print(f"Confidence      : {confidence:.6f}")

    if has_inf:
        print(
            "\nNote: -Inf was present in the log-probability output."
        )
        print(
            "It was converted to probability 0."
        )

    print("=" * 60)

    return {
        "valid": True,
        "prediction": predicted_class,
        "probabilities": probabilities,
        "reason": None
    }


# ============================================================
# Zero-input test
# ============================================================
def create_zero_input():

    # IMPORTANT:
    #
    # "zero input" here means zero-valued normalized input.
    #
    # This is useful for checking whether the NCNN model
    # itself generates NaN/Inf independent of image content.
    #

    image = np.zeros(
        (28, 28),
        dtype=np.float32
    )

    return image


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description="MNIST NCNN inference"
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
        "--image",
        help="Path to input image"
    )

    parser.add_argument(
        "--zero-test",
        action="store_true",
        help="Run zero-input numerical stability test"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Check arguments
    # --------------------------------------------------------
    if args.image is None and not args.zero_test:

        parser.error(
            "Please provide either --image or --zero-test"
        )

    if args.image is not None and args.zero_test:

        parser.error(
            "Use either --image or --zero-test, not both"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print("\nLoading NCNN model...")

    print(f"Param: {args.param}")
    print(f"Bin  : {args.bin}")

    net = load_ncnn_model(
        args.param,
        args.bin
    )

    print("NCNN model loaded successfully.")

    # --------------------------------------------------------
    # Normal image inference
    # --------------------------------------------------------
    if args.image is not None:

        print("\n" + "=" * 60)
        print("IMAGE INFERENCE")
        print("=" * 60)

        image = preprocess_image(args.image)

        output = run_inference(
            net,
            image
        )

        result = analyze_output(output)

        if not result["valid"]:

            print("\n" + "=" * 60)
            print("FINAL RESULT")
            print("=" * 60)

            print("Predicted digit: INVALID")
            print(f"Reason: {result['reason']}")

            print(
                "\nDo NOT use this prediction for evaluation."
            )

            print("=" * 60)

    # --------------------------------------------------------
    # Zero-input test
    # --------------------------------------------------------
    elif args.zero_test:

        print("\n" + "=" * 60)
        print("ZERO INPUT TEST")
        print("=" * 60)

        image = create_zero_input()

        print(
            f"Zero-test input range: "
            f"{image.min():.4f} to {image.max():.4f}"
        )

        output = run_inference(
            net,
            image
        )

        result = analyze_output(output)

        print("\n" + "=" * 60)

        if result["valid"]:
            print("Zero-input test PASSED.")
            print(
                "The model produced finite output."
            )
        else:
            print("Zero-input test FAILED.")
            print(
                "The NCNN model produces invalid "
                "NaN/Inf values even for a simple input."
            )

        print("=" * 60)


if __name__ == "__main__":
    main()
