# # Crop processing utility

# # # # Single images

# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os

# # Load the exported TFLite model
# tflite_model = YOLO("best_float16.tflite")

# # Load the image
# image_path = r"C:\Users\HP\Downloads\WhatsApp Image 2026-07-09 at 6.20.44 PM.jpeg"
# image = cv2.imread(image_path)

# # Run inference
# results = tflite_model(image_path)

# # Create output directory for cropped faces
# output_dir = "cropped_faces_PR_new"
# os.makedirs(output_dir, exist_ok=True)

# best_box = None
# max_confidence = 0

# # Extract the result from the list (since you're only passing one image)
# if results:
#     result = results[0]
#     for box in result.boxes:
#         confidence = float(box.conf[0])
#         if confidence > max_confidence:
#             max_confidence = confidence
#             best_box = box

# # Crop and save only the face with the highest confidence
# if best_box is not None:
#     x1, y1, x2, y2 = map(int, best_box.xyxy[0])
#     cropped_face = image[y1:y2, x1:x2]

#     # Save cropped face
#     face_filename = os.path.join(output_dir, f"face_best_conf_{max_confidence:.2f}.jpg")
#     cv2.imwrite(face_filename, cropped_face)

#     # Optional: display the cropped face
#     cv2.imshow("Best Face", cropped_face)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("No face detected.")




# # # Multiple Images

# # import cv2
# # import numpy as np
# # from ultralytics import YOLO
# # import os

# # # Load the exported TFLite model
# # tflite_model = YOLO(r"best_float16.tflite")

# # # Input directory containing images
# # input_dir = r"D:\AI\FaceRecognition\test_cases\data\Oriented_face_test\Oriented_face_test"  # Replace with your folder path
# # output_dir =r"D:\AI\FaceRecognition\test_cases\data\cropped_yolov5"
# # os.makedirs(output_dir, exist_ok=True)

# # face_count = 0

# # # Loop through each image in the directory
# # for filename in os.listdir(input_dir):
# #     if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
# #         image_path = os.path.join(input_dir, filename)
# #         image = cv2.imread(image_path)

# #         if image is None:
# #             print(f"Warning: Could not read image {image_path}")
# #             continue

# #         # Run inference
# #         results = tflite_model(image_path)

# #         # Loop through detections
# #         for result in results:
# #             for box in result.boxes:
# #                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
# #                 confidence = float(box.conf[0])  # Confidence score

# #                 # Crop face region
# #                 cropped_face = image[y1:y2, x1:x2]

# #                 # Save cropped face
# #                 base_name = os.path.splitext(filename)[0]
# #                 face_filename = os.path.join(output_dir, f"{base_name}_face_{face_count}_conf_{confidence:.2f}.jpg")
# #                 cv2.imwrite(face_filename, cropped_face)

# #                 # Optional: display cropped face
# #                 # cv2.imshow(f"Face {face_count}", cropped_face)
# #                 # cv2.waitKey(0)
# #                 # cv2.destroyAllWindows()

# #                 face_count += 1

# # print("Cropping complete.")




# # import cv2
# # import os
# # from ultralytics import YOLO

# # # Load TFLite YOLO model
# # tflite_model = YOLO(r"best_float16.tflite")

# # # Input: Images/class_name/image.jpg
# # input_root = r"D:\AI\FaceRecognition\data\test2"

# # # Output: cropped_yolov5/class_name/
# # output_root = r"D:\AI\FaceRecognition\data\test3"
# # os.makedirs(output_root, exist_ok=True)

# # face_count = 0

# # # Loop through class folders
# # for class_name in os.listdir(input_root):
# #     class_input_dir = os.path.join(input_root, class_name)

# #     if not os.path.isdir(class_input_dir):
# #         continue

# #     # Create output class folder
# #     class_output_dir = os.path.join(output_root, class_name)
# #     os.makedirs(class_output_dir, exist_ok=True)

# #     # Loop through images
# #     for filename in os.listdir(class_input_dir):
# #         if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
# #             continue

# #         image_path = os.path.join(class_input_dir, filename)
# #         image = cv2.imread(image_path)

# #         if image is None:
# #             print(f"⚠️ Could not read image: {image_path}")
# #             continue

# #         # Run YOLO inference
# #         results = tflite_model(image_path)

# #         best_box = None
# #         best_conf = 0.0

# #         # Collect the highest confidence face
# #         for result in results:
# #             if result.boxes is None:
# #                 continue

# #             for box in result.boxes:
# #                 conf = float(box.conf[0])
# #                 if conf > best_conf:
# #                     best_conf = conf
# #                     best_box = box

# #         # If no face detected
# #         if best_box is None:
# #             print(f"❌ No face detected in: {image_path}")
# #             continue

# #         # Extract bounding box
# #         x1, y1, x2, y2 = map(int, best_box.xyxy[0])

# #         # Clamp to image boundaries
# #         h, w = image.shape[:2]
# #         x1, y1 = max(0, x1), max(0, y1)
# #         x2, y2 = min(w, x2), min(h, y2)

# #         if x2 <= x1 or y2 <= y1:
# #             print(f"⚠️ Invalid bounding box in: {image_path}")
# #             continue

# #         # Crop face
# #         cropped_face = image[y1:y2, x1:x2]

# #         # Save
# #         base_name = os.path.splitext(filename)[0]
# #         save_path = os.path.join(
# #             class_output_dir,
# #             f"{base_name}_face_conf_{best_conf:.2f}.jpg"
# #         )

# #         cv2.imwrite(save_path, cropped_face)
# #         face_count += 1

# # print(f"✅ Cropping complete. Total faces saved: {face_count}")




import argparse
import os


def select_best_box(boxes):
    """Return the bounding box with the highest confidence."""
    best_box = None
    max_confidence = -1.0

    for box in boxes:
        confidence = float(box.conf[0])

        if confidence > max_confidence:
            max_confidence = confidence
            best_box = box

    return best_box, max_confidence


def main():
    import cv2
    from ultralytics import YOLO
    parser = argparse.ArgumentParser(description="Detect and crop the best face.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model", default="best_float16.tflite", help="Path to YOLO model"
    )
    parser.add_argument("--output", default="cropped_faces", help="Output directory")
    args = parser.parse_args()

    image = cv2.imread(args.image)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    model = YOLO(args.model)
    results = model(image)

    os.makedirs(args.output, exist_ok=True)

    best_box, confidence = select_best_box(results[0].boxes)

    if best_box is None:
        print("No face detected.")
        return

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    cropped_face = image[y1:y2, x1:x2]

    filename = os.path.join(
        args.output, f"face_best_conf_{confidence:.2f}.jpg"
    )

    if not cv2.imwrite(filename, cropped_face):
        raise RuntimeError(f"Could not save cropped image: {filename}")

    print(f"Cropped face saved to: {filename}")


if __name__ == "__main__":
    main()
