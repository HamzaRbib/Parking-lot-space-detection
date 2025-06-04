# 🚗 Intelligent Parking System: Detection of Free and Occupied Parking Spots

This project explores two deep learning approaches for detecting parking availability in small parking lots using image data. It is a final project by students of Université Hassan II de Casablanca, as part of the **Licence d’excellence en Intelligence Artificielle** program.

## 📌 Project Description

We developed and compared two different methods for detecting free and occupied parking spots:

### 1. YOLOv8-Based Detection

* **Objective:** Detect both vehicles and empty spots directly from full parking lot images.
* **Approach:** Fine-tune a pre-trained YOLOv8 model using annotated bounding boxes for cars and free spaces.
* **Tooling:** Ultralytics library + Roboflow dataset.
* **Outcome:** Good real-time detection capabilities but some confusion with background areas.

### 2. CNN-Based Classification

* **Objective:** Classify individual parking spots as "empty" or "occupied".
* **Approach:** Use black and white mask images to isolate individual parking spaces and train a Convolutional Neural Network (CNN) for binary classification.
* **Tooling:** OpenCV for image processing + custom CNN architecture.
* **Outcome:** Achieved high accuracy of **98.69%** on test data, with minimal overfitting.

---

## 🧠 Technologies Used

* Python
* YOLOv8 (Ultralytics)
* TensorFlow / Keras (for CNN)
* OpenCV (for image segmentation and masking)
* Roboflow (for dataset preparation)

---

## 📊 Results

| Method | Accuracy | Notes                                                            |
| ------ | -------- | ---------------------------------------------------------------- |
| YOLOv8 | \~89–92% | Some confusion with background; good for real-time detection     |
| CNN    | 98.69%   | High precision on classification task with better generalization |

---

## 📁 Dataset

* Source: [Roboflow - Deteksi Parkir Kosong Dataset](https://universe.roboflow.com/skripsijeremy/deteksiparkirkosong/dataset/8)
* Annotation: Bounding boxes for YOLOv8, segmented spots for CNN classification.

---

## 📚 References

* [YOLOv8 Documentation](https://yolov8.org/how-to-use-fine-tune-yolov8/)
* [Roboflow Dataset](https://universe.roboflow.com/skripsijeremy/deteksiparkirkosong/dataset/8)

---

## 👨‍💻 Authors

* **Hamza Rbib**
* **Imrane Taya**
* **Soumia Zahir**

### 🎓 Supervised by:

* **Pr. El-Habib Benlahmar**
* **Pr. Oussama Kaich**

---

## 📌 Conclusion

While YOLOv8 provides a robust object detection pipeline suitable for real-time applications, our CNN-based approach demonstrated superior classification performance for small-scale parking analysis. A hybrid approach could combine the strengths of both models in future iterations.

---

Would you like me to generate a Markdown file version of this so you can directly use or download it?
