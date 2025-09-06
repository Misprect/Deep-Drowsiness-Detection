# Deep Drowsiness Detection  

A real-time AI-powered system to detect driver or user drowsiness using facial and behavioral cues such as prolonged eye closure, yawning, and head pose dynamics.  
The project leverages **computer vision and deep learning** to enhance road safety, workplace productivity, and health monitoring.
this is a collaborative team project

---

## Features  
- **Real-Time Detection** of drowsiness using webcam input.  
- **YOLOv5** for facial detection and landmark localization.  
- **Eye Aspect Ratio (EAR)** & **Mouth Aspect Ratio (MAR)** for blink/yawn detection.  
- **LSTM networks** for temporal sequence learning (reduces false positives).  
- **Custom alert system** with visual, audio, and haptic feedback.  
- Modular design, adaptable for **desktop, embedded devices (Raspberry Pi, Jetson Nano)**, and cloud deployment.  

---

## Tech Stack  
- **Programming Language:** Python  
- **Libraries & Frameworks:** PyTorch, OpenCV, MediaPipe/dlib, Matplotlib  
- **Models:** YOLOv5, CNN, LSTM  
- **Deployment:** Local (desktop/embedded), optional cloud support  

---

## Datasets  
- **Custom dataset** (recorded with webcam under varied conditions)  

---

## How It Works  
1. Capture live video input from webcam.  
2. Detect face and landmarks using **YOLOv5 + MediaPipe/dlib**.  
3. Compute **EAR, MAR, and head pose angles**.  
4. Feed sequential features into **LSTM network** for drowsiness classification.  
5. Trigger **alerts** (visual/audio/haptic) when drowsiness is detected.  

---

## Results  
- Robust detection across varied lighting and user conditions.  
- Temporal analysis with LSTM reduced false positives compared to single-frame methods.  
- Potential applications in **road safety, healthcare, and workplace monitoring**.  

---

## Future Scope  
- Incorporation of **transformer-based models** (Vision Transformers).  
- **Multimodal data fusion** with physiological signals (heart rate, motion).  
- Cloud-based analytics dashboard with federated learning.  

---

## Contributors  
- Aryan Shrivastav  
- Kshitij Pratap Tomer  
- Aaryak Bhargava  
- Aryaman Jain  

---

## License  
This project is developed for academic purposes. For commercial use, please contact the contributors.  
