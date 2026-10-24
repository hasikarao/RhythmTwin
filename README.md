# RhythmTwin — AI-Powered Heart Rhythm Digital Twin

## 1. Team Details
- Name: Hasika Pallepati
- **Institution:** KLH University
- **Team Name:** RhythmTwin

## 2. Project Title
**RhythmTwin — AI-Powered Heart Rhythm Digital Twin**

## 3. Problem Statement
Electrocardiogram (ECG) signals contain important information about heart rhythm. Analysing these signals can be challenging, and machine learning can help explore patterns associated with different heartbeat categories.

RhythmTwin is a research prototype that uses ECG data and a machine learning model to classify heartbeat patterns and display results through an interactive dashboard.

## 4. Healthcare Use Case
RhythmTwin demonstrates how AI-assisted ECG analysis could support research and educational exploration of heartbeat patterns. The dashboard displays ECG signals, predicted heartbeat categories, and summary visualisations.

**Important:** RhythmTwin is a research prototype, not a medical device. Its predictions must not be used to diagnose conditions, guide treatment, or make emergency decisions.

## 5. Technical Stack
- Python
- Streamlit
- NumPy and Pandas
- Scikit-learn
- WFDB and SciPy
- Plotly
- Joblib

## 6. AI/ML Model Details
RhythmTwin uses a Random Forest machine learning model trained using ECG heartbeat data from the MIT-BIH Arrhythmia Database.

The prototype processes ECG heartbeat features and displays model predictions in the dashboard. Its performance requires further independent validation before any clinical use could be considered.

## 7. System Architecture
The application follows this workflow:

**MIT-BIH ECG Dataset → ECG Data Preparation → Trained Random Forest Model → Heartbeat Classification → Streamlit Dashboard → Visualisation of ECG Waveforms, Prediction Summaries, and Detected Beat Results**

## 8. How to Run
1. Clone this repository:

   ```bash
   git clone https://github.com/hasikarao/RhythmTwin.git
   ```

2. Open the project folder:

   ```bash
   cd RhythmTwin
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   python -m streamlit run app_test.py
   ```

Ensure the required model and ECG data files are available in the locations expected by the application.

## 9. Demo Video
[**Watch the RhythmTwin Project Demonstration**](https://youtu.be/6R_xPPUYrR4)

The video is **Unlisted**. Reviewers can watch it using the link.

## 10. Project Presentation
[**View the RhythmTwin Project Presentation (PowerPoint)**](Rhythm_twin_ppt.pptx)

## 11. Architecture Diagram
[**View the System Architecture Diagram (PDF)**](RhythmTwin_Architecture.pdf)

These resource links work once the corresponding files have been uploaded to the root folder of this repository with the exact filenames shown above.

## 12. Open-Source License
A license has not yet been specified for this repository. External datasets remain subject to their original licenses and terms.

## 13. Future Improvements
- Validate the model using independent ECG records.
- Improve performance across underrepresented heartbeat categories.
- Add clearer model evaluation metrics, including per-class precision, recall, and F1-score.
- Improve the dashboard's interpretability and usability.
- Explore better confidence and uncertainty reporting.

## 14. Responsible Use
RhythmTwin is an experimental research prototype intended for educational and research exploration. It is not a medical device or diagnostic tool and must not be used to make clinical or emergency decisions.
