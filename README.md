# RhythmTwin — AI-Powered Heart Rhythm Digital Twin

## 1. Team Details
- **Submission Type:** Individual Project
- **Institution:** KLH University
- **Team Name:** RhythmTwin

## 2. Project Title
**RhythmTwin — AI-Powered Heart Rhythm Digital Twin**

## 3. Problem Statement
Electrocardiogram (ECG) signals contain important information about heart rhythm. Analysing these signals can be challenging, and machine learning can help explore patterns associated with different heartbeat categories.

RhythmTwin is a research prototype that uses ECG data and a machine learning model to classify heartbeat patterns and display results through an interactive dashboard.

## 4. Healthcare Use Case
RhythmTwin demonstrates how AI-assisted ECG analysis could support research and educational exploration of heartbeat patterns. The dashboard displays ECG signals, predicted heartbeat categories, and summary visualisations.

**Important:** RhythmTwin is a research prototype, not a medical device. Its predictions must not be used to diagnose conditions or guide treatment or emergency decisions.

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
ECG Data → Signal Processing and Feature Preparation → Trained Machine Learning Model → Heartbeat Classification → Streamlit Dashboard → Results Visualisation

An architecture diagram will be added to this repository.

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
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python -m streamlit run app_test.py
   ```

Ensure the required model and ECG data files are available in the locations expected by the application.

## 9. Demo Video
A 15–20 minute project demonstration video will be added here.

## 10. Presentation
The project presentation PDF/PPT will be added here.

## 11. Architecture Diagram
The system architecture diagram PDF/PPT will be added here.

## 12. Open-Source License
A license will be added after selecting an appropriate open-source license.

## 13. Future Improvements
- Validate the model using independent ECG records.
- Improve performance across underrepresented heartbeat categories.
- Add clearer model evaluation metrics.
- Improve the dashboard's interpretability and usability.
