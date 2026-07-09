# Academic Performance Predictor

Academic Performance Predictor is a Flask-based web application that analyzes student academic data, predicts semester-wise risk levels, and generates a more detailed per-student report. The app accepts a CSV upload, converts it into the internal workbook format used by the model pipeline, runs predictions for semesters 2 to 5, and shows both a full results table and a separate summary view for students in the highest risk band.

## What It Does

- Upload student data as a CSV file.
- Convert the uploaded CSV into the structured format required by the prediction pipeline.
- Predict risk for a selected semester using the pre-trained semester models.
- Show a full table of students with their predicted risk band and probability.
- Open a summary page that lists only the students classified as MAJOR risk.
- Generate a richer per-student report with trend analysis and subject guidance.

## Project Structure

```text
Academic-performance-predictor/
  backend/
    app.py
    model.py
    train_sem4_model.py
    requirements.txt
  fronted/
    index.html
    summary.html
  model/
    sem2_pkl/
    sem3_pkl/
    sem4_pkl/
    sem5_pkl/
  notebook/
    convert_parsed_students_iot_automation_with_sem.py
  resource/
```

The Flask entrypoint is [backend/app.py](backend/app.py). It serves the upload page from the `fronted/` folder, sends the CSV through the converter script in [notebook/convert_parsed_students_iot_automation_with_sem.py](notebook/convert_parsed_students_iot_automation_with_sem.py), and then uses the helpers in [backend/model.py](backend/model.py) for prediction and reporting.

## Requirements

- Python 3.10 or newer is recommended.
- The Python packages listed in [backend/requirements.txt](backend/requirements.txt).
- Pre-trained model artifacts in the semester folders under [model/](model).

## Setup

1. Clone the repository and open the project root.
2. Create and activate a virtual environment.
3. Install the Python dependencies:

```bash
pip install -r backend/requirements.txt
```

4. Start the Flask app from the repository root:

```bash
python backend/app.py
```

5. Open the local address shown in the terminal, then use the upload form on the home page.

## How To Use

### 1. Upload student data

Use the form on [fronted/index.html](fronted/index.html) to upload a CSV file. The app expects the parsed student format produced by this project pipeline, not a raw arbitrary student sheet.

The converter currently reads fields such as `name`, `regno`, `department` or `branch`, `admission_category`, `hosteller`, `credits`, `sgpa`, `cgpa`, and `arrears`, then derives semester averages and other internal features.

### 2. Choose a semester

Select the target semester from the upload page. The backend supports semesters 2, 3, 4, and 5.

### 3. Review predictions

After upload, the app shows:

- the predicted risk probability for each student,
- the risk band label used by the UI,
- a summary count of students in each band,
- and a separate major-risk summary page.

The main result page is rendered from [fronted/index.html](fronted/index.html), and the summary page is rendered from [fronted/summary.html](fronted/summary.html).

### 4. Generate a detailed report

The report endpoint accepts the selected student payload and returns:

- a detailed written report,
- a trend summary,
- and a simple embedded trend visualization.

If the GitHub-hosted model client is configured, the report can be generated through the LLM-backed path. Otherwise, the app falls back to a local report builder.

## Model Flow

The prediction path inside [backend/model.py](backend/model.py) works like this:

1. Load the semester-specific bundle from `model/semX_pkl/`.
2. Engineer the required features from the converted student dataframe.
3. Scale the feature matrix with the stored scaler.
4. Predict risk probability with the stored random forest model.
5. Convert the probability into a HIGH or LOW model-level risk label.

The UI then applies its own display bands for MINIMAL, MEDIUM, and MAJOR risk.

## Training Notes

The repository includes a training script for semester 4 in [backend/train_sem4_model.py](backend/train_sem4_model.py).

- It expects `combined_finals_with_attendance.xlsx` in the repository root.
- It writes semester-4 artifacts to the repository root.
- The runtime loader in [backend/model.py](backend/model.py) reads artifacts from [model/sem4_pkl](model/sem4_pkl).

If you retrain semester 4, copy the generated files into [model/sem4_pkl](model/sem4_pkl) so the app can load them at runtime.

## Environment Variables

The report flow reads these optional environment variables:

- `GITHUB_TOKEN` for the hosted LLM client.
- `LLM_BASE_URL` to override the default inference endpoint.
- `LLM_MODEL` to choose the model name.

If none of these are set, the app still works and uses the local report fallback.

## Notes

- The folder name is `fronted`, not `frontend`.
- The app is intended to be started from the repository root so the relative paths resolve correctly.
- Semester 2 to 5 model folders must already contain the joblib artifacts expected by [backend/model.py](backend/model.py).

## License

No license file is included in the repository snapshot.