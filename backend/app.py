from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile

from flask import Flask, jsonify, render_template, request
import pandas as pd


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _log_llm_env_status() -> None:
    print("LLM env check:")
    print(f"  env file exists: { (BASE_DIR / '.env').exists() }")
    github_set = bool(os.getenv("GITHUB_TOKEN"))
    openai_set = bool(os.getenv("OPENAI_API_KEY"))
    print(f"  GITHUB_TOKEN set: { github_set }")
    print(f"  OPENAI_API_KEY set: { openai_set }")
    print(f"  LLM_BASE_URL: { os.getenv('LLM_BASE_URL') }")
    print(f"  LLM_MODEL: { os.getenv('LLM_MODEL') }")
    if github_set:
        print("  LLM auth source: GITHUB_TOKEN")
    elif openai_set:
        print("  LLM auth source: OPENAI_API_KEY")
    else:
        print("  LLM auth source: none")

try:
    from model import (
        MODEL_RISK_THRESHOLD,
        build_student_from_form,
        build_students_from_xlsx,
        build_trend_artifacts,
        generate_combined_report,
        predict_risk_batch,
    )
except ImportError:
    from backend.model import (
        MODEL_RISK_THRESHOLD,
        build_student_from_form,
        build_students_from_xlsx,
        build_trend_artifacts,
        generate_combined_report,
        predict_risk_batch,
    )

BASE_DIR = Path(__file__).resolve().parent
_load_env_file(BASE_DIR / ".env")
_log_llm_env_status()
FRONTEND_DIR = BASE_DIR.parent / "fronted"
CONVERTER_SCRIPT = BASE_DIR.parent / "notebook" / "convert_parsed_students_iot_automation_with_sem.py"

app = Flask(__name__, template_folder=str(FRONTEND_DIR))
LATEST_ANALYSIS = {
    "filename": None,
    "summary": None,
    "major_students": [],
}


def _num(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_selected_semester(value, default: int = 4) -> int:
    try:
        sem = int(value)
    except (TypeError, ValueError):
        return default
    return sem if sem in {2, 3, 4, 5} else default


def _load_csv_converter():
    spec = importlib.util.spec_from_file_location("csv_to_xlsx_converter", CONVERTER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load converter script: {CONVERTER_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    convert_fn = getattr(module, "convert_csv_to_backend_xlsx", None)
    if convert_fn is None:
        raise RuntimeError("Converter script does not expose convert_csv_to_backend_xlsx")
    return convert_fn


def _build_students_from_uploaded_csv(upload) -> tuple[pd.DataFrame, list[str]]:
    convert_csv_to_backend_xlsx = _load_csv_converter()

    with tempfile.TemporaryDirectory(prefix="student-risk-upload-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        csv_path = temp_dir_path / "uploaded_students.csv"
        xlsx_path = temp_dir_path / "converted_students.xlsx"

        upload.save(csv_path)
        convert_csv_to_backend_xlsx(csv_path, xlsx_path)
        return build_students_from_xlsx(xlsx_path)


def _build_major_risk_summary(major_students: list[dict], filename: str | None) -> list[str]:
    if not major_students:
        source = filename or "the latest analyzed file"
        return [f"No students were classified in the MAJOR risk band for {source}."]

    total = len(major_students)
    avg_probability = sum(student["risk_probability_value"] for student in major_students) / total
    with_arrears = sum(1 for student in major_students if (student.get("arrears") or 0) > 0)
    hostellers = sum(
        1
        for student in major_students
        if str(student.get("hosteller", "")).strip().lower() == "hosteler"
    )
    branches = sorted({student.get("branch", "").strip() for student in major_students if student.get("branch", "").strip()})

    lines = [
        f"{total} student(s) from {filename or 'the latest analyzed upload'} are currently in the MAJOR risk band.",
        f"The average risk probability across this group is {avg_probability:.2%}.",
        f"{with_arrears} student(s) in this group have arrears recorded.",
        f"{hostellers} student(s) in this group are marked as hostellers.",
    ]
    if branches:
        lines.append(f"Branches represented: {', '.join(branches)}.")
    return lines


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    warnings = []
    error = None
    summary = None
    filename = None
    selected_semester = 4

    if request.method == "POST":
        upload = request.files.get("csv_file")
        selected_semester = _parse_selected_semester(request.form.get("selected_semester"), default=4)
        if upload is None or upload.filename == "":
            error = "Please upload a CSV file to generate predictions."
        elif Path(upload.filename).suffix.lower() != ".csv":
            error = "Only CSV uploads are supported."
        else:
            filename = upload.filename
            try:
                student_df, warnings = _build_students_from_uploaded_csv(upload)
            except Exception as exc:
                error = f"CSV conversion failed: {exc}"
            else:
                if student_df.empty:
                    error = "No valid student records found in the uploaded CSV file."
                else:
                    probs, levels = predict_risk_batch(student_df, selected_semester)
                    for idx, row in student_df.iterrows():
                        prob = float(probs[idx])
                        if prob <= 0.80:
                            band = "MINIMAL"
                            band_class = "risk-minimal"
                        elif prob >= 0.90:
                            band = "MAJOR"
                            band_class = "risk-major"
                        else:
                            band = "MEDIUM"
                            band_class = "risk-medium"

                        results.append(
                            {
                                "regno": row.get("regno", ""),
                                "branch": row.get("branch", ""),
                                "admission_category": row.get("admission_category", ""),
                                "hosteller": row.get("hosteller", ""),
                                "cgpa": row.get("CGPA"),
                                "sgpa_sem1": row.get("SGPA_Sem1"),
                                "sgpa_sem2": row.get("SGPA_Sem2"),
                                "sgpa_sem3": row.get("SGPA_Sem3"),
                                "sgpa_sem4": row.get("SGPA_Sem4"),
                                "sgpa_sem5": row.get("SGPA_Sem5"),
                                "has_failure_before_sem5": row.get("Has_Failure_Before_Sem5", 0),
                                "internal_avg_sem1": row.get("Internal_Avg_Sem1"),
                                "internal_avg_sem2": row.get("Internal_Avg_Sem2"),
                                "internal_avg_sem3": row.get("Internal_Avg_Sem3"),
                                "internal_avg_sem4": row.get("Internal_Avg_Sem4"),
                                "internal_avg_sem5": row.get("Internal_Avg_Sem5"),
                                "external_avg_sem1": row.get("Ext_Avg_Sem1"),
                                "external_avg_sem2": row.get("Ext_Avg_Sem2"),
                                "external_avg_sem3": row.get("Ext_Avg_Sem3"),
                                "external_avg_sem4": row.get("Ext_Avg_Sem4"),
                                "external_avg_sem5": row.get("Ext_Avg_Sem5"),
                                "absent_percentage": row.get("Absent Percentage (%)"),
                                "selected_semester": selected_semester,
                                "arrears": row.get("arrears", 0),
                                "risk_probability": f"{prob:.2%}",
                                "risk_level": levels[idx],
                                "risk_band": band,
                                "risk_class": band_class,
                                "payload": {
                                    "regno": row.get("regno", ""),
                                    "branch": row.get("branch", ""),
                                    "admission_category": row.get("admission_category", ""),
                                    "hosteller": row.get("hosteller", ""),
                                    "CGPA": _num(row.get("CGPA")),
                                    "SGPA_Sem1": _num(row.get("SGPA_Sem1")),
                                    "SGPA_Sem2": _num(row.get("SGPA_Sem2")),
                                    "SGPA_Sem3": _num(row.get("SGPA_Sem3")),
                                    "SGPA_Sem4": _num(row.get("SGPA_Sem4")),
                                    "SGPA_Sem5": _num(row.get("SGPA_Sem5")),
                                    "Has_Failure_Before_Sem5": _num(row.get("Has_Failure_Before_Sem5", 0)) or 0,
                                    "arrears": _num(row.get("arrears", 0)) or 0,
                                    "Internal_Avg_Sem1": _num(row.get("Internal_Avg_Sem1")),
                                    "Internal_Avg_Sem2": _num(row.get("Internal_Avg_Sem2")),
                                    "Internal_Avg_Sem3": _num(row.get("Internal_Avg_Sem3")),
                                    "Internal_Avg_Sem4": _num(row.get("Internal_Avg_Sem4")),
                                    "Internal_Avg_Sem5": _num(row.get("Internal_Avg_Sem5")),
                                    "Ext_Avg_Sem1": _num(row.get("Ext_Avg_Sem1")),
                                    "Ext_Avg_Sem2": _num(row.get("Ext_Avg_Sem2")),
                                    "Ext_Avg_Sem3": _num(row.get("Ext_Avg_Sem3")),
                                    "Ext_Avg_Sem4": _num(row.get("Ext_Avg_Sem4")),
                                    "Ext_Avg_Sem5": _num(row.get("Ext_Avg_Sem5")),
                                    "Absent Percentage (%)": _num(row.get("Absent Percentage (%)")),
                                    "selected_semester": selected_semester,
                                },
                            }
                        )

                    total = len(results)
                    summary = {
                        "total": total,
                        "major": sum(1 for r in results if r["risk_band"] == "MAJOR"),
                        "medium": sum(1 for r in results if r["risk_band"] == "MEDIUM"),
                        "minimal": sum(1 for r in results if r["risk_band"] == "MINIMAL"),
                    }
                    major_students = [
                        {
                            "regno": row["regno"],
                            "branch": row["branch"],
                            "hosteller": row["hosteller"],
                            "cgpa": row["cgpa"],
                            "arrears": row["arrears"],
                            "risk_probability": row["risk_probability"],
                            "risk_probability_value": float(row["risk_probability"].strip("%")) / 100.0,
                        }
                        for row in results
                        if row["risk_band"] == "MAJOR"
                    ]
                    LATEST_ANALYSIS["filename"] = filename
                    LATEST_ANALYSIS["summary"] = summary
                    LATEST_ANALYSIS["major_students"] = major_students

    return render_template(
        "index.html",
        results=results,
        warnings=warnings,
        error=error,
        summary=summary,
        filename=filename,
        has_summary=bool(LATEST_ANALYSIS["summary"]),
    )


@app.route("/summary", methods=["GET"])
def summary_page():
    latest_summary = LATEST_ANALYSIS["summary"]
    filename = LATEST_ANALYSIS["filename"]
    major_students = LATEST_ANALYSIS["major_students"]
    summary_lines = _build_major_risk_summary(major_students, filename)

    return render_template(
        "summary.html",
        filename=filename,
        summary=latest_summary,
        major_students=major_students,
        summary_lines=summary_lines,
        has_analysis=bool(latest_summary),
    )


@app.route("/report", methods=["POST"])
def report():
    payload = request.get_json(silent=True) or {}
    if not payload:
        return jsonify({"error": "Missing student data."}), 400

    selected_semester = _parse_selected_semester(payload.get("selected_semester"), default=4)
    print(
        f"[report] /report request regno={payload.get('regno', '')} "
        f"selected_semester={selected_semester}"
    )
    student_df = build_student_from_form(payload)
    result = generate_combined_report(student_df, selected_semester)
    trend = build_trend_artifacts(student_df, selected_semester)

    return jsonify(
        {
            "report": result.report_text,
            "trend_summary": trend["summary_text"],
            "trend_plot": trend["plot_png_base64"],
            "trend_semesters": trend["semesters"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
