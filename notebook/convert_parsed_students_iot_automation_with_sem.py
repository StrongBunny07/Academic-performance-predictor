from __future__ import annotations

import ast
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


GRADE_BASE = {
    "S": 91.0,
    "A+": 86.0,
    "A": 75.0,
    "B": 65.0,
    "C": 55.0,
    "D": 50.0,
}


def parse_list_cell(value) -> list:
    if pd.isna(value) or value == "" or value == "[]":
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def normalize_branch(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    for prefix in ["B.Tech.-", "B.Tech. -", "B.Tech.", "BTECH", "B.E.-", "B.E. -", "B.E."]:
        if text.upper().startswith(prefix.upper()):
            text = text[len(prefix):].lstrip(" -")
            break
    return text.strip()


def parse_sgpa(value) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for item in parse_list_cell(value):
        if not isinstance(item, list) or len(item) < 2:
            continue
        try:
            sem = int(float(item[0]))
            sgpa_val = float(item[1])
        except (TypeError, ValueError):
            continue
        result[sem] = sgpa_val
    return result


def parse_cgpa(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    parsed = parse_list_cell(value)
    if parsed:
        try:
            return float(parsed[0])
        except (TypeError, ValueError):
            pass
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan


def parse_arrears(value) -> Tuple[int, int]:
    if pd.isna(value):
        return 0, 0
    if isinstance(value, (int, float)):
        count = int(value)
        return count, 1 if count > 0 else 0

    items = parse_list_cell(value)
    if not items:
        return 0, 0

    count = len(items)
    has_failure = 0
    for item in items:
        if isinstance(item, list) and item:
            try:
                sem = int(float(item[0]))
            except (TypeError, ValueError):
                continue
            if sem < 5:
                has_failure = 1
                break
    return count, has_failure


def grade_base(grade: str) -> float:
    text = str(grade).strip().upper()
    if text.startswith("A+"):
        return GRADE_BASE["A+"]
    if text.startswith("S"):
        return GRADE_BASE["S"]
    if text.startswith("A"):
        return GRADE_BASE["A"]
    if text.startswith("B"):
        return GRADE_BASE["B"]
    if text.startswith("C"):
        return GRADE_BASE["C"]
    if text.startswith("D"):
        return GRADE_BASE["D"]
    return GRADE_BASE["D"]


def _compute_semester_averages(credits_list: list) -> Tuple[Dict[int, float], Dict[int, float]]:
    internal_scores: Dict[int, List[float]] = {sem: [] for sem in range(1, 6)}
    external_scores: Dict[int, List[float]] = {sem: [] for sem in range(1, 6)}

    for item in credits_list:
        if not isinstance(item, list) or len(item) < 4:
            continue

        try:
            sem = int(float(item[0]))
        except (TypeError, ValueError):
            continue

        if sem not in internal_scores:
            continue

        try:
            internal_mark = float(item[2])
        except (TypeError, ValueError):
            continue

        grade = item[3]
        internal_scores[sem].append(internal_mark)
        external_scores[sem].append(grade_base(grade) - internal_mark)

    internal_avg = {
        sem: (float(np.mean(values)) if values else np.nan)
        for sem, values in internal_scores.items()
    }
    external_avg = {
        sem: (float(np.mean(values)) if values else np.nan)
        for sem, values in external_scores.items()
    }
    return internal_avg, external_avg


def convert_csv_to_backend_xlsx(input_csv: Path, output_xlsx: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    rows = []

    for _, row in df.iterrows():
        credits_list = parse_list_cell(row.get("credits"))
        internal_avg, external_avg = _compute_semester_averages(credits_list)

        arrears_count, has_failure = parse_arrears(row.get("arrears"))
        if arrears_count == 0:
            arrears_count = sum(
                1
                for item in credits_list
                if isinstance(item, list) and len(item) >= 4
                and str(item[3]).strip().upper() == "F"
                and int(float(item[0])) < 5
            )
            has_failure = 1 if arrears_count > 0 else 0

        sgpa_map = parse_sgpa(row.get("sgpa"))
        cgpa_val = parse_cgpa(row.get("cgpa"))

        output_row = {
            "name": row.get("name", ""),
            "regno": str(row.get("regno", "")).strip(),
            "branch": normalize_branch(row.get("department", row.get("branch", ""))),
            "admission_category": str(row.get("admission_category", "")).strip(),
            "hosteller": str(row.get("hosteller", "")).strip(),
            "CGPA": cgpa_val,
            "SGPA_Sem1": sgpa_map.get(1, np.nan),
            "SGPA_Sem2": sgpa_map.get(2, np.nan),
            "SGPA_Sem3": sgpa_map.get(3, np.nan),
            "SGPA_Sem4": sgpa_map.get(4, np.nan),
            "SGPA_Sem5": sgpa_map.get(5, np.nan),
            "Has_Failure_Before_Sem5": has_failure,
            "arrears": arrears_count,
            "Internal_Avg_Sem1": internal_avg.get(1, np.nan),
            "Internal_Avg_Sem2": internal_avg.get(2, np.nan),
            "Internal_Avg_Sem3": internal_avg.get(3, np.nan),
            "Internal_Avg_Sem4": internal_avg.get(4, np.nan),
            "Internal_Avg_Sem5": internal_avg.get(5, np.nan),
            "Ext_Avg_Sem1": external_avg.get(1, np.nan),
            "Ext_Avg_Sem2": external_avg.get(2, np.nan),
            "Ext_Avg_Sem3": external_avg.get(3, np.nan),
            "Ext_Avg_Sem4": external_avg.get(4, np.nan),
            "Ext_Avg_Sem5": external_avg.get(5, np.nan),
            "Absent Percentage (%)": np.nan,
        }
        rows.append(output_row)

    result_df = pd.DataFrame(rows)
    required_columns = [
        "name",
        "regno",
        "branch",
        "admission_category",
        "hosteller",
        "CGPA",
        "SGPA_Sem1",
        "SGPA_Sem2",
        "SGPA_Sem3",
        "SGPA_Sem4",
        "SGPA_Sem5",
        "Has_Failure_Before_Sem5",
        "arrears",
        "Internal_Avg_Sem1",
        "Internal_Avg_Sem2",
        "Internal_Avg_Sem3",
        "Internal_Avg_Sem4",
        "Internal_Avg_Sem5",
        "Ext_Avg_Sem1",
        "Ext_Avg_Sem2",
        "Ext_Avg_Sem3",
        "Ext_Avg_Sem4",
        "Ext_Avg_Sem5",
    ]
    result_df = result_df.dropna(subset=required_columns).copy()
    ordered_columns = [
        "name",
        "regno",
        "branch",
        "admission_category",
        "hosteller",
        "CGPA",
        "SGPA_Sem1",
        "SGPA_Sem2",
        "SGPA_Sem3",
        "SGPA_Sem4",
        "SGPA_Sem5",
        "Has_Failure_Before_Sem5",
        "arrears",
        "Internal_Avg_Sem1",
        "Internal_Avg_Sem2",
        "Internal_Avg_Sem3",
        "Internal_Avg_Sem4",
        "Internal_Avg_Sem5",
        "Ext_Avg_Sem1",
        "Ext_Avg_Sem2",
        "Ext_Avg_Sem3",
        "Ext_Avg_Sem4",
        "Ext_Avg_Sem5",
        "Absent Percentage (%)",
    ]
    result_df = result_df.reindex(columns=ordered_columns)
    result_df.to_excel(output_xlsx, index=False)
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert parsed student CSV into a backend-ready XLSX workbook."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("parsed_students_iot_automation_with_sem.csv"),
        help="Input CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("parsed_students_iot_automation_backend_ready.xlsx"),
        help="Output XLSX file",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    result_df = convert_csv_to_backend_xlsx(args.input, args.output)
    print(f"Wrote {len(result_df)} rows to {args.output}")
    print("Columns:", ", ".join(result_df.columns))


if __name__ == "__main__":
    main()
