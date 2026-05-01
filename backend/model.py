from __future__ import annotations

import base64
from dataclasses import dataclass
import ast
import io
import hashlib
import os
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import json
import re
import pdfplumber
from openai import OpenAI
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_TARGET_SEMESTER = int(os.getenv("MODEL_TARGET_SEMESTER", "4"))
MODEL_RISK_THRESHOLD = float(os.getenv("MODEL_RISK_THRESHOLD", "0.33"))


@dataclass
class SemesterModelBundle:
    semester: int
    model: object
    imputer: object
    scaler: object
    feature_names: List[str]
    base_columns: List[str]
    kmeans: object


_SEMESTER_BUNDLES: Dict[int, SemesterModelBundle] = {}
_REPORT_CACHE: Optional[Dict[str, str]] = None
_REPORT_CACHE_VERSION = "2026-04-17-llm-v2"


def load_semester_bundle(target_semester: int) -> SemesterModelBundle:
    if target_semester in _SEMESTER_BUNDLES:
        return _SEMESTER_BUNDLES[target_semester]

    folder = MODEL_DIR / f"sem{target_semester}_pkl"
    bundle = SemesterModelBundle(
        semester=target_semester,
        model=joblib.load(folder / f"risk_rf_model_sem{target_semester}.pkl"),
        imputer=joblib.load(folder / f"imputer_sem{target_semester}.pkl"),
        scaler=joblib.load(folder / f"scaler_sem{target_semester}.pkl"),
        feature_names=joblib.load(folder / f"feature_names_sem{target_semester}.pkl"),
        base_columns=joblib.load(folder / f"base_columns_sem{target_semester}.pkl"),
        kmeans=joblib.load(folder / f"kmeans_model_sem{target_semester}.pkl"),
    )
    _SEMESTER_BUNDLES[target_semester] = bundle
    return bundle


def _coerce_float(value) -> float:
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _report_cache_path() -> Path:
    return BASE_DIR / ".report_cache.json"


def _load_report_cache() -> Dict[str, str]:
    global _REPORT_CACHE
    if _REPORT_CACHE is not None:
        return _REPORT_CACHE
    cache_path = _report_cache_path()
    if cache_path.exists():
        try:
            _REPORT_CACHE = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(_REPORT_CACHE, dict):
                _REPORT_CACHE = {}
        except Exception:
            _REPORT_CACHE = {}
    else:
        _REPORT_CACHE = {}
    return _REPORT_CACHE


def _save_report_cache() -> None:
    if _REPORT_CACHE is None:
        return
    try:
        _report_cache_path().write_text(json.dumps(_REPORT_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _make_report_cache_key(student_raw_df: pd.DataFrame, target_semester: int) -> str:
    row = student_raw_df.iloc[0].to_dict()
    selected_fields = {
        "version": _REPORT_CACHE_VERSION,
        "model": os.getenv("LLM_MODEL", "gpt-4o"),
        "regno": str(row.get("regno", "")).strip(),
        "semester": target_semester,
        "arrears": row.get("arrears", 0),
        "cgpa": row.get("CGPA"),
        "internal": [row.get(f"Internal_Avg_Sem{i}") for i in range(1, 6)],
        "external": [row.get(f"Ext_Avg_Sem{i}") for i in range(1, 6)],
        "sgpa": [row.get(f"SGPA_Sem{i}") for i in range(1, 6)],
    }
    payload = json.dumps(selected_fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_trend_artifacts(student_raw_df: pd.DataFrame, target_semester: int) -> Dict[str, object]:
    source_row = student_raw_df.iloc[0]

    completed_semester = max(target_semester - 1, 1)
    semesters = list(range(1, completed_semester + 1))
    internal_values: List[float] = []
    external_values: List[float] = []
    for sem in semesters:
        internal_values.append(_coerce_float(source_row.get(f"Internal_Avg_Sem{sem}")))
        external_values.append(_coerce_float(source_row.get(f"Ext_Avg_Sem{sem}")))

    internal_valid = [v for v in internal_values if pd.notna(v)]
    external_valid = [v for v in external_values if pd.notna(v)]

    summary_lines: List[str] = []
    if internal_valid and external_valid:
        summary_lines.append(
            "Trend chart shows internal and external averages across completed semesters."
        )
    else:
        summary_lines.append("Insufficient trend data available.")

    def _fmt_num(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:.1f}"

    def _bar_rects(values: List[float], x0: float, y0: float, width: float, height: float,
                   vmin: float, vmax: float, offset: float, bar_width: float) -> List[Tuple[float, float, float, float]]:
        rects: List[Tuple[float, float, float, float]] = []
        count = max(len(semesters), 1)
        group_width = width / count
        span = vmax - vmin if vmax > vmin else 1.0
        baseline = y0 + height - ((0 - vmin) / span) * height
        baseline = min(max(baseline, y0), y0 + height)
        for idx, val in enumerate(values):
            if pd.isna(val):
                continue
            center_x = x0 + (idx * group_width) + (group_width / 2)
            bar_x = center_x + offset - (bar_width / 2)
            value_y = y0 + height - ((float(val) - vmin) / span) * height
            top_y = min(value_y, baseline)
            bar_height = max(abs(baseline - value_y), 1.0)
            rects.append((bar_x, top_y, bar_width, bar_height))
        return rects

    all_vals = [v for v in internal_values + external_values if pd.notna(v)]
    y_min = min(all_vals) if all_vals else 0.0
    y_max = max(all_vals) if all_vals else 100.0
    y_pad = max((y_max - y_min) * 0.1, 5.0)
    y_min -= y_pad
    y_max += y_pad
    chart_x = 80
    chart_y = 90
    chart_width = 1040
    chart_height = 260
    group_width = chart_width / max(len(semesters), 1)
    single_bar_width = min(28.0, max(group_width * 0.22, 12.0))
    internal_bars = _bar_rects(
        internal_values, chart_x, chart_y, chart_width, chart_height, y_min, y_max, -single_bar_width * 0.65, single_bar_width
    )
    external_bars = _bar_rects(
        external_values, chart_x, chart_y, chart_width, chart_height, y_min, y_max, single_bar_width * 0.65, single_bar_width
    )

    def _bars(rects: List[Tuple[float, float, float, float]], color: str) -> str:
        return "".join(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{color}" />'
            for x, y, w, h in rects
        )

    def _grid_lines(x0: float, y0: float, width: float, height: float, step_count: int = 5) -> str:
        lines = []
        for i in range(step_count + 1):
            y = y0 + (height / step_count) * i
            lines.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1" />')
        return "".join(lines)

    def _y_labels(x_pos: float, y0: float, height: float, vmin: float, vmax: float, step_count: int = 5) -> str:
        labels = []
        for i in range(step_count + 1):
            val = vmax - (vmax - vmin) * (i / step_count)
            y = y0 + (height / step_count) * i + 4
            labels.append(f'<text x="{x_pos}" y="{y:.1f}" font-size="11" fill="#6b7280" text-anchor="end">{val:.0f}</text>')
        return "".join(labels)

    def _x_labels(sem_x0: float, sem_width: float, count: int, y_pos: float) -> str:
        labels = []
        for idx, sem in enumerate(semesters):
            x = sem_x0 + ((idx + 0.5) * (sem_width / max(count, 1)))
            labels.append(f'<text x="{x:.1f}" y="{y_pos}" font-size="11" fill="#6b7280" text-anchor="middle">Sem {sem}</text>')
        return "".join(labels)

    svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" width="1200" height="420" viewBox="0 0 1200 420" role="img" aria-label="Semester trend analysis">
            <rect width="1200" height="420" fill="#ffffff" rx="18" />
      <text x="40" y="32" font-size="18" font-weight="700" fill="#1f2937">Semester Trend Analysis</text>
      <text x="40" y="52" font-size="12" fill="#6b7280">Internal and external averages up to Semester {target_semester}</text>

            <text x="80" y="76" font-size="14" font-weight="600" fill="#374151">Bar Chart</text>
            <rect x="80" y="90" width="1040" height="260" fill="#f9fafb" stroke="#d1d5db" />
            {_grid_lines(80, 90, 1040, 260)}
            <line x1="80" y1="350" x2="1120" y2="350" stroke="#9ca3af" />
            <line x1="80" y1="90" x2="80" y2="350" stroke="#9ca3af" />
            {_y_labels(70, 90, 260, y_min, y_max)}
            {_x_labels(80, 1040, len(semesters), 375)}
      {_bars(internal_bars, "#0e5b5b")}
      {_bars(external_bars, "#b12a2a")}
    </svg>
    """
    plot_b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")

    return {
        "semesters": semesters,
        "internal_values": internal_values,
        "external_values": external_values,
        "summary_text": "\n".join(summary_lines),
        "plot_png_base64": plot_b64,
    }

# -------------------------------
# Static mappings and data for report
# -------------------------------
static_prerequisite_map = {
    "ENGINEERING MATHEMATICS - II": ["ENGINEERING MATHEMATICS - I"],
    "OBJECT ORIENTED PROGRAMMING IN C++": ["PROBLEM SOLVING & PROGRAMMING IN C"],
    "ENGINEERING CHEMISTRY": ["ENGINEERING PHYSICS"],
    "BASIC ELECTRONICS ENGINEERING": ["BASIC ELECTRICAL ENGINEERING"],
    "BASIC MECHANICAL ENGINEERING": ["ENGINEERING MECHANICS"],
    "ENGINEERING GRAPHICS": ["ENGINEERING MECHANICS"],
    "BIOLOGY FOR ENGINEERS": [],
    "ENGINEERING MATHEMATICS - III": ["ENGINEERING MATHEMATICS - II"],
    "JAVA PROGRAMMING": ["OBJECT ORIENTED PROGRAMMING IN C++"],
    "COMPUTER ORGANIZATION": ["DIGITAL SYSTEM DESIGN"],
    "DATA STRUCTURES": ["PROBLEM SOLVING & PROGRAMMING IN C", "OBJECT ORIENTED PROGRAMMING IN C++"],
    "DIGITAL SYSTEM DESIGN": ["BASIC ELECTRONICS ENGINEERING"],
    "DATA STRUCTURES LABORATORY": ["DATA STRUCTURES"],
    "DIGITAL SYSTEM DESIGN LABORATORY": ["DIGITAL SYSTEM DESIGN"],
    "DISCRETE STRUCTURES": ["ENGINEERING MATHEMATICS - III"],
    "DESIGN & ANALYSIS OF ALGORITHMS": ["DATA STRUCTURES", "DISCRETE STRUCTURES", "ENGINEERING MATHEMATICS - III"],
    "FUNDAMENTALS OF DATABASE MANAGEMENT SYSTEMS": ["DATA STRUCTURES"],
    "COMPUTER ARCHITECTURE": ["COMPUTER ORGANIZATION", "DIGITAL SYSTEM DESIGN"],
    "COMPUTER SYSTEM DESIGN LABORATORY": ["DIGITAL SYSTEM DESIGN", "COMPUTER ARCHITECTURE"],
    "DESIGN & ANALYSIS OF ALGORITHMS LABORATORY": ["DESIGN & ANALYSIS OF ALGORITHMS", "DATA STRUCTURES"],
    "COMPUTER GRAPHICS USING OPENGL": ["ENGINEERING MATHEMATICS - III", "DATA STRUCTURES"],
    "OBJECT ORIENTED ANALYSIS & DESIGN": ["OBJECT ORIENTED PROGRAMMING IN C++"],
    "MATHEMATICS FOR CYBER SECURITY": ["ENGINEERING MATHEMATICS - III"],
    "THEORY OF COMPUTATION": ["DISCRETE STRUCTURES", "ENGINEERING MATHEMATICS – III"],
    "OPERATING SYSTEMS": ["COMPUTER ORGANIZATION", "DATA STRUCTURES"],
    "COMPUTER NETWORKS": ["COMPUTER ORGANIZATION"],
    "SOFT SKILLS - I": [],
    "STATISTICAL FOUNDATIONS FOR COMPUTER SCIENCE": ["ENGINEERING MATHEMATICS – III"],
    "NETWORK TOOLS & TECHNIQUES": ["COMPUTER NETWORKS"],
    "ARTIFICIAL INTELLIGENCE": ["DATA STRUCTURES", "DISCRETE STRUCTURES"],
    "PYTHON PROGRAMMING WITH WEB FRAMEWORKS": ["PROBLEM SOLVING & PROGRAMMING IN C"],
    "LINUX PROGRAMMING": [],
    "DIGITAL IMAGE PROCESSING": ["ENGINEERING MATHEMATICS – III", "DATA STRUCTURES"],
    "SYSTEM SOFTWARE": ["COMPUTER ORGANIZATION"],
    "SYSTEM MODELLING & SIMULATION": ["ENGINEERING MATHEMATICS – III"],
    "SCRIPT PROGRAMMING": ["PROBLEM SOLVING & PROGRAMMING IN C"],
}

LOCAL_TOPIC_HINTS = {
    "ENGINEERING MATHEMATICS - III": [
        "Matrices and systems of linear equations",
        "Vector calculus and partial derivatives",
        "Laplace transforms",
        "Fourier series",
        "Complex differentiation and integration",
    ],
    "COMPUTER ORGANIZATION": [
        "Number systems and binary arithmetic",
        "Instruction set architecture",
        "CPU datapath and control unit",
        "Pipelining and hazards",
        "Memory hierarchy and cache",
    ],
    "DATA STRUCTURES": [
        "Arrays, linked lists and stacks",
        "Queues and circular queues",
        "Trees and tree traversals",
        "Hashing techniques",
        "Graphs and graph algorithms",
    ],
    "DISCRETE STRUCTURES": [
        "Logic and propositions",
        "Sets, relations and functions",
        "Counting principles and combinatorics",
        "Graph theory basics",
        "Trees and recurrence relations",
    ],
    "PROBLEM SOLVING & PROGRAMMING IN C": [
        "C syntax and control structures",
        "Functions and modular programming",
        "Arrays and strings",
        "Pointers and memory handling",
        "File handling and debugging",
    ],
    "ENGINEERING MATHEMATICS - II": [
        "Ordinary differential equations",
        "Matrices and determinants",
        "Sequences and series",
        "Complex numbers and basic transforms",
        "Applications in engineering problems",
    ],
    "OBJECT ORIENTED PROGRAMMING IN C++": [
        "Classes and objects",
        "Inheritance and polymorphism",
        "Templates and exception handling",
        "STL containers and iterators",
        "File handling in C++",
    ],
    "OBJECT ORIENTED ANALYSIS & DESIGN": [
        "Object-oriented principles and modeling",
        "Use case, class and sequence diagrams",
        "CRC cards and interaction diagrams",
        "Design patterns basics",
        "UML notation and documentation",
    ],
    "ENGINEERING CHEMISTRY": [
        "Electrochemistry",
        "Corrosion and its prevention",
        "Water treatment",
        "Polymer chemistry",
        "Fuel cells and batteries",
    ],
    "BASIC ELECTRONICS ENGINEERING": [
        "Semiconductor devices",
        "Diodes and transistor operation",
        "Amplifiers and logic gates",
        "Operational amplifiers",
        "Digital electronics basics",
    ],
    "BASIC MECHANICAL ENGINEERING": [
        "Thermodynamics basics",
        "Heat transfer fundamentals",
        "Engineering materials",
        "Manufacturing processes",
        "IC engines and power systems",
    ],
    "ENGINEERING GRAPHICS": [
        "Orthographic projections",
        "Isometric views",
        "Sectional views",
        "Dimensioning and tolerances",
        "Projection of solids",
    ],
    "BIOLOGY FOR ENGINEERS": [
        "Cell structure and function",
        "Genetics and inheritance",
        "Enzymes and metabolism",
        "Biomolecules",
        "Basic biotechnology applications",
    ],
    "ENGINEERING PHYSICS": [
        "Optics and wave motion",
        "Quantum mechanics basics",
        "Semiconductor physics",
        "Material properties",
        "Laws of thermodynamics",
    ],
    "BASIC CIVIL ENGINEERING": [
        "Surveying basics",
        "Building materials",
        "Structural elements",
        "Water resources and sanitation",
        "Roads and transportation basics",
    ],
    "BASIC ELECTRICAL ENGINEERING": [
        "Circuit laws and analysis",
        "AC and DC machines",
        "Transformers",
        "Electrical measurements",
        "Wiring and safety",
    ],
    "ENGINEERING MECHANICS": [
        "Force systems and moments",
        "Equilibrium of rigid bodies",
        "Friction",
        "Centroid and moment of inertia",
        "Work, energy and power",
    ],
    "TECHNICAL COMMUNICATION": [
        "Grammar and sentence structure",
        "Technical writing",
        "Listening and speaking skills",
        "Presentation skills",
        "Email and report writing",
    ],
    "JAVA PROGRAMMING": [
        "Java syntax and control flow",
        "Classes, objects and inheritance",
        "Interfaces and packages",
        "Exception handling",
        "Collections and multithreading",
    ],
    "DIGITAL SYSTEM DESIGN": [
        "Boolean algebra and logic simplification",
        "Combinational circuits",
        "Sequential circuits",
        "Flip-flops and counters",
        "Finite state machines",
    ],
    "DIGITAL SYSTEM DESIGN LABORATORY": [
        "Logic gate implementation",
        "Combinational circuit design",
        "Sequential circuit design",
        "Timing analysis",
        "Hardware debugging",
    ],
    "DATA STRUCTURES LABORATORY": [
        "Array and list implementations",
        "Stacks and queues",
        "Trees and traversals",
        "Sorting and searching",
        "Graphs and hashing experiments",
    ],
    "DESIGN & ANALYSIS OF ALGORITHMS": [
        "Asymptotic analysis",
        "Divide and conquer",
        "Greedy algorithms",
        "Dynamic programming",
        "Graph algorithms",
    ],
    "FUNDAMENTALS OF DATABASE MANAGEMENT SYSTEMS": [
        "Relational model and ER diagrams",
        "SQL queries and joins",
        "Normalization",
        "Transaction management",
        "Indexing and query processing",
    ],
    "COMPUTER ARCHITECTURE": [
        "Instruction set architecture",
        "Pipelining and hazards",
        "Memory hierarchy",
        "CPU design basics",
        "I/O organization",
    ],
    "COMPUTER SYSTEM DESIGN LABORATORY": [
        "Digital logic simulation",
        "Hardware description basics",
        "Circuit implementation",
        "Timing and verification",
        "System-level debugging",
    ],
    "DESIGN & ANALYSIS OF ALGORITHMS LABORATORY": [
        "Algorithm implementation",
        "Complexity measurement",
        "Sorting and searching experiments",
        "Graph algorithm coding",
        "Dynamic programming practice",
    ],
    "COMPUTER GRAPHICS USING OPENGL": [
        "Graphics pipeline and rendering stages",
        "2D and 3D transformations",
        "Projection and viewing",
        "Lighting, shading and color models",
        "OpenGL primitives and rasterization",
    ],
    "MATHEMATICS FOR CYBER SECURITY": [
        "Number theory and modular arithmetic",
        "Congruence relations and modular inverses",
        "Finite fields and polynomials",
        "Probability and combinatorics in security",
        "Cryptographic mathematics basics",
    ],
    "COMPUTER NETWORKS LABORATORY": [
        "Socket programming and client-server basics",
        "IP addressing and subnetting practice",
        "Routing and connectivity experiments",
        "Packet capture and analysis",
        "Network configuration and diagnostics",
    ],
    "OPERATING SYSTEMS LABORATORY": [
        "Process creation and scheduling experiments",
        "Shell commands and scripting",
        "File system and permissions practice",
        "Synchronization and deadlock demonstrations",
        "Memory management exercises",
    ],
    "THEORY OF COMPUTATION": [
        "Finite automata and regular languages",
        "Regular expressions",
        "Context-free grammars",
        "Pushdown automata",
        "Turing machines and decidability",
    ],
    "OPERATING SYSTEMS": [
        "Process states and scheduling",
        "Threads and synchronization",
        "Deadlocks and resource allocation",
        "Memory management and paging",
        "File systems and disk management",
    ],
    "COMPUTER NETWORKS": [
        "OSI and TCP/IP models",
        "Physical and data link layer concepts",
        "Routing and switching",
        "Transport layer protocols",
        "Congestion control and flow control",
    ],
    "SOFT SKILLS - I": [
        "Communication fundamentals",
        "Group discussion techniques",
        "Presentation skills",
        "Interview preparation",
        "Professional writing",
    ],
    "STATISTICAL FOUNDATIONS FOR COMPUTER SCIENCE": [
        "Probability basics",
        "Random variables and distributions",
        "Estimation and hypothesis testing",
        "Correlation and regression",
        "Statistical inference",
    ],
    "NETWORK TOOLS & TECHNIQUES": [
        "Network diagnostics tools",
        "Subnetting and addressing",
        "Packet analysis and monitoring",
        "Routing and connectivity troubleshooting",
        "Security and performance tools",
    ],
    "ARTIFICIAL INTELLIGENCE": [
        "Problem solving by search",
        "Knowledge representation",
        "Inference and reasoning",
        "Planning and decision making",
        "Machine learning foundations",
    ],
    "PYTHON PROGRAMMING WITH WEB FRAMEWORKS": [
        "Python syntax and data structures",
        "Functions, modules and OOP",
        "Web routing and request handling",
        "Templates and forms",
        "REST APIs and framework basics",
    ],
    "LINUX PROGRAMMING": [
        "Shell commands and scripting",
        "File permissions and processes",
        "Pipes, signals and IPC",
        "Text processing utilities",
        "Automation with bash",
    ],
    "DIGITAL IMAGE PROCESSING": [
        "Image fundamentals and representation",
        "Filtering and enhancement",
        "Transforms and frequency domain",
        "Segmentation and thresholding",
        "Morphological operations",
    ],
    "SYSTEM SOFTWARE": [
        "Assemblers and macro processors",
        "Loaders and linkers",
        "Compilation phases",
        "Memory allocation schemes",
        "Program translation basics",
    ],
    "SYSTEM MODELLING & SIMULATION": [
        "System modeling concepts",
        "Discrete event simulation",
        "Random number generation",
        "Queuing models",
        "Validation and verification",
    ],
    "SCRIPT PROGRAMMING": [
        "Shell scripting basics",
        "Regular expressions",
        "awk and sed usage",
        "File and text processing",
        "Automation and scripting workflows",
    ],
}

DEFAULT_SEMESTER_COURSES = {
    2: [
        "ENGINEERING MATHEMATICS - II",
        "OBJECT ORIENTED PROGRAMMING IN C++",
        "ENGINEERING CHEMISTRY",
        "BASIC ELECTRONICS ENGINEERING",
        "BASIC MECHANICAL ENGINEERING",
        "ENGINEERING GRAPHICS",
        "BIOLOGY FOR ENGINEERS",
    ],
    3: [
        "ENGINEERING MATHEMATICS - III",
        "JAVA PROGRAMMING",
        "COMPUTER ORGANIZATION",
        "DATA STRUCTURES",
        "DIGITAL SYSTEM DESIGN",
        "DATA STRUCTURES LABORATORY",
        "DIGITAL SYSTEM DESIGN LABORATORY",
    ],
    4: [
        "DISCRETE STRUCTURES",
        "DESIGN & ANALYSIS OF ALGORITHMS",
        "FUNDAMENTALS OF DATABASE MANAGEMENT SYSTEMS",
        "COMPUTER ARCHITECTURE",
        "COMPUTER SYSTEM DESIGN LABORATORY",
        "DESIGN & ANALYSIS OF ALGORITHMS LABORATORY",
    ],
    5: [
        "THEORY OF COMPUTATION",
        "OPERATING SYSTEMS",
        "COMPUTER NETWORKS",
        "SOFT SKILLS - I",
        "STATISTICAL FOUNDATIONS FOR COMPUTER SCIENCE",
        "NETWORK TOOLS & TECHNIQUES",
        "ARTIFICIAL INTELLIGENCE",
        "PYTHON PROGRAMMING WITH WEB FRAMEWORKS",
        "LINUX PROGRAMMING",
        "DIGITAL IMAGE PROCESSING",
        "SYSTEM SOFTWARE",
        "SYSTEM MODELLING & SIMULATION",
        "SCRIPT PROGRAMMING",
    ],
}

grade_to_point = {"S": 10, "A+": 9, "A": 8, "B": 7, "C": 6, "D": 5, "E": 4, "F": 0}

_SEMESTER_FILE = BASE_DIR / "semester_wise_subjects_final.xlsx"
if _SEMESTER_FILE.exists():
    df_semester = pd.read_excel(_SEMESTER_FILE)
else:
    df_semester = pd.DataFrame()

SYLLABUS_PDF = BASE_DIR / "resource" / "B.Tech. CSE 180 - Scheme and Syllabus - 2021-22 onwards.pdf"
if not SYLLABUS_PDF.exists():
    alt_syllabus_pdf = BASE_DIR / "B.Tech. CSE 180 - Scheme and Syllabus - 2021-22 onwards.pdf"
    if alt_syllabus_pdf.exists():
        SYLLABUS_PDF = alt_syllabus_pdf
_SYLLABUS_TEXT: str | None = None
_TOPICS_CACHE: Dict[str, List[str]] = {}


# -------------------------------
# Helpers for report analysis
# -------------------------------

def normalize_course_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name).upper().strip()
    name = name.replace("–", "-").replace("—", "-")
    name = " ".join(name.split())
    return name


def compute_grade_point(grade_str: str) -> int:
    if pd.isna(grade_str):
        return 0
    return grade_to_point.get(str(grade_str).strip().upper(), 0)


def parse_semester_subjects(sem_str: str) -> Dict[str, int]:
    if pd.isna(sem_str) or sem_str == "":
        return {}
    items = sem_str.split(";")
    res: Dict[str, int] = {}
    for item in items:
        if ":" in item:
            course, grade = item.split(":", 1)
            course = normalize_course_name(course)
            res[course] = compute_grade_point(grade.strip())
    return res


def parse_semester_5_courses(sem5_str: str) -> List[str]:
    if pd.isna(sem5_str) or sem5_str == "":
        return []
    items = sem5_str.split(";")
    res = []
    for item in items:
        if ":" in item:
            course = item.split(":", 1)[0].strip()
        else:
            course = item.strip()
        res.append(normalize_course_name(course))
    return res


def get_student_data(
    regno: str, target_semester: int
) -> Tuple[Dict[str, int], Dict[str, int], float, Dict[int, List[Tuple[str, int]]], List[str]] | Tuple[None, None, None, None, None]:
    if df_semester.empty:
        return None, None, None, None, None

    student_row = df_semester[df_semester["regno"].astype(str) == str(regno)]
    if student_row.empty:
        return None, None, None, None, None

    previous_semesters = max(1, target_semester - 1)
    grade_dict: Dict[str, int] = {}
    sem_dict: Dict[str, int] = {}
    all_points: List[int] = []
    semester_grades: Dict[int, List[Tuple[str, int]]] = {sem: [] for sem in range(1, previous_semesters + 1)}

    row = student_row.iloc[0]
    for sem in range(1, previous_semesters + 1):
        col = f"Semester_{sem}_Subjects"
        if col in student_row.columns:
            data = parse_semester_subjects(row.get(col, ""))
            for course, pt in data.items():
                grade_dict[course] = pt
                sem_dict[course] = sem
                all_points.append(pt)
                semester_grades[sem].append((course, pt))

    avg_point = float(np.mean(all_points)) if all_points else 0.0
    target_col = f"Semester_{target_semester}_Subjects"
    target_courses = parse_semester_5_courses(row.get(target_col, "")) if target_col in student_row.columns else []

    return grade_dict, sem_dict, avg_point, semester_grades, target_courses


def _load_syllabus_text() -> str:
    global _SYLLABUS_TEXT
    if _SYLLABUS_TEXT is not None:
        return _SYLLABUS_TEXT

    if not SYLLABUS_PDF.exists():
        _SYLLABUS_TEXT = ""
        return _SYLLABUS_TEXT

    syllabus_text = ""
    try:
        with pdfplumber.open(SYLLABUS_PDF) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    syllabus_text += text + "\n"
    except Exception:
        _SYLLABUS_TEXT = ""
        return _SYLLABUS_TEXT

    max_chars = 15000
    if len(syllabus_text) > max_chars:
        syllabus_text = syllabus_text[:max_chars]

    _SYLLABUS_TEXT = syllabus_text
    return _SYLLABUS_TEXT


def _parse_topics_json(output: str) -> List[str]:
    output = output.strip()
    try:
        data = json.loads(output)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", output, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
        except json.JSONDecodeError:
            return []
    return []


def _sanitize_report_text(report_text: str) -> str:
    lines = report_text.splitlines()
    kept_lines: List[str] = []
    separator_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped and set(stripped) == {"="} and len(stripped) >= 10:
            separator_count += 1
            kept_lines.append(line)
            if separator_count >= 2:
                break
            continue
        kept_lines.append(line)

    report_text = "\n".join(kept_lines).strip()

    unwanted_markers = [
        "📚 ALL SUBJECTS AND GRADE POINTS",
    ]
    cut_positions = [report_text.find(marker) for marker in unwanted_markers if marker in report_text]
    cut_positions = [pos for pos in cut_positions if pos > 0]
    if cut_positions:
        report_text = report_text[: min(cut_positions)].rstrip()
    return report_text.strip()


def _build_performance_trend_section(trend_artifacts: Dict[str, object]) -> str:
    semesters = trend_artifacts.get("semesters", []) or []
    internal_values = trend_artifacts.get("internal_values", []) or []
    external_values = trend_artifacts.get("external_values", []) or []

    if not semesters or not internal_values or not external_values:
        return "📊 PERFORMANCE TREND (Semester-wise Averages)\nInsufficient data available."

    def _is_valid(value: float | None) -> bool:
        if value is None:
            return False
        try:
            return not pd.isna(value)
        except Exception:
            return True

    def _fmt_value(value: float | None, width: int = 15) -> str:
        if not _is_valid(value):
            return f"{'N/A':<{width}}"
        return f"{float(value):<{width}.2f}"

    def _fmt_delta(value: float | None, width: int = 12) -> str:
        if value is None:
            return f"{'N/A':<{width}}"
        return f"{float(value):<+{width}.2f}"

    def _first_last(values: List[float]) -> Tuple[float | None, float | None]:
        valid = [v for v in values if _is_valid(v)]
        if not valid:
            return None, None
        return valid[0], valid[-1]

    lines: List[str] = []
    lines.append("📊 PERFORMANCE TREND (Semester-wise Averages)")
    lines.append("📌 Note: External marks are out of 100 (percentage).")
    lines.append("-" * 60)
    lines.append(f"{'Semester':<10} {'Internal Avg':<15} {'External Avg':<15} {'Δ Internal':<12} {'Δ External':<12}")
    lines.append("-" * 60)

    prev_int: float | None = None
    prev_ext: float | None = None
    for idx, sem in enumerate(semesters):
        int_val = internal_values[idx] if idx < len(internal_values) else None
        ext_val = external_values[idx] if idx < len(external_values) else None

        if idx == 0 and _is_valid(int_val):
            delta_int = 0.0
        elif _is_valid(int_val) and _is_valid(prev_int):
            delta_int = float(int_val) - float(prev_int)
        else:
            delta_int = None

        if idx == 0 and _is_valid(ext_val):
            delta_ext = 0.0
        elif _is_valid(ext_val) and _is_valid(prev_ext):
            delta_ext = float(ext_val) - float(prev_ext)
        else:
            delta_ext = None

        line = (
            f"Semester {sem:<5} "
            f"{_fmt_value(int_val)} "
            f"{_fmt_value(ext_val)} "
            f"{_fmt_delta(delta_int)} "
            f"{_fmt_delta(delta_ext)}"
        )
        lines.append(line)

        if _is_valid(int_val):
            prev_int = float(int_val)
        if _is_valid(ext_val):
            prev_ext = float(ext_val)

    lines.append("-" * 60)

    first_int, last_int = _first_last(internal_values)
    first_ext, last_ext = _first_last(external_values)
    if first_int is None or last_int is None or first_ext is None or last_ext is None:
        recommendation = "Not enough trend data to generate a recommendation."
    else:
        internal_improve = last_int - first_int
        external_improve = last_ext - first_ext
        if internal_improve > 0 and external_improve > 0:
            recommendation = "✅ Both internal and external marks improved. Keep up the good work!"
        elif internal_improve > 0 and external_improve <= 0:
            recommendation = (
                "⚠️ Internal marks improved, but external marks did not. Focus on exam preparation "
                "(timed practice, past papers)."
            )
        elif internal_improve <= 0 and external_improve > 0:
            recommendation = (
                "⚠️ External marks improved, but internal consistency dropped. Strengthen coursework "
                "and assignments."
            )
        else:
            recommendation = "⚠️ Both internal and external marks declined. Seek academic support and form study groups."

    lines.append(f"\n💡 KEY RECOMMENDATION: {recommendation}")
    return "\n".join(lines)


def _insert_trend_section(report_text: str, trend_section: str) -> str:
    if not trend_section or "PERFORMANCE TREND" in report_text:
        return report_text

    lines = report_text.splitlines()
    sep_indexes = [
        idx for idx, line in enumerate(lines)
        if line.strip() and set(line.strip()) == {"="} and len(line.strip()) >= 10
    ]
    if not sep_indexes:
        return report_text.rstrip() + "\n\n" + trend_section

    insert_at = sep_indexes[-1]
    updated = lines[:insert_at] + ["", trend_section] + lines[insert_at:]
    return "\n".join(updated).rstrip()


def get_topics_from_llm(subject: str, client: "OpenAI") -> List[str]:
    subject_norm = normalize_course_name(subject)
    if subject_norm in _TOPICS_CACHE:
        return _TOPICS_CACHE[subject_norm]

    syllabus_text = _load_syllabus_text()
    if not syllabus_text:
        _TOPICS_CACHE[subject_norm] = []
        return []

    prompt = (
        "You are a syllabus expert. For the course "
        f"\"{subject}\", list the main topics (3-8 key topics) that are essential "
        "for understanding the subject. Output ONLY a JSON list of strings. "
        "Do not include any extra text. Example: [\"Topic1\", \"Topic2\", ...]\n\n"
        "Syllabus text (for reference):\n"
        f"{syllabus_text}\n"
    )

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            temperature=0.3,
            max_tokens=300,
        )
        output = response.choices[0].message.content or ""
        topics = _parse_topics_json(output)
        if not topics:
            topics = LOCAL_TOPIC_HINTS.get(subject_norm, [])
        _TOPICS_CACHE[subject_norm] = topics
        return topics
    except Exception:
        topics = LOCAL_TOPIC_HINTS.get(subject_norm, [])
        _TOPICS_CACHE[subject_norm] = topics
        return topics


# -------------------------------
# Feature engineering (matches notebook logic)
# -------------------------------

def _normalize_hosteller(value) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float, np.integer, np.floating, bool)):
        return 1 if float(value) > 0 else 0
    text = str(value).strip().lower()
    return 1 if text.startswith("hostel") else 0


def _select_semester_columns(student_df: pd.DataFrame, target_semester: int) -> pd.DataFrame:
    student_df = student_df.copy()

    if "Has_Failure_Before_Sem5" in student_df.columns and "arrears" not in student_df.columns:
        student_df["arrears"] = student_df["Has_Failure_Before_Sem5"]

    if "arrears" not in student_df.columns:
        student_df["arrears"] = 0

    if "hosteller" not in student_df.columns:
        student_df["hosteller"] = 0
    student_df["hosteller"] = student_df["hosteller"].apply(_normalize_hosteller)

    if "Absent Percentage (%)" not in student_df.columns:
        student_df["Absent Percentage (%)"] = np.nan
    else:
        student_df["Absent Percentage (%)"] = pd.to_numeric(student_df["Absent Percentage (%)"], errors="coerce")

    for sem in range(1, 6):
        internal_col = f"Internal_Avg_Sem{sem}"
        external_col = f"Ext_Avg_Sem{sem}"
        if internal_col not in student_df.columns:
            student_df[internal_col] = np.nan
        if external_col not in student_df.columns:
            student_df[external_col] = np.nan
        student_df[internal_col] = pd.to_numeric(student_df[internal_col], errors="coerce")
        student_df[external_col] = pd.to_numeric(student_df[external_col], errors="coerce")

    # Treat the selected semester as the current one and only keep earlier semesters as inputs.
    for sem in range(target_semester, 6):
        internal_col = f"Internal_Avg_Sem{sem}"
        external_col = f"Ext_Avg_Sem{sem}"
        if internal_col in student_df.columns:
            student_df[internal_col] = np.nan
        if external_col in student_df.columns:
            student_df[external_col] = np.nan

    return student_df


def _completed_semester_average(student_df: pd.DataFrame, target_semester: int, prefix: str) -> float:
    values: List[float] = []
    for sem in range(1, target_semester):
        col = f"{prefix}_Sem{sem}"
        if col not in student_df.columns:
            continue
        value = pd.to_numeric(student_df.iloc[0].get(col), errors="coerce")
        if pd.notna(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def _build_base_frame(student_df: pd.DataFrame, bundle: SemesterModelBundle, target_semester: int) -> pd.DataFrame:
    student_df = _select_semester_columns(student_df, target_semester)
    frame = student_df.copy()

    frame = pd.get_dummies(frame, columns=["admission_category", "branch"], drop_first=True)
    imputer_cols = list(bundle.imputer.feature_names_in_)
    raw_frame = pd.DataFrame(index=frame.index)

    for col in imputer_cols:
        if col in frame.columns:
            raw_frame[col] = frame[col]
        else:
            raw_frame[col] = 0 if (col.startswith("admission_category_") or col.startswith("branch_")) else np.nan

    raw_frame = raw_frame.reindex(columns=imputer_cols)
    imputed = bundle.imputer.transform(raw_frame)
    X_base = pd.DataFrame(imputed, columns=imputer_cols, index=frame.index)
    return X_base


def engineer_features(student_df: pd.DataFrame, bundle: SemesterModelBundle, target_semester: int) -> pd.DataFrame:
    X_base = _build_base_frame(student_df, bundle, target_semester)

    for sem in range(2, target_semester + 1):
        X_base[f"Internal_trend_Sem{sem}"] = (
            X_base[f"Internal_Avg_Sem{sem}"] - X_base[f"Internal_Avg_Sem{sem - 1}"]
        )

    for sem in range(2, target_semester):
        X_base[f"Ext_trend_Sem{sem}"] = X_base[f"Ext_Avg_Sem{sem}"] - X_base[f"Ext_Avg_Sem{sem - 1}"]

    for sem in range(1, target_semester):
        X_base[f"Int_Ext_gap_Sem{sem}"] = X_base[f"Internal_Avg_Sem{sem}"] - X_base[f"Ext_Avg_Sem{sem}"]

    for sem in range(3, target_semester + 1):
        X_base[f"Internal_roll2_Sem{sem}"] = (
            X_base[f"Internal_Avg_Sem{sem - 1}"] + X_base[f"Internal_Avg_Sem{sem}"]
        ) / 2
        X_base[f"Internal_std_Sem{sem - 1}_{sem}"] = X_base[
            [f"Internal_Avg_Sem{sem - 1}", f"Internal_Avg_Sem{sem}"]
        ].std(axis=1)

    X_base["Weighted_Internal"] = sum(
        X_base[f"Internal_Avg_Sem{sem}"] for sem in range(1, target_semester + 1)
    ) / target_semester

    if target_semester > 1:
        X_base["Weighted_External"] = sum(
            X_base[f"Ext_Avg_Sem{sem}"] for sem in range(1, target_semester)
        ) / (target_semester - 1)
    else:
        X_base["Weighted_External"] = np.nan

    for sem in range(1, target_semester):
        X_base[f"Int_Ext_ratio_Sem{sem}"] = X_base[f"Internal_Avg_Sem{sem}"] / (
            X_base[f"Ext_Avg_Sem{sem}"] + 1e-5
        )

    X_base["arrears_x_Weighted_Internal"] = X_base["arrears"] * X_base["Weighted_Internal"]
    X_base["arrears_x_Absent"] = X_base["arrears"] * X_base["Absent Percentage (%)"]

    bins = [0, 5, 10, 15, 20, 25, 100]
    labels = [f"absent_{i}_{j}" for i, j in zip(bins[:-1], bins[1:])]
    X_base["Absent_Bin"] = pd.cut(X_base["Absent Percentage (%)"], bins=bins, labels=labels)
    absent_dummies = pd.get_dummies(X_base["Absent_Bin"], prefix="absent_bin")
    X_base = pd.concat([X_base, absent_dummies], axis=1)
    for col in absent_dummies.columns:
        X_base[f"{col}_x_Weighted_Internal"] = X_base[col] * X_base["Weighted_Internal"]
        if "Weighted_External" in X_base.columns:
            X_base[f"{col}_x_Weighted_External"] = X_base[col] * X_base["Weighted_External"]
    X_base = X_base.drop(columns=["Absent_Bin"])

    traj_cols = [f"Internal_Avg_Sem{i}" for i in range(1, target_semester + 1)]
    X_base["trajectory_cluster"] = bundle.kmeans.predict(X_base[traj_cols])

    return X_base.reindex(columns=bundle.feature_names, fill_value=0)


# -------------------------------
# Public API
# -------------------------------

@dataclass
class ReportResult:
    risk_probability: float
    risk_level: str
    report_text: str


DEFAULT_STUDENT = {
    "Has_Failure_Before_Sem5": 0,
    "Internal_Avg_Sem1": 42.5,
    "Internal_Avg_Sem2": 40.2,
    "Internal_Avg_Sem3": 35.1,
    "Internal_Avg_Sem4": 33.8,
    "Internal_Avg_Sem5": 31.5,
    "Ext_Avg_Sem1": 65.0,
    "Ext_Avg_Sem2": 58.0,
    "Ext_Avg_Sem3": 52.0,
    "Ext_Avg_Sem4": 48.0,
    "Ext_Avg_Sem5": 45.0,
    "hosteller": "Day Scholar",
    "admission_category": "SASTRA-General",
    "branch": "Computer Science & Engineering",
    "regno": "127003194",
    "CGPA": 5.5,
    "SGPA_Sem1": 6.0,
    "SGPA_Sem2": 5.8,
    "SGPA_Sem3": 5.5,
    "SGPA_Sem4": 5.2,
    "SGPA_Sem5": 5.0,
    "arrears": 0,
    "Absent Percentage (%)": 0.0,
}


def predict_risk(student_raw_df: pd.DataFrame, target_semester: int) -> Tuple[float, str]:
    bundle = load_semester_bundle(target_semester)
    X_student = engineer_features(student_raw_df, bundle, target_semester)
    X_student_scaled = bundle.scaler.transform(X_student)
    prob_risk = float(bundle.model.predict_proba(X_student_scaled)[0, 1])
    risk_level = "HIGH" if prob_risk >= MODEL_RISK_THRESHOLD else "LOW"
    return prob_risk, risk_level


def predict_risk_batch(student_raw_df: pd.DataFrame, target_semester: int) -> Tuple[np.ndarray, List[str]]:
    bundle = load_semester_bundle(target_semester)
    X_student = engineer_features(student_raw_df, bundle, target_semester)
    X_student_scaled = bundle.scaler.transform(X_student)
    probs = bundle.model.predict_proba(X_student_scaled)[:, 1]
    levels = ["HIGH" if p >= MODEL_RISK_THRESHOLD else "LOW" for p in probs]
    return probs, levels


def generate_report(student_raw_df: pd.DataFrame, target_semester: int) -> ReportResult:
    prob_risk, risk_level = predict_risk(student_raw_df, target_semester)

    regno = student_raw_df.iloc[0].get("regno", "")
    arrears_raw = student_raw_df.iloc[0].get("arrears", 0)
    try:
        arrears_value = pd.to_numeric(arrears_raw, errors="coerce")
        arrears = 0 if pd.isna(arrears_value) else int(arrears_value)
    except Exception:
        arrears = 0
    grade_dict, sem_dict, avg_point, semester_grades, target_courses = get_student_data(regno, target_semester)
    avg_internal = _completed_semester_average(student_raw_df, target_semester, "Internal_Avg")
    avg_external = _completed_semester_average(student_raw_df, target_semester, "Ext_Avg")

    lines: List[str] = []
    lines.append("ACADEMIC REVIEW")
    lines.append("=" * 70)
    lines.append(f"Model Risk Probability: {prob_risk:.2%} ({risk_level} risk)")
    lines.append(f"Average Internal Marks (Sem 1-{max(target_semester - 1, 1)}): {avg_internal:.2f}")
    lines.append(f"Average External Marks (Sem 1-{max(target_semester - 1, 1)}): {avg_external:.2f}")
    lines.append(f"Arrears: {arrears}")

    if grade_dict is None:
        return ReportResult(
            prob_risk,
            risk_level,
            _build_generic_semester_report(prob_risk, risk_level, target_semester, avg_internal, avg_external),
        )

    weak_subjects = [(course, pt) for course, pt in grade_dict.items() if pt < 6]
    lines.append(f"Overall Average Grade Point (Sem 1-{max(target_semester - 1, 1)}): {avg_point:.2f}")

    if weak_subjects:
        lines.append("\nWeak Subjects (Grade D/E/F):")
        for course, pt in sorted(weak_subjects):
            lines.append(f"  - {course} (point {pt})")
    elif arrears > 0:
        lines.append("\nNo weak subjects (below C grade) found in previous semesters.")
        lines.append("Arrears were recorded, so the student should still be reviewed carefully.")
    else:
        lines.append("\nNo weak subjects (below C grade) found in previous semesters.")

    lines.append("\nRecommendations:")
    if risk_level == "HIGH":
        lines.append("  - Focus on foundational gaps and revise weak subjects first.")
        lines.append("  - Practice past papers and target improving internal scores.")
        lines.append("  - Seek help early for prerequisite-heavy courses.")
    else:
        lines.append("  - Keep revision steady and monitor internal scores regularly.")
        lines.append("  - Continue strengthening prerequisites for the next semester.")

    return ReportResult(prob_risk, risk_level, "\n".join(lines))


def _get_llm_client() -> Optional["OpenAI"]:
    api_key = os.getenv("GITHUB_TOKEN")
    if not api_key:
        return None

    base_url = os.getenv("LLM_BASE_URL") or "https://models.inference.ai.azure.com"
    return OpenAI(base_url=base_url, api_key=api_key)


def generate_combined_report(student_raw_df: pd.DataFrame, target_semester: int) -> ReportResult:
    prob_risk, risk_level = predict_risk(student_raw_df, target_semester)
    regno = student_raw_df.iloc[0].get("regno", "")
    arrears_raw = student_raw_df.iloc[0].get("arrears", 0)
    try:
        arrears_value = pd.to_numeric(arrears_raw, errors="coerce")
        arrears = 0 if pd.isna(arrears_value) else int(arrears_value)
    except Exception:
        arrears = 0
    avg_point = _coerce_float(student_raw_df.iloc[0].get("CGPA"))
    target_courses = DEFAULT_SEMESTER_COURSES.get(target_semester, [])
    avg_internal = _completed_semester_average(student_raw_df, target_semester, "Internal_Avg")
    avg_external = _completed_semester_average(student_raw_df, target_semester, "Ext_Avg")
    trend_artifacts = build_trend_artifacts(student_raw_df, target_semester)
    client = _get_llm_client()
    model_name = os.getenv("LLM_MODEL", "gpt-4o")
    cache_key = _make_report_cache_key(student_raw_df, target_semester)
    report_cache = _load_report_cache()
    cached_report = report_cache.get(cache_key)

    print(
        f"[report] generate_combined_report regno={regno} semester={target_semester} "
        f"client={'present' if client is not None else 'missing'} model={model_name}"
    )
    if cached_report:
        print(
            f"[report] cache hit for regno={regno}, semester={target_semester} "
            f"(chars={len(cached_report)})"
        )
        trend_section = _build_performance_trend_section(trend_artifacts)
        merged_report = _insert_trend_section(cached_report, trend_section)
        return ReportResult(prob_risk, risk_level, merged_report)

    course_topics: Dict[str, List[str]] = {
        normalize_course_name(course): LOCAL_TOPIC_HINTS.get(normalize_course_name(course), [])
        for course in target_courses
    }
    subject_wise_analysis = [
        {"course": course, "course_topics": course_topics.get(normalize_course_name(course), [])}
        for course in target_courses
    ]

    data_payload = {
        "regno": regno,
        "semester": target_semester,
        "risk_probability": round(prob_risk, 4),
        "risk_level": risk_level,
        "avg_point": None if pd.isna(avg_point) else round(float(avg_point), 2),
        "average_internal_marks": None if pd.isna(avg_internal) else round(avg_internal, 2),
        "average_external_marks": None if pd.isna(avg_external) else round(avg_external, 2),
        "arrears": arrears,
        "trend_summary": trend_artifacts["summary_text"],
        "trend_semesters": trend_artifacts["semesters"],
        "trend_internal_values": trend_artifacts["internal_values"],
        "trend_external_values": trend_artifacts["external_values"],
        "weak_subjects": [],
        "prerequisite_weaknesses": [],
        "target_courses": target_courses,
        "subject_wise_analysis": subject_wise_analysis,
        "semester_grades": {},
        "all_prereqs_overview": {},
        "prereq_topics": {},
        "source": "uploaded workbook",
    }

    if client is None:
        print("[report] LLM client missing, using local fallback report builder")
        report_text = _build_detailed_report_from_data(data_payload)
        trend_section = _build_performance_trend_section(trend_artifacts)
        report_text = _insert_trend_section(report_text, trend_section)
        return ReportResult(prob_risk, risk_level, report_text)

    system_msg = (
        "You are an academic advisor. Use ONLY the facts provided in DATA. "
        "Do not invent courses, marks, topics, or outcomes. "
        "If data is missing, explicitly say 'Not available'. "
        "Follow the required output format exactly and do not add extra sections. "
        "Do not wrap the response in code fences."
    )
    user_msg = (
        "DATA (JSON):\n"
        f"{data_payload}\n\n"
        "OUTPUT FORMAT (use exactly these headings and separators):\n"
        "ACADEMIC REVIEW\n"
        "======================================================================\n"
        "📊 Model Risk Probability: <percent> (<risk level> risk)\n"
        "📚 SUBJECT-WISE DETAILED ANALYSIS:\n"
        "For each course in the selected semester, include only:\n"
        "• course name\n"
        "• important topics to learn for that course\n"
        "Do not include prerequisite subjects in this section.\n"
        "\n"
        "📌 RECOMMENDATIONS:\n"
        "• <rec 1>\n"
        "• <rec 2>\n"
        "• <rec 3>\n"
        "======================================================================\n"
        "\n"
        "RULES:\n"
        "- Use only DATA.\n"
        "- If a list is empty, use the specified 'Not available' text.\n"
        "- If arrears > 0, mention them explicitly and do not ignore them.\n"
        "- Do not omit the subject-wise detailed analysis block.\n"
        "- Use the provided course topics from DATA.\n"
        "- Keep the structure and symbols exactly as shown.\n"
        "- Do not add any extra commentary."
    )

    try:
        print(f"[report] Calling LLM for regno={regno}, semester={target_semester}")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=1400,
        )
        report_text = (response.choices[0].message.content or "").strip()
        if report_text.startswith("```"):
            report_text = report_text.strip("`").strip()
        if not report_text:
            raise ValueError("Empty LLM response")
        print(f"[report] LLM response received (chars={len(report_text)})")
        report_text = _sanitize_report_text(report_text)
        trend_section = _build_performance_trend_section(trend_artifacts)
        report_text = _insert_trend_section(report_text, trend_section)
        report_cache[cache_key] = report_text
        _save_report_cache()
        print(f"[report] cached LLM response for regno={regno}, semester={target_semester}")
        return ReportResult(prob_risk, risk_level, report_text)
    except Exception as exc:
        print(f"[report] LLM call failed: {exc!r}")
        traceback.print_exc()
        if cached_report:
            print(f"[report] returning cached report after LLM failure for regno={regno}, semester={target_semester}")
            return ReportResult(prob_risk, risk_level, cached_report)
        print("[report] LLM call failed, using local fallback report builder")
        report_text = _build_detailed_report_from_data(data_payload)
        trend_section = _build_performance_trend_section(trend_artifacts)
        report_text = _insert_trend_section(report_text, trend_section)
        return ReportResult(prob_risk, risk_level, report_text)


def _build_detailed_report_from_data(data: Dict[str, object]) -> str:
    risk_prob = data.get("risk_probability", 0.0)
    risk_level = data.get("risk_level", "LOW")
    semester = int(data.get("semester", 5) or 5)
    avg_point = data.get("avg_point", 0.0)
    avg_internal = data.get("average_internal_marks")
    avg_external = data.get("average_external_marks")
    arrears_raw = data.get("arrears", 0)
    try:
        arrears_value = pd.to_numeric(arrears_raw, errors="coerce")
        arrears = 0 if pd.isna(arrears_value) else int(arrears_value)
    except Exception:
        arrears = 0
    weak_subjects = data.get("weak_subjects", []) or []
    prereq_weaknesses = data.get("prerequisite_weaknesses", []) or []
    semester_grades = data.get("semester_grades", {}) or {}
    all_prereqs = data.get("all_prereqs_overview", {}) or {}
    prereq_topics = data.get("prereq_topics", {}) or {}
    subject_wise_analysis = data.get("subject_wise_analysis", []) or []
    target_courses = data.get("target_courses", []) or data.get("semester_5_courses", []) or []
    if not target_courses:
        target_courses = DEFAULT_SEMESTER_COURSES.get(semester, [])

    def _fmt(value) -> str:
        if value is None:
            return "Not available"
        try:
            if pd.isna(value):
                return "Not available"
        except Exception:
            pass
        return str(value)

    lines: List[str] = []
    lines.append("ACADEMIC REVIEW")
    lines.append("=" * 70)
    lines.append(f"📊 Model Risk Probability: {risk_prob:.2%} ({risk_level} risk)")
    if avg_internal is not None:
        lines.append(f"📈 Average Internal Marks: {_fmt(avg_internal)}")
    if avg_external is not None:
        lines.append(f"📉 Average External Marks: {_fmt(avg_external)}")
    lines.append(f"Arrears: {arrears}")
    if avg_point is not None:
        try:
            if not pd.isna(avg_point):
                lines.append(f"📈 Overall Average Grade Point (Sem 1-{max(semester - 1, 1)}): {_fmt(avg_point)}")
        except Exception:
            lines.append(f"📈 Overall Average Grade Point (Sem 1-{max(semester - 1, 1)}): {_fmt(avg_point)}")
    lines.append("")
    if weak_subjects:
        lines.append("⚠️ WEAK SUBJECTS (Grade D/E/F):")
        for item in weak_subjects:
            lines.append(f"   • {item['course']} (point {item['point']})")
    elif arrears > 0:
        lines.append("No weak subjects (below C grade) found in previous semesters.")
        lines.append("Arrears were recorded, so the student should still be reviewed carefully.")
    else:
        lines.append("No weak subjects (below C grade) found in previous semesters.")
    lines.append("")
    lines.append(f"🔸 PREREQUISITE WEAKNESSES AFFECTING SEMESTER {semester}:")
    if prereq_weaknesses:
        for item in prereq_weaknesses:
            sem_label = "?" if item["semester"] == -1 else item["semester"]
            lines.append(
                f"• {item['course']} requires {item['prereq']} (Sem {sem_label}, point {item['point']}) – below average/weak."
            )
    else:
        lines.append("None.")
    lines.append("")
    lines.append("📌 RECOMMENDATIONS:")
    lines.append("• Review weak subjects and prerequisite topics before focusing on current courses.")
    lines.append("• Seek extra help or tutoring for foundational gaps.")
    lines.append("• Use past exam papers and mock tests to improve.")
    if subject_wise_analysis:
        lines.append("")
        lines.append("📚 SUBJECT-WISE DETAILED ANALYSIS:")
        for item in subject_wise_analysis:
            course = item.get("course", "Unknown subject")
            course_topics = item.get("course_topics", []) or []
            lines.append(f"• {course}")
            if course_topics:
                lines.append("  Important topics to learn:")
                for topic in course_topics:
                    lines.append(f"    - {topic}")
            else:
                lines.append("  Important topics to learn: Not available")
    elif target_courses:
        lines.append("")
        lines.append("📚 SUBJECT-WISE DETAILED ANALYSIS:")
        for course in target_courses:
            lines.append(f"• {course}")
            lines.append("  Important topics to learn: Not available")
    lines.append("=" * 70)
    return "\n".join(lines)


def _build_generic_semester_report(
    prob_risk: float,
    risk_level: str,
    target_semester: int,
    avg_internal: float,
    avg_external: float,
    weak_subjects: List[Tuple[str, int]] | None = None,
) -> str:
    lines: List[str] = []
    lines.append("ACADEMIC REVIEW")
    lines.append("=" * 70)
    lines.append(f"Model Risk Probability: {prob_risk:.2%} ({risk_level} risk)")
    lines.append(f"Average Internal Marks (Sem 1-{max(target_semester - 1, 1)}): {avg_internal:.2f}")
    lines.append(f"Average External Marks (Sem 1-{max(target_semester - 1, 1)}): {avg_external:.2f}")
    lines.append("")
    if weak_subjects:
        lines.append("Weak Subjects (Grade D/E/F):")
        for course, pt in sorted(weak_subjects):
            lines.append(f"  - {course} (point {pt})")
    else:
        lines.append("Weak subjects: Not available from the uploaded aggregate workbook.")
    lines.append("")
    lines.append("Recommendations:")
    lines.append("  - Focus on the lowest-scoring completed semesters first.")
    lines.append("  - Keep attendance and internal marks steady in the next semester.")
    lines.append("  - Review core concepts tied to the completed subjects.")
    return "\n".join(lines)


def build_student_from_form(form: Dict[str, str]) -> pd.DataFrame:
    data = DEFAULT_STUDENT.copy()

    for key in data.keys():
        if key in form and form[key] != "":
            data[key] = form[key]

    numeric_fields = {
        "Has_Failure_Before_Sem5",
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
        "CGPA",
        "SGPA_Sem1",
        "SGPA_Sem2",
        "SGPA_Sem3",
        "SGPA_Sem4",
        "SGPA_Sem5",
        "arrears",
    }

    for field in numeric_fields:
        if field in data:
            try:
                data[field] = float(data[field])
            except (TypeError, ValueError):
                data[field] = np.nan

    # Make sure arrears exists for feature engineering
    if "arrears" not in data:
        data["arrears"] = 0

    return pd.DataFrame([data])


# -------------------------------
# XLSX ingestion helpers
# -------------------------------

def _parse_list_cell(value) -> List:
    if pd.isna(value) or value == "" or value == "[]":
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _normalize_credit_course(name: str) -> str:
    return str(name).strip().upper().replace("-", "–")


COURSE_MAPPING = {
    # ---------- Semester I ----------
    _normalize_credit_course("PROBLEM SOLVING & PROGRAMMING IN C"): (1, 4),
    _normalize_credit_course("ENGINEERING MATHEMATICS - I"): (1, 4),
    _normalize_credit_course("ENGINEERING PHYSICS"): (1, 4),
    _normalize_credit_course("BASIC ELECTRICAL ENGINEERING"): (1, 3),
    _normalize_credit_course("BASIC CIVIL ENGINEERING"): (1, 3),
    _normalize_credit_course("ENGINEERING MECHANICS"): (1, 3),
    _normalize_credit_course("INTRODUCTION TO ENGINEERING DESIGN"): (1, 2),
    _normalize_credit_course("TECHNICAL COMMUNICATION"): (1, 2),
    # ---------- Semester II ----------
    _normalize_credit_course("BIOLOGY FOR ENGINEERS"): (2, 2),
    _normalize_credit_course("ENGINEERING MATHEMATICS - II"): (2, 4),
    _normalize_credit_course("OBJECT ORIENTED PROGRAMMING IN C++"): (2, 4),
    _normalize_credit_course("ENGINEERING CHEMISTRY"): (2, 4),
    _normalize_credit_course("BASIC ELECTRONICS ENGINEERING"): (2, 3),
    _normalize_credit_course("BASIC MECHANICAL ENGINEERING"): (2, 3),
    _normalize_credit_course("ENGINEERING GRAPHICS"): (2, 3),
    # ---------- Semester III ----------
    _normalize_credit_course("ENGINEERING MATHEMATICS - III"): (3, 4),
    _normalize_credit_course("JAVA PROGRAMMING"): (3, 4),
    _normalize_credit_course("COMPUTER ORGANIZATION"): (3, 4),
    _normalize_credit_course("DATA STRUCTURES"): (3, 4),
    _normalize_credit_course("DIGITAL SYSTEM DESIGN"): (3, 4),
    _normalize_credit_course("DATA STRUCTURES LABORATORY"): (3, 1),
    _normalize_credit_course("DIGITAL SYSTEM DESIGN LABORATORY"): (3, 1),
    # ---------- Semester IV ----------
    _normalize_credit_course("DISCRETE STRUCTURES"): (4, 4),
    _normalize_credit_course("DESIGN & ANALYSIS OF ALGORITHMS"): (4, 4),
    _normalize_credit_course("FUNDAMENTALS OF DATABASE MANAGEMENT SYSTEMS"): (4, 3),
    _normalize_credit_course("COMPUTER ARCHITECTURE"): (4, 4),
    _normalize_credit_course("COMPUTER SYSTEM DESIGN LABORATORY"): (4, 1),
    _normalize_credit_course("DESIGN & ANALYSIS OF ALGORITHMS LABORATORY"): (4, 1),
    # Semester IV Electives
    _normalize_credit_course("ENGINEERING MATHEMATICS - IV"): (4, 4),
    _normalize_credit_course("MATHEMATICS FOR CYBER SECURITY"): (4, 4),
    _normalize_credit_course("COMPUTER GRAPHICS USING OPENGL"): (4, 4),
    _normalize_credit_course("OBJECT ORIENTED ANALYSIS & DESIGN"): (4, 4),
    # ---------- Semester V ----------
    _normalize_credit_course("THEORY OF COMPUTATION"): (5, 4),
    _normalize_credit_course("OPERATING SYSTEMS"): (5, 3),
    _normalize_credit_course("COMPUTER NETWORKS"): (5, 4),
    _normalize_credit_course("OPERATING SYSTEMS LABORATORY"): (5, 1),
    _normalize_credit_course("COMPUTER NETWORKS LABORATORY"): (5, 1),
    _normalize_credit_course("SOFT SKILLS - I"): (5, 1),
    # Semester V Electives
    _normalize_credit_course("STATISTICAL FOUNDATIONS FOR COMPUTER SCIENCE"): (5, 4),
    _normalize_credit_course("NETWORK TOOLS & TECHNIQUES"): (5, 4),
    _normalize_credit_course("ARTIFICIAL INTELLIGENCE"): (5, 4),
    _normalize_credit_course("PYTHON PROGRAMMING WITH WEB FRAMEWORKS"): (5, 4),
    _normalize_credit_course("LINUX PROGRAMMING"): (5, 4),
    _normalize_credit_course("DIGITAL IMAGE PROCESSING"): (5, 4),
    _normalize_credit_course("SYSTEM SOFTWARE"): (5, 4),
    _normalize_credit_course("SYSTEM MODELLING & SIMULATION"): (5, 4),
    _normalize_credit_course("SCRIPT PROGRAMMING"): (5, 4),
}


def _is_empty_credit_cell(value) -> bool:
    if pd.isna(value) or value == "" or value == "[]":
        return True
    if isinstance(value, list):
        return len(value) == 0
    try:
        return len(ast.literal_eval(value)) == 0
    except (ValueError, SyntaxError, TypeError):
        return True


def _get_group(course_list: List) -> str:
    if not isinstance(course_list, list) or len(course_list) < 2:
        return "UNKNOWN"
    second_course = str(course_list[1][0]).strip().upper()
    if "ENGINEERING CHEMISTRY" in second_course:
        return "B"
    if "BASIC ELECTRICAL ENGINEERING" in second_course:
        return "A"
    return "UNKNOWN"


def _compute_avg(course_list: List, sem_target: int, group: str, unmatched: set) -> float | None:
    total = 0.0
    credits_sum = 0.0

    c_sub = _normalize_credit_course("PROBLEM SOLVING & PROGRAMMING IN C")
    oop_sub = _normalize_credit_course("OBJECT ORIENTED PROGRAMMING IN C++")

    for item in course_list:
        if not isinstance(item, list) or len(item) < 2:
            continue
        name_raw = item[0]
        name = _normalize_credit_course(name_raw)
        try:
            score = float(item[1])
        except (TypeError, ValueError):
            continue

        if name not in COURSE_MAPPING:
            unmatched.add(name_raw)
            continue

        sem, credit = COURSE_MAPPING[name]

        if group == "A":
            if sem == sem_target:
                total += score * credit
                credits_sum += credit
        elif group == "B":
            if sem_target == 1:
                if name == c_sub:
                    total += score * credit
                    credits_sum += credit
                elif sem == 2 and name != oop_sub:
                    total += score * credit
                    credits_sum += credit
            elif sem_target == 2:
                if name == oop_sub:
                    total += score * credit
                    credits_sum += credit
                elif sem == 1 and name != c_sub:
                    total += score * credit
                    credits_sum += credit
            else:
                if sem == sem_target:
                    total += score * credit
                    credits_sum += credit
        else:
            if sem == sem_target:
                total += score * credit
                credits_sum += credit

    if credits_sum == 0:
        return None
    return total / credits_sum


def _parse_sgpa(value) -> Dict[int, float]:
    items = _parse_list_cell(value)
    res: Dict[int, float] = {}
    for item in items:
        if not isinstance(item, list) or len(item) < 2:
            continue
        try:
            sem = int(float(item[0]))
            sgpa_val = float(item[1])
        except (TypeError, ValueError):
            continue
        res[sem] = sgpa_val
    return res


def _parse_cgpa(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    parsed = _parse_list_cell(value)
    if parsed:
        try:
            return float(parsed[0])
        except (TypeError, ValueError):
            pass
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan


def _parse_arrears(value) -> Tuple[int, int]:
    if pd.isna(value):
        return 0, 0
    if isinstance(value, (int, float)):
        count = int(value)
        return count, 1 if count > 0 else 0
    items = _parse_list_cell(value)
    if not items:
        return 0, 0
    count = len(items)
    has_before_sem5 = 0
    for item in items:
        if isinstance(item, list) and item:
            try:
                sem = int(float(item[0]))
            except (TypeError, ValueError):
                continue
            if sem < 5:
                has_before_sem5 = 1
                break
    return count, 1 if has_before_sem5 else 0


def _normalize_branch(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    for prefix in ["B.Tech.-", "B.Tech. -", "B.Tech.", "BTECH", "B.E.-", "B.E. -", "B.E."]:
        if text.upper().startswith(prefix.upper()):
            text = text[len(prefix):].lstrip(" -")
            break
    return text.strip()


def _compute_semester_avgs(course_list: List, group: str, sem_max: int, unmatched: set) -> List[float | None]:
    return [_compute_avg(course_list, sem, group, unmatched) for sem in range(1, sem_max + 1)]


def _build_students_from_direct_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if "regno" not in df.columns:
        warnings.append("Missing required column: regno")
        return pd.DataFrame([]), warnings

    required_columns = [
        "Internal_Avg_Sem1",
        "Internal_Avg_Sem2",
        "Internal_Avg_Sem3",
        "Internal_Avg_Sem4",
        "Internal_Avg_Sem5",
        "Ext_Avg_Sem1",
        "Ext_Avg_Sem2",
        "Ext_Avg_Sem3",
        "Ext_Avg_Sem4",
    ]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        warnings.append(
            "Missing required columns for direct upload: " + ", ".join(missing_required)
        )
        return pd.DataFrame([]), warnings

    students: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        arrears_value = row.get("arrears", row.get("Has_Failure_Before_Sem5", 0))
        try:
            arrears_count = int(float(arrears_value)) if not pd.isna(arrears_value) else 0
        except (TypeError, ValueError):
            arrears_count = 0

        try:
            has_failure = int(float(row.get("Has_Failure_Before_Sem5", arrears_count)))
        except (TypeError, ValueError):
            has_failure = 1 if arrears_count > 0 else 0

        student = DEFAULT_STUDENT.copy()
        student.update(
            {
                "regno": str(row.get("regno", "")).strip(),
                "branch": _normalize_branch(row.get("department", row.get("branch", ""))),
                "admission_category": str(row.get("admission_category", "")).strip(),
                "hosteller": str(row.get("hosteller", "")).strip(),
                "CGPA": pd.to_numeric(row.get("CGPA", np.nan), errors="coerce"),
                "SGPA_Sem1": pd.to_numeric(row.get("SGPA_Sem1", np.nan), errors="coerce"),
                "SGPA_Sem2": pd.to_numeric(row.get("SGPA_Sem2", np.nan), errors="coerce"),
                "SGPA_Sem3": pd.to_numeric(row.get("SGPA_Sem3", np.nan), errors="coerce"),
                "SGPA_Sem4": pd.to_numeric(row.get("SGPA_Sem4", np.nan), errors="coerce"),
                "SGPA_Sem5": pd.to_numeric(row.get("SGPA_Sem5", np.nan), errors="coerce"),
                "Has_Failure_Before_Sem5": has_failure,
                "arrears": arrears_count,
                "Absent Percentage (%)": pd.to_numeric(row.get("Absent Percentage (%)", np.nan), errors="coerce"),
                "Internal_Avg_Sem1": pd.to_numeric(row.get("Internal_Avg_Sem1", np.nan), errors="coerce"),
                "Internal_Avg_Sem2": pd.to_numeric(row.get("Internal_Avg_Sem2", np.nan), errors="coerce"),
                "Internal_Avg_Sem3": pd.to_numeric(row.get("Internal_Avg_Sem3", np.nan), errors="coerce"),
                "Internal_Avg_Sem4": pd.to_numeric(row.get("Internal_Avg_Sem4", np.nan), errors="coerce"),
                "Internal_Avg_Sem5": pd.to_numeric(row.get("Internal_Avg_Sem5", np.nan), errors="coerce"),
                "Ext_Avg_Sem1": pd.to_numeric(row.get("Ext_Avg_Sem1", np.nan), errors="coerce"),
                "Ext_Avg_Sem2": pd.to_numeric(row.get("Ext_Avg_Sem2", np.nan), errors="coerce"),
                "Ext_Avg_Sem3": pd.to_numeric(row.get("Ext_Avg_Sem3", np.nan), errors="coerce"),
                "Ext_Avg_Sem4": pd.to_numeric(row.get("Ext_Avg_Sem4", np.nan), errors="coerce"),
            }
        )
        students.append(student)

    return pd.DataFrame(students), warnings


def build_students_from_xlsx(file_storage) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_excel(file_storage)
    warnings: List[str] = []

    if "regno" not in df.columns:
        warnings.append("Missing required column: regno")
        return pd.DataFrame([]), warnings

    direct_required = {
        "Internal_Avg_Sem1",
        "Internal_Avg_Sem2",
        "Internal_Avg_Sem3",
        "Internal_Avg_Sem4",
        "Internal_Avg_Sem5",
        "Ext_Avg_Sem1",
        "Ext_Avg_Sem2",
        "Ext_Avg_Sem3",
        "Ext_Avg_Sem4",
    }
    if direct_required.issubset(df.columns):
        direct_df, direct_warnings = _build_students_from_direct_df(df)
        warnings.extend(direct_warnings)
        if not direct_df.empty:
            return direct_df, warnings
        if direct_warnings:
            return direct_df, warnings

    if "credits" not in df.columns:
        warnings.append(
            "This file does not include the raw 'credits' column, and it does not match the direct semester-average upload format."
        )
        return pd.DataFrame([]), warnings

    df = df[~df["credits"].apply(_is_empty_credit_cell)].copy()
    if df.empty:
        warnings.append("No valid rows found after removing empty credits.")
        return pd.DataFrame([]), warnings

    unmatched: set = set()
    external_source_cols = [
        "external_marks",
        "external_credits",
        "external",
        "external_scores",
        "ext_marks",
        "ext_credits",
    ]

    students: List[Dict[str, object]] = []
    external_fallback_used = False

    for _, row in df.iterrows():
        credits_list = _parse_list_cell(row.get("credits"))
        group = _get_group(credits_list)
        internal_avgs = _compute_semester_avgs(credits_list, group, 5, unmatched)

        external_avgs = None
        for col in external_source_cols:
            if col in df.columns:
                external_list = _parse_list_cell(row.get(col))
                candidate = _compute_semester_avgs(external_list, group, 5, unmatched)
                if any(val is not None for val in candidate):
                    external_avgs = candidate
                    break

        if external_avgs is None:
            external_avgs = internal_avgs
            external_fallback_used = True

        sgpa_map = _parse_sgpa(row.get("sgpa"))
        cgpa_val = _parse_cgpa(row.get("cgpa"))
        arrears_count, has_failure = _parse_arrears(row.get("arrears"))

        student = DEFAULT_STUDENT.copy()
        student.update(
            {
                "regno": str(row.get("regno", "")).strip(),
                "branch": _normalize_branch(row.get("department", row.get("branch", ""))),
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
                "Absent Percentage (%)": pd.to_numeric(row.get("Absent Percentage (%)", np.nan), errors="coerce"),
                "Internal_Avg_Sem1": internal_avgs[0],
                "Internal_Avg_Sem2": internal_avgs[1],
                "Internal_Avg_Sem3": internal_avgs[2],
                "Internal_Avg_Sem4": internal_avgs[3],
                "Internal_Avg_Sem5": internal_avgs[4],
                "Ext_Avg_Sem1": external_avgs[0],
                "Ext_Avg_Sem2": external_avgs[1],
                "Ext_Avg_Sem3": external_avgs[2],
                "Ext_Avg_Sem4": external_avgs[3],
            }
        )

        students.append(student)

    if external_fallback_used:
        warnings.append("External marks columns not found; using internal averages as fallback.")

    if unmatched:
        sorted_unmatched = sorted(set(str(c) for c in unmatched))
        if len(sorted_unmatched) > 20:
            preview = ", ".join(sorted_unmatched[:20])
            warnings.append(
                f"Unmatched courses ignored (showing 20 of {len(sorted_unmatched)}): {preview}"
            )
        else:
            warnings.append(f"Unmatched courses ignored: {', '.join(sorted_unmatched)}")

    return pd.DataFrame(students), warnings
