# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:19:57 2026

@author: elise
"""


# Build .exe:
#   pyinstaller --noconfirm --onefile --windowed --name GLORIA gloria.py

from __future__ import annotations

import io
import os
import json
import hashlib
import threading
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

pd.set_option("future.no_silent_downcasting", True)

APP_NAME = "GLORIA"
APP_SUBTITLE = "Simplified provenance attribution"
APP_VERSION = "1.10.0"

APP_DIR = Path.home() / "Documents" / "GLORIA"
MODELS_DIR = APP_DIR / "GLORIA_models"
APP_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TARGETS = [
    "Century",
    "Global region",
    "Region",
    "Country",
    "City",
    "Site",
    "Form",
    "Colour",
]

CONFIG = {
    "random_state": 42,
    "cv_folds": 3,
    "scoring": "f1_weighted",
    "smote_min_samples": 10,
    "rf_base_params": {
        "class_weight": "balanced",
        "n_jobs": 1
    },
    "rf_param_dist": {
        "n_estimators": np.arange(300, 501, 50),
        "max_depth": [30, 40, 50],
        "min_samples_split": np.arange(2, 11),
        "min_samples_leaf": [1],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    },
    "calibration": {
        "method": "sigmoid",
        "cv": 5
    }
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(col).strip() for col in result.columns]
    return result


def dataframe_hash(df: pd.DataFrame) -> str:
    normalized = normalize_columns(df).copy()
    return hashlib.sha256(
        pd.util.hash_pandas_object(normalized, index=True).values
    ).hexdigest()


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_sheet_names(file_path: str) -> list[str]:
    try:
        excel_file = pd.ExcelFile(file_path)
        return excel_file.sheet_names
    except Exception as exc:
        raise ValueError(f"Unable to read Excel sheets:\n{exc}") from exc


def load_database_columns(file_path: str, sheet_name: str) -> list[str]:
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
        df = normalize_columns(df)
        return list(df.columns)
    except Exception as exc:
        raise ValueError(f"Unable to read database columns:\n{exc}") from exc


def safe_output_name(input_path: str) -> str:
    stem = Path(input_path).stem
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_", " ") else "_" for ch in stem).strip()
    return f"{clean or 'output'}_GLORIA.xlsx"


def build_model_path(database_hash: str, target: str) -> str:
    target_safe = target.replace("/", "_").replace("\\", "_")
    model_dir = MODELS_DIR / database_hash
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir / f"{target_safe}_model.joblib")

def apply_smote_with_min_class_threshold(
    X_train,
    y_train,
    min_samples: int,
    random_state: int
    ):


    classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()

    sampling_strategy = {}

    for cls, count in zip(classes, counts):
        if count < max_count and count >= min_samples:
            sampling_strategy[cls] = max_count

    if len(sampling_strategy) == 0:
        print("SMOTE skipped: no class meets the minimum threshold.")
        return X_train, y_train

    # k_neighbors must be smaller than the smallest oversampled class size
    smallest_resampled_class = min(
        counts[np.isin(classes, list(sampling_strategy.keys()))]
    )
    k_neighbors = min(5, smallest_resampled_class - 1)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    return smote.fit_resample(X_train, y_train)

def train_model_for_target(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    common_columns: list[str],
    optimize_hyperparams: bool,
    resample_method: str | None,
    n_iter_search: int,
    config: dict
):
    print(f"Training model for {target_name}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))

    class_counts = np.bincount(y_encoded)
    min_samples = class_counts.min()

    X_train, y_train = X_scaled.copy(), y_encoded.copy()

    if resample_method == "smote":
        X_train, y_train = apply_smote_with_min_class_threshold(
        X_train=X_train,
        y_train=y_train,
        min_samples=config["smote_min_samples"],
        random_state=config["random_state"]
    )
    elif resample_method == "undersample":
        rus = RandomUnderSampler(random_state=config["random_state"])
        X_train, y_train = rus.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(
        random_state=config["random_state"],
        **config["rf_base_params"]
    )

    cv = StratifiedKFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=config["random_state"]
    )

    if optimize_hyperparams:
        search = RandomizedSearchCV(
            rf,
            param_distributions=config["rf_param_dist"],
            n_iter=n_iter_search,
            cv=cv,
            scoring=config["scoring"],
            random_state=config["random_state"],
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        rf = search.best_estimator_
        print(f"Best CV {config['scoring']}: {search.best_score_:.3f}")

    rf.fit(X_train, y_train)

    calibrated_rf = CalibratedClassifierCV(
        rf,
        method=config["calibration"]["method"],
        cv=config["calibration"]["cv"]
    )
    calibrated_rf.fit(X_train, y_train)

    print("Probabilities calibrated")

    return {
        "model": calibrated_rf,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": common_columns
    }


def predict_with_bundle(bundle, X_unknown):
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    features = bundle["features"]

    X_scaled = scaler.transform(X_unknown[features])

    proba = model.predict_proba(X_scaled)
    pred_idx = np.argmax(proba, axis=1)

    return (
        label_encoder.inverse_transform(pred_idx),
        np.max(proba, axis=1).round(2)
    )


def GLORIA_v10_gui(
    database_path: str,
    database_sheet: str,
    data_unknown_path: str,
    unknown_sheet: str,
    target_columns: list[str],
    optimize_hyperparams: bool = True,
    resample_method: str | None = None,
    n_iter_search: int = 25,
    force_retrain: bool = False,
    progress_callback=None,
):
    print("\n================ GLORIA v10 ================\n")

    models_dir = str(MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)

    if progress_callback:
        progress_callback(5, "Loading Excel files...")

    data_known = pd.read_excel(database_path, sheet_name=database_sheet)
    data_unknown = pd.read_excel(data_unknown_path, sheet_name=unknown_sheet)

    data_known = normalize_columns(data_known)
    data_unknown = normalize_columns(data_unknown)

    database_hash = dataframe_hash(data_known)

    exclude_cols = target_columns + [
        "ID", "Reference", "ID(Ref)", "Site", "City", "Country",
        "Region", "Global region", "Form", "Data Method?", "Colour",
        "Mg/Ca", "Mg", "Date - Early", "Date - Mean",
        "Date - Late", "Date",
        "(Na2O + MgO)/Sommes des autres"
    ]

    feature_columns = [column for column in data_known.columns if column not in exclude_cols]
    common_columns = [column for column in feature_columns if column in data_unknown.columns]

    if not target_columns:
        raise ValueError("Please select at least one target.")

    missing_targets = [target for target in target_columns if target not in data_known.columns]
    if missing_targets:
        raise ValueError(f"These selected targets are missing from the database sheet: {', '.join(missing_targets)}")

    if not common_columns:
        raise ValueError("No common usable feature columns were found between the selected sheets.")

    if progress_callback:
        progress_callback(15, "Cleaning known data...")

    replace_map = {
        "": 0,
        "REF": 0,
        "#VALUE!": 0,
        "-": 0,
        "<LOD": 0,
        "< LOD": 0,
        np.nan: 0,
        "BD": 0,
        "ND": 0
    }

    X_known = (
        data_known[common_columns]
        .replace(replace_map)
        .infer_objects(copy=False)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    if progress_callback:
        progress_callback(25, "Cleaning unknown data...")

    X_unknown = (
        data_unknown[common_columns]
        .replace(replace_map)
        .infer_objects(copy=False)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    results = {}
    confidences = {}
    total_targets = len(target_columns)

    for index, target in enumerate(target_columns, start=1):
        base_progress = 25 + int((index - 1) * (55 / max(total_targets, 1)))
        if progress_callback:
            progress_callback(base_progress, f"Processing target: {target}")

        print(f"\n{'=' * 60}")
        print(f"Target: {target}")

        model_path = build_model_path(database_hash, target)

        if os.path.exists(model_path) and not force_retrain:
            print("Loading existing model bundle")
            bundle = joblib.load(model_path)
        else:
            if os.path.exists(model_path):
                print("Retraining requested")
            else:
                print("No existing model found -> training")

            bundle = train_model_for_target(
                X=X_known,
                y=data_known[target],
                target_name=target,
                common_columns=common_columns,
                optimize_hyperparams=optimize_hyperparams,
                resample_method=resample_method,
                n_iter_search=n_iter_search,
                config=CONFIG
            )
            joblib.dump(bundle, model_path)

        preds, confs = predict_with_bundle(bundle, X_unknown)
        results[target] = preds
        confidences[f"{target}_confidence"] = confs

    if progress_callback:
        progress_callback(90, "Building output file...")

    df_out = pd.DataFrame(index=data_unknown.index)

    sample_column = next(
    (col for col in data_unknown.columns if str(col).strip().lower() in {"samples", "sample"}),
    None
    )

    if sample_column:
        df_out.insert(0, "Samples", data_unknown[sample_column])

    preferred_prediction_order = ["Global region", "Century"]
    remaining_predictions = [target for target in target_columns if target not in preferred_prediction_order]
    ordered_predictions = [target for target in preferred_prediction_order if target in target_columns] + remaining_predictions

    for target in ordered_predictions:
        if target in results:
            df_out[target] = results[target]

    for target in ordered_predictions:
        confidence_name = f"{target}_confidence"
        if confidence_name in confidences:
            df_out[confidence_name] = confidences[confidence_name]

    metadata = {
        "app_version": APP_VERSION,
        "database_file": Path(database_path).name,
        "database_sheet": database_sheet,
        "unknown_file": Path(data_unknown_path).name,
        "unknown_sheet": unknown_sheet,
        "target_columns": target_columns,
        "optimize_hyperparams": optimize_hyperparams,
        "resample_method": resample_method,
        "n_iter_search": n_iter_search,
        "force_retrain": force_retrain,
        "n_common_features": len(common_columns),
        "common_features": common_columns,
        "database_hash": database_hash,
    }

    metadata_path = APP_DIR / "last_run_metadata.json"
    save_json(metadata_path, metadata)

    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="predictions")

    output_buffer.seek(0)

    print(f"\nPredictions ready")
    print(f"Models stored in: {models_dir}")
    print(f"Database hash: {database_hash}")

    if progress_callback:
        progress_callback(100, "Done")

    return df_out, output_buffer.getvalue(), database_hash


class FileCard(ttk.Frame):
    def __init__(self, master: tk.Misc, title: str, button_text: str, on_change=None) -> None:
        super().__init__(master, style="Card.TFrame", padding=16)
        self.file_path = tk.StringVar()
        self.sheet_name = tk.StringVar()
        self.summary_var = tk.StringVar(value="No file selected.")
        self.on_change = on_change

        ttk.Label(self, text=title, style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )

        ttk.Label(self, text="File", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(14, 6), padx=(0, 8)
        )
        self.file_entry = ttk.Entry(self, textvariable=self.file_path, state="readonly", style="App.TEntry")
        self.file_entry.grid(row=1, column=1, sticky="ew", pady=(14, 6))
        ttk.Button(self, text=button_text, style="Accent.TButton", command=self.choose_file).grid(
            row=1, column=2, padx=(10, 0), pady=(14, 6)
        )

        ttk.Label(self, text="Sheet", style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", pady=(6, 6), padx=(0, 8)
        )
        self.sheet_combo = ttk.Combobox(self, textvariable=self.sheet_name, state="readonly", height=12)
        self.sheet_combo.grid(row=2, column=1, sticky="ew", pady=(6, 6))
        self.sheet_combo.bind("<<ComboboxSelected>>", self._on_sheet_selected)

        ttk.Button(self, text="Refresh", command=self.refresh_sheets).grid(
            row=2, column=2, padx=(10, 0), pady=(6, 6)
        )

        ttk.Label(self, textvariable=self.summary_var, style="Small.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        self.columnconfigure(1, weight=1)

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
        )
        if not path:
            return

        self.file_path.set(path)
        self.refresh_sheets()

    def refresh_sheets(self) -> None:
        path = self.file_path.get().strip()
        if not path:
            self.sheet_combo["values"] = []
            self.sheet_name.set("")
            self.summary_var.set("No file selected.")
            if self.on_change:
                self.on_change()
            return

        try:
            sheets = load_sheet_names(path)
        except Exception as exc:
            self.sheet_combo["values"] = []
            self.sheet_name.set("")
            self.summary_var.set("Unable to read workbook.")
            messagebox.showerror("Error", str(exc))
            if self.on_change:
                self.on_change()
            return

        self.sheet_combo["values"] = sheets
        if sheets:
            self.sheet_name.set(sheets[0])

        self.summary_var.set(f"{Path(path).name} — {len(sheets)} sheet(s) available")
        if self.on_change:
            self.on_change()

    def _on_sheet_selected(self, _event=None) -> None:
        if self.on_change:
            self.on_change()

    def get_values(self) -> tuple[str, str]:
        return self.file_path.get().strip(), self.sheet_name.get().strip()


class TargetSelector(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, style="Card.TFrame", padding=16)
        self.available_targets: list[str] = []

        ttk.Label(self, text="Targets", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            self,
            text="Choose one or more prediction targets from the database headers.",
            style="Small.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(6, 10))

        list_wrap = ttk.Frame(self, style="Card.TFrame")
        list_wrap.pack(fill="both", expand=True)

        self.listbox = tk.Listbox(
            list_wrap,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=8,
            bg="#fbfcff",
            fg="#22304a",
            borderwidth=0,
            relief="flat",
            font=("Segoe UI", 10),
            activestyle="none",
        )
        self.listbox.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_wrap, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        button_row = ttk.Frame(self, style="Card.TFrame")
        button_row.pack(fill="x", pady=(10, 0))

        ttk.Button(button_row, text="Select all", command=self.select_all).pack(side="left")
        ttk.Button(button_row, text="Clear", command=self.clear_selection).pack(side="left", padx=(8, 0))

    def set_targets(self, targets: list[str]) -> None:
        self.available_targets = targets
        self.listbox.delete(0, tk.END)
        for target in targets:
            self.listbox.insert(tk.END, target)

        default_targets = []
        for preferred in ["Century", "Global region"]:
            if preferred in targets:
                default_targets.append(preferred)

        if default_targets:
            for idx, target in enumerate(targets):
                if target in default_targets:
                    self.listbox.selection_set(idx)

    def get_selected_targets(self) -> list[str]:
        selected_indices = self.listbox.curselection()
        return [self.available_targets[idx] for idx in selected_indices]

    def select_all(self) -> None:
        if self.available_targets:
            self.listbox.selection_set(0, tk.END)

    def clear_selection(self) -> None:
        self.listbox.selection_clear(0, tk.END)


class GloriaPrettyApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_NAME} — Desktop")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        window_width = min(1120, int(screen_width * 0.9))
        window_height = min(820, int(screen_height * 0.9))
        
        self.geometry(f"{window_width}x{window_height}")
        self.minsize(900, 600)

        self.queue: Queue[tuple[str, Any]] = Queue()
        self.worker_thread: threading.Thread | None = None
        self.output_excel_bytes: bytes | None = None

        self.recalculate_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")
        self.progress_var = tk.DoubleVar(value=0)
        self.preview_var = tk.StringVar(value="No result yet.")
        self.target_info_var = tk.StringVar(value="No database selected yet.")

        self._configure_style()
        self._build_ui()
        self.after(100, self._poll_queue)

    def _configure_style(self) -> None:
        self.configure(bg="#f4f6fb")

        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", font=("Segoe UI", 10))
        style.configure("App.TFrame", background="#f4f6fb")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Header.TFrame", background="#f4f6fb")
        style.configure("Footer.TFrame", background="#f4f6fb")

        style.configure(
            "Title.TLabel",
            background="#f4f6fb",
            foreground="#1f2a44",
            font=("Segoe UI", 22, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#f4f6fb",
            foreground="#5a6780",
            font=("Segoe UI", 10),
        )
        style.configure(
            "CardTitle.TLabel",
            background="#ffffff",
            foreground="#1f2a44",
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Muted.TLabel",
            background="#ffffff",
            foreground="#5f6b7a",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Small.TLabel",
            background="#ffffff",
            foreground="#6d7888",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 10),
        )
        style.configure(
            "BigAccent.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(18, 12),
        )
        style.configure(
            "Soft.TButton",
            padding=(14, 10),
        )
        style.configure(
            "App.TEntry",
            fieldbackground="#fbfcff",
            borderwidth=1,
            padding=8,
        )
        style.configure(
            "Pretty.Horizontal.TProgressbar",
            troughcolor="#e8edf6",
            background="#5372f0",
            bordercolor="#e8edf6",
            lightcolor="#5372f0",
            darkcolor="#5372f0",
        )

    def _build_ui(self) -> None:
        container = ttk.Frame(self, style="App.TFrame")
        container.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(container, bg="#f4f6fb", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="App.TFrame", padding=20)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _resize_frame(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", _resize_frame)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        root = scrollable_frame
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        header = ttk.Frame(root, style="Header.TFrame")
        header.pack(fill="x", pady=(0, 14))

        ttk.Label(header, text=APP_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text=APP_SUBTITLE, style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        ttk.Label(
            header,
            text="Load both Excel files, choose the sheets, select targets, then run the attribution.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(8, 0))

        top_grid = ttk.Frame(root, style="App.TFrame")
        top_grid.pack(fill="x", pady=(0, 14))
        top_grid.columnconfigure(0, weight=1)
        top_grid.columnconfigure(1, weight=1)

        self.database_card = FileCard(top_grid, "1) Reference database", "Browse", on_change=self.refresh_targets_from_database)
        self.database_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.unknown_card = FileCard(top_grid, "2) Unknown data", "Browse")
        self.unknown_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        middle_grid = ttk.Frame(root, style="App.TFrame")
        middle_grid.pack(fill="x", pady=(0, 14))
        middle_grid.columnconfigure(0, weight=1)
        middle_grid.columnconfigure(1, weight=1)

        self.target_selector = TargetSelector(middle_grid)
        self.target_selector.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        options_card = ttk.Frame(middle_grid, style="Card.TFrame", padding=16)
        options_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ttk.Label(options_card, text="Options", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Checkbutton(
            options_card,
            text="Recalculate trees",
            variable=self.recalculate_var,
        ).pack(anchor="w", pady=(12, 0))
        ttk.Label(
            options_card,
            text="If unchecked, existing saved models will be reused only for the same database hash.",
            style="Small.TLabel",
            justify="left",
            wraplength=420,
        ).pack(anchor="w", pady=(6, 0))
        ttk.Label(
            options_card,
            textvariable=self.target_info_var,
            style="Small.TLabel",
            justify="left",
            wraplength=420,
        ).pack(anchor="w", pady=(18, 0))

        lower_grid = ttk.Frame(root, style="App.TFrame")
        lower_grid.pack(fill="both", expand=True, pady=(0, 14))
        lower_grid.columnconfigure(0, weight=1)
        lower_grid.columnconfigure(1, weight=1)
        lower_grid.rowconfigure(0, weight=1)

        preview_card = ttk.Frame(lower_grid, style="Card.TFrame", padding=16)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        ttk.Label(preview_card, text="Summary", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            preview_card,
            textvariable=self.preview_var,
            style="Small.TLabel",
            justify="left",
            wraplength=460,
        ).pack(anchor="w", pady=(10, 0))

        help_block = (
            "Output column order:\n"
            "• Samples\n"
            "• predictions\n"
            "• confidence scores\n\n"
            "Models are stored by database hash."
        )
        ttk.Label(
            preview_card,
            text=help_block,
            style="Small.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(18, 0))

        logs_card = ttk.Frame(lower_grid, style="Card.TFrame", padding=16)
        logs_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ttk.Label(logs_card, text="Log", style="CardTitle.TLabel").pack(anchor="w")

        text_wrap = ttk.Frame(logs_card, style="Card.TFrame")
        text_wrap.pack(fill="both", expand=True, pady=(10, 0))

        self.log_text = tk.Text(
            text_wrap,
            height=14,
            wrap="word",
            borderwidth=0,
            relief="flat",
            bg="#fbfcff",
            fg="#22304a",
            font=("Consolas", 10),
            padx=10,
            pady=10,
        )
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(text_wrap, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.configure(state="disabled")

        footer = ttk.Frame(root, style="Footer.TFrame")
        footer.pack(fill="x")

        self.progress = ttk.Progressbar(
            footer,
            style="Pretty.Horizontal.TProgressbar",
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
        )
        self.progress.pack(fill="x")

        bottom_line = ttk.Frame(footer, style="Footer.TFrame")
        bottom_line.pack(fill="x", pady=(8, 0))

        ttk.Label(bottom_line, textvariable=self.status_var, style="Subtitle.TLabel").pack(side="left")
        ttk.Label(
            bottom_line,
            text=f"Models folder: {MODELS_DIR}",
            style="Subtitle.TLabel",
        ).pack(side="right")

        actions = ttk.Frame(root, style="App.TFrame")
        actions.pack(fill="x", pady=(16, 0))

        self.run_button = ttk.Button(
            actions,
            text="Run attribution",
            style="BigAccent.TButton",
            command=self.start_processing,
        )
        self.run_button.pack(side="left")

        self.save_button = ttk.Button(
            actions,
            text="Save result",
            style="Soft.TButton",
            command=self.save_result,
            state="disabled",
        )
        self.save_button.pack(side="left", padx=(10, 0))

        ttk.Button(
            actions,
            text="Quit",
            style="Soft.TButton",
            command=self.destroy,
        ).pack(side="right")

        self._append_log("Application ready.")
        self._append_log(f"Models folder: {MODELS_DIR}")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_running_state(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")

    def refresh_targets_from_database(self) -> None:
        database_path, database_sheet = self.database_card.get_values()

        if not database_path or not database_sheet:
            self.target_selector.set_targets([])
            self.target_info_var.set("No database selected yet.")
            return

        try:
            columns = load_database_columns(database_path, database_sheet)
            available_targets = [col for col in ALLOWED_TARGETS if col in columns]
            self.target_selector.set_targets(available_targets)
            self.target_info_var.set(
                f"Available targets in this database sheet:\n"
                f"{', '.join(available_targets) if available_targets else 'None'}"
            )
            self._append_log(f"Detected available targets: {', '.join(available_targets) if available_targets else 'None'}")
        except Exception as exc:
            self.target_selector.set_targets([])
            self.target_info_var.set("Unable to detect targets from the selected database sheet.")
            messagebox.showerror("Error", str(exc))

    def _update_preview_before_run(
        self,
        database_path: str,
        database_sheet: str,
        unknown_path: str,
        unknown_sheet: str,
        selected_targets: list[str],
    ) -> None:
        self.preview_var.set(
            f"Database: {Path(database_path).name}\n"
            f"Database sheet: {database_sheet}\n\n"
            f"Unknown file: {Path(unknown_path).name}\n"
            f"Unknown sheet: {unknown_sheet}\n\n"
            f"Targets: {', '.join(selected_targets)}\n"
            f"Resampling: smote\n"
            f"Hyperparameter search: enabled"
        )

    def start_processing(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Processing", "A process is already running.")
            return

        database_path, database_sheet = self.database_card.get_values()
        unknown_path, unknown_sheet = self.unknown_card.get_values()
        selected_targets = self.target_selector.get_selected_targets()

        if not database_path:
            messagebox.showwarning("Missing file", "Please choose the reference database file.")
            return
        if not database_sheet:
            messagebox.showwarning("Missing sheet", "Please choose the database sheet.")
            return
        if not unknown_path:
            messagebox.showwarning("Missing file", "Please choose the unknown data file.")
            return
        if not unknown_sheet:
            messagebox.showwarning("Missing sheet", "Please choose the unknown data sheet.")
            return
        if not selected_targets:
            messagebox.showwarning("Missing target", "Please select at least one target.")
            return

        self.output_excel_bytes = None
        self.save_button.configure(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("Starting...")
        self._update_preview_before_run(
            database_path,
            database_sheet,
            unknown_path,
            unknown_sheet,
            selected_targets,
        )

        self._append_log("")
        self._append_log("=== New run ===")
        self._append_log(f"Database: {database_path}")
        self._append_log(f"Database sheet: {database_sheet}")
        self._append_log(f"Unknown file: {unknown_path}")
        self._append_log(f"Unknown sheet: {unknown_sheet}")
        self._append_log(f"Targets: {', '.join(selected_targets)}")
        self._append_log(f"Recalculate trees: {'yes' if self.recalculate_var.get() else 'no'}")

        self._set_running_state(True)

        self.worker_thread = threading.Thread(
            target=self._worker_run,
            args=(
                database_path,
                database_sheet,
                unknown_path,
                unknown_sheet,
                selected_targets,
                self.recalculate_var.get(),
            ),
            daemon=True,
        )
        self.worker_thread.start()

    def _worker_run(
        self,
        database_path: str,
        database_sheet: str,
        unknown_path: str,
        unknown_sheet: str,
        selected_targets: list[str],
        force_retrain: bool,
    ) -> None:
        try:
            def progress_callback(percent: int, message: str) -> None:
                self.queue.put(("progress", (percent, message)))

            output_df, excel_bytes, database_hash = GLORIA_v10_gui(
                database_path=database_path,
                database_sheet=database_sheet,
                data_unknown_path=unknown_path,
                unknown_sheet=unknown_sheet,
                target_columns=selected_targets,
                optimize_hyperparams=True,
                resample_method="smote",
                n_iter_search=30,
                force_retrain=force_retrain,
                progress_callback=progress_callback,
            )
            self.queue.put(("success", (output_df, excel_bytes, unknown_path, selected_targets, database_hash)))
        except Exception as exc:
            error_text = f"{exc}\n\n{traceback.format_exc()}"
            self.queue.put(("error", error_text))

    def _poll_queue(self) -> None:
        try:
            while True:
                event_type, payload = self.queue.get_nowait()

                if event_type == "progress":
                    percent, message = payload
                    self.progress_var.set(percent)
                    self.status_var.set(message)
                    self._append_log(message)

                elif event_type == "success":
                    output_df, excel_bytes, unknown_path, selected_targets, database_hash = payload
                    self.output_excel_bytes = excel_bytes
                    self.progress_var.set(100)
                    self.status_var.set("Finished.")
                    self.save_button.configure(state="normal")
                    self._set_running_state(False)

                    self.preview_var.set(
                        f"Result ready.\n\n"
                        f"Exported rows: {len(output_df)}\n"
                        f"Output columns: {len(output_df.columns)}\n"
                        f"Source file: {Path(unknown_path).name}\n"
                        f"Targets used: {', '.join(selected_targets)}\n"
                        f"Database hash: {database_hash[:12]}...\n\n"
                        f"You can now save the Excel result."
                    )

                    self._append_log(f"Database hash: {database_hash}")
                    self._append_log("Processing finished successfully.")
                    messagebox.showinfo(
                        "Success",
                        "Processing is complete.\nClick 'Save result' to export the Excel file.",
                    )

                elif event_type == "error":
                    self.progress_var.set(0)
                    self.status_var.set("Error.")
                    self._set_running_state(False)
                    self._append_log("An error occurred.")
                    self._append_log(payload)
                    messagebox.showerror("Error", payload.split("\n\n", 1)[0])

        except Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def save_result(self) -> None:
        if not self.output_excel_bytes:
            messagebox.showwarning("No result", "There is no result to save.")
            return

        unknown_path, _ = self.unknown_card.get_values()
        default_name = safe_output_name(unknown_path)

        save_path = filedialog.asksaveasfilename(
            title="Save output Excel file",
            defaultextension=".xlsx",
            initialfile=default_name,
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not save_path:
            return

        try:
            Path(save_path).write_bytes(self.output_excel_bytes)
        except Exception as exc:
            messagebox.showerror("Error", f"Unable to save file:\n{exc}")
            return

        self._append_log(f"Saved file: {save_path}")
        self.status_var.set("File saved.")
        messagebox.showinfo("Saved", f"Result saved to:\n{save_path}")


def main() -> None:
    app = GloriaPrettyApp()
    app.mainloop()


if __name__ == "__main__":
    main()