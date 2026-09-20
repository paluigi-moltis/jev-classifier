"""Flet GUI: text classification with TypeSafe Jev via Vercel AI Gateway.

Replicates the ollama-classifier-gui workflow (Settings -> Data -> Schema ->
Results) with Jev as the only classification engine.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import flet as ft

from .engine import Label, classify_async

MODEL_ID = "typesafe-ai/jev"


def _config_path() -> Path:
    base = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    d = Path(base) / "jev-classifier"
    d.mkdir(parents=True, exist_ok=True)
    return d / "config.json"


def load_config() -> dict[str, Any]:
    try:
        return json.loads(_config_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(config: dict[str, Any]) -> None:
    _config_path().write_text(json.dumps(config, indent=2))


class JevClassifierApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.config = load_config()

        # App state
        self.data_rows: list[dict[str, str]] = []
        self.data_path: str | None = None
        self.results: list[dict[str, Any]] = []
        self._classifying = False
        self.api_key_present = bool(os.environ.get("AI_GATEWAY_API_KEY"))

        # UI refs (strong refs kept on self; Refs are weak)
        self.model_field = ft.Ref[ft.TextField]()
        self.key_status = ft.Ref[ft.Text]()
        self.ask_urgency_switch = ft.Ref[ft.Switch]()
        self.ask_complaint_switch = ft.Ref[ft.Switch]()
        self.data_path_text = ft.Ref[ft.Text]()
        self.text_column_dropdown = ft.Ref[ft.Dropdown]()
        self.data_preview = ft.Ref[ft.Text]()
        self.labels_field = ft.Ref[ft.TextField]()
        self.run_btn = ft.Ref[ft.Button]()
        self.progress = ft.Ref[ft.ProgressBar]()
        self.results_status = ft.Ref[ft.Text]()
        self.results_table = ft.Ref[ft.DataTable]()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def main(self) -> None:
        self.page.title = "Jev Classifier — TypeSafe via Vercel AI Gateway"
        self.page.padding = 15
        self.page.window.width = 1000
        self.page.window.height = 720

        root = self._build_root()
        self.page.add(root)
        self.page.update()

    # ------------------------------------------------------------------
    # Builders (strong-local-then-assign pattern for every Ref)
    # ------------------------------------------------------------------

    def _build_root(self) -> ft.Tabs:
        settings_view = self._build_settings_view()
        data_view = self._build_data_view()
        schema_view = self._build_schema_view()
        results_view = self._build_results_view()
        root = ft.Tabs(
            length=4,
            selected_index=0,
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(
                        tabs=[
                            ft.Tab("Settings", icon=ft.Icons.SETTINGS_OUTLINED),
                            ft.Tab("Data", icon=ft.Icons.UPLOAD_FILE),
                            ft.Tab("Schema", icon=ft.Icons.LABEL_OUTLINE),
                            ft.Tab("Results", icon=ft.Icons.TABLE_CHART_OUTLINED),
                        ]
                    ),
                    ft.TabBarView(
                        expand=True,
                        controls=[settings_view, data_view, schema_view, results_view],
                    ),
                ],
            ),
        )
        self._root = root
        return root

    def _build_settings_view(self) -> ft.Column:
        model_field = ft.TextField(
            "typesafe-ai/jev",
            ref=self.model_field,
            label="Model ID (Vercel AI Gateway)",
            expand=True,
        )
        key_status = ft.Text(
            "API key: FOUND (AI_GATEWAY_API_KEY)" if self.api_key_present
            else "API key: MISSING — set AI_GATEWAY_API_KEY in the environment or .env.local",
            ref=self.key_status,
            color=ft.Colors.GREEN if self.api_key_present else ft.Colors.RED,
        )
        ask_urgency = ft.Switch(
            "Ask urgency (score question)",
            ref=self.ask_urgency_switch,
            value=bool(self.config.get("ask_urgency", True)),
        )
        ask_complaint = ft.Switch(
            "Ask complaint (boolean question)",
            ref=self.ask_complaint_switch,
            value=bool(self.config.get("ask_complaint", True)),
        )
        save_btn = ft.Button(
            "Save settings",
            icon=ft.Icons.SAVE,
            on_click=self._on_save_settings,
        )
        view = ft.Column(
            expand=True,
            controls=[
                ft.Text("Engine", theme_style=ft.TextThemeStyle.TITLE_MEDIUM),
                model_field,
                key_status,
                ft.Divider(),
                ft.Text("Extra questions (run in parallel with classification)", theme_style=ft.TextThemeStyle.TITLE_MEDIUM),
                ask_urgency,
                ask_complaint,
                ft.Container(content=save_btn),
            ],
        )
        self._settings_view = view
        return view

    def _build_data_view(self) -> ft.Column:
        pick_btn = ft.Button(
            "Load CSV…",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._on_pick_file,
        )
        path_text = ft.Text("No file loaded", ref=self.data_path_text, expand=True)
        column_dropdown = ft.Dropdown(
            ref=self.text_column_dropdown,
            label="Text column",
            expand=True,
            on_select=self._on_column_selected,
        )
        preview = ft.Text("", ref=self.data_preview, selectable=True)
        view = ft.Column(
            expand=True,
            controls=[
                ft.Row([pick_btn, path_text]),
                column_dropdown,
                preview,
            ],
        )
        self._data_view = view
        return view

    def _build_schema_view(self) -> ft.Column:
        labels_value = self.config.get(
            "labels",
            "billing | Charges, invoices, refunds, payment issues\n"
            "technical | Bugs, errors, outages, broken features\n"
            "sales | Pricing, plans, upgrades, new accounts\n"
            "support | Account changes and general assistance",
        )
        labels_field = ft.TextField(
            ref=self.labels_field,
            label="Labels — one per line: name | description",
            multiline=True,
            min_lines=8,
            value=labels_value,
            expand=True,
        )
        view = ft.Column(
            expand=True,
            controls=[
                ft.Text(
                    "One label per line. Format: name | description. "
                    "Max 255 labels (Jev Choice limit).",
                ),
                labels_field,
            ],
        )
        self._schema_view = view
        return view

    def _build_results_view(self) -> ft.Column:
        run_btn = ft.Button(
            "Classify",
            ref=self.run_btn,
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_run,
        )
        save_btn = ft.Button(
            "Save results CSV…",
            icon=ft.Icons.SAVE,
            on_click=self._on_save_results,
        )
        progress = ft.ProgressBar(ref=self.progress, visible=False)
        status = ft.Text("Load data and labels, then classify.", ref=self.results_status)
        table = ft.DataTable(
            ref=self.results_table,
            columns=[
                ft.DataColumn(ft.Text("Text")),
                ft.DataColumn(ft.Text("Predicted")),
                ft.DataColumn(ft.Text("Confidence")),
                ft.DataColumn(ft.Text("Urgency")),
                ft.DataColumn(ft.Text("Complaint")),
            ],
            rows=[],
        )
        view = ft.Column(
            expand=True,
            controls=[
                ft.Row([run_btn, save_btn]),
                progress,
                status,
                ft.Column([ft.Row([table])], scroll=ft.ScrollMode.AUTO, expand=True),
            ],
        )
        self._results_view = view
        return view

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _on_save_settings(self, _e: ft.ControlEvent) -> None:
        self.config["model"] = (self.model_field.current and self.model_field.current.value) or MODEL_ID
        self.config["ask_urgency"] = bool(self.ask_urgency_switch.current.value)
        self.config["ask_complaint"] = bool(self.ask_complaint_switch.current.value)
        save_config(self.config)
        self.page.show_dialog(
            ft.SnackBar(ft.Text("Settings saved."), duration=2000)
        )

    async def _on_pick_file(self, _e: ft.ControlEvent) -> None:
        files = await ft.FilePicker().pick_files(
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["csv", "txt"],
            dialog_title="Select a CSV file to classify",
        )
        if not files:
            return
        self._load_csv(files[0].path)

    def _load_csv(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self.data_rows = list(reader)
        self.data_path = path
        if self.data_path_text.current:
            self.data_path_text.current.value = f"{path}  ({len(self.data_rows)} rows)"
        columns = list(self.data_rows[0].keys()) if self.data_rows else []
        if self.text_column_dropdown.current:
            self.text_column_dropdown.current.options = [
                ft.DropdownOption(c) for c in columns
            ]
            self.text_column_dropdown.current.value = columns[0] if columns else None
        preview_text = "\n".join(
            " | ".join(str(v)[:40] for v in row.values()) for row in self.data_rows[:5]
        )
        if self.data_preview.current:
            self.data_preview.current.value = preview_text or "(empty file)"
        self.page.update()

    def _on_column_selected(self, _e: ft.ControlEvent) -> None:
        self.page.update()

    def _parse_labels(self) -> list[Label]:
        raw = self.labels_field.current.value or ""
        labels: list[Label] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                name, desc = (part.strip() for part in line.split("|", 1))
            else:
                name, desc = line, ""
            if name:
                labels.append(Label(name, desc))
        return labels

    async def _on_run(self, _e: ft.ControlEvent) -> None:
        if self._classifying:
            return
        column = self.text_column_dropdown.current.value if self.text_column_dropdown.current else None
        if not self.data_rows or not column:
            self.page.show_dialog(ft.SnackBar(ft.Text("Load a CSV and pick the text column first."), duration=3000))
            return
        labels = self._parse_labels()
        if not labels:
            self.page.show_dialog(ft.SnackBar(ft.Text("Define at least one label in Schema."), duration=3000))
            return
        if not self.api_key_present:
            self.page.show_dialog(ft.SnackBar(ft.Text("AI_GATEWAY_API_KEY is not set."), duration=3000))
            return

        self._classifying = True
        self.results = []
        if self.run_btn.current:
            self.run_btn.current.disabled = True
        if self.progress.current:
            self.progress.current.visible = True
        if self.results_table.current:
            self.results_table.current.rows = []
        self.page.update()

        model_id = (self.model_field.current and self.model_field.current.value) or MODEL_ID
        ask_urgency = bool(self.ask_urgency_switch.current.value)
        ask_complaint = bool(self.ask_complaint_switch.current.value)

        try:
            for i, row in enumerate(self.data_rows, start=1):
                text = str(row.get(column, "")).strip()
                try:
                    r = await classify_async(
                        text, labels,
                        ask_urgency=ask_urgency,
                        ask_complaint=ask_complaint,
                        model_id=model_id,
                    )
                except Exception as exc:
                    r = None
                    if self.results_status.current:
                        self.results_status.current.value = f"Row {i}: error {type(exc).__name__}: {exc}"
                self.results.append({
                    "text": text,
                    "predicted": r.label if r else "ERROR",
                    "confidence": r.confidence if r else None,
                    "urgency": r.urgency if r else None,
                    "complaint": r.complaint_probability if r else None,
                })
                if self.results_status.current:
                    self.results_status.current.value = f"Classifying… {i}/{len(self.data_rows)}"
                self._refresh_table()
                self.page.update()
        finally:
            self._classifying = False
            if self.run_btn.current:
                self.run_btn.current.disabled = False
            if self.progress.current:
                self.progress.current.visible = False
            errors = sum(1 for r in self.results if r["predicted"] == "ERROR")
            if self.results_status.current:
                self.results_status.current.value = (
                    f"Done: {len(self.results)} rows, {errors} errors."
                )
            self.page.update()

    def _refresh_table(self) -> None:
        if not self.results_table.current:
            return
        self.results_table.current.rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(r["text"][:60], tooltip=r["text"])),
                    ft.DataCell(ft.Text(str(r["predicted"]))),
                    ft.DataCell(ft.Text(f"{r['confidence']:.2f}" if r["confidence"] is not None else "—")),
                    ft.DataCell(ft.Text(f"{r['urgency']:.2f}" if r["urgency"] is not None else "—")),
                    ft.DataCell(ft.Text(f"{r['complaint']:.2f}" if r["complaint"] is not None else "—")),
                ]
            )
            for r in self.results
        ]

    async def _on_save_results(self, _e: ft.ControlEvent) -> None:
        if not self.results:
            self.page.show_dialog(ft.SnackBar(ft.Text("No results to save."), duration=2000))
            return
        path = await ft.FilePicker().save_file(
            file_name="jev_results.csv",
            dialog_title="Save classification results",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["text", "predicted", "confidence", "urgency", "complaint"]
            )
            writer.writeheader()
            writer.writerows(self.results)
        self.page.show_dialog(ft.SnackBar(ft.Text(f"Saved {len(self.results)} rows."), duration=2000))


def launch() -> None:
    try:
        from dotenv import load_dotenv

        for candidate in (Path.cwd() / ".env.local", Path(__file__).resolve().parents[2] / ".env.local"):
            if candidate.exists():
                load_dotenv(candidate)
                break
    except ImportError:
        pass
    ft.run(lambda page: JevClassifierApp(page).main())


if __name__ == "__main__":
    launch()
