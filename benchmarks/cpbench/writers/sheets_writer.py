from __future__ import annotations

from typing import Any

from .rows import to_values


class SheetsWriter:
    def __init__(
        self,
        spreadsheet_id: str,
        credentials_path: str,
        worksheet_name: str = "results",
    ) -> None:
        self.spreadsheet_id = spreadsheet_id
        self.credentials_path = credentials_path
        self.worksheet_name = worksheet_name

    def _open_worksheet(self) -> Any:
        try:
            import gspread
        except ImportError as exc:
            raise ImportError(
                "gspread is required to write to Sheets. Install it with "
                "'pip install gspread', or use --dry-run to preview rows."
            ) from exc

        client = gspread.service_account(filename=self.credentials_path)
        spreadsheet = client.open_by_key(self.spreadsheet_id)
        try:
            return spreadsheet.worksheet(self.worksheet_name)
        except gspread.WorksheetNotFound:
            return spreadsheet.add_worksheet(title=self.worksheet_name, rows=1, cols=26)

    def append(self, rows: list[dict[str, Any]], columns: list[str]) -> int:
        if not rows:
            return 0

        worksheet = self._open_worksheet()
        header = worksheet.row_values(1)
        merged = header + [column for column in columns if column not in header]
        if merged != header:
            worksheet.update([merged], "A1")

        worksheet.append_rows(
            [to_values(row, merged) for row in rows],
            value_input_option="USER_ENTERED",
        )

        return len(rows)
