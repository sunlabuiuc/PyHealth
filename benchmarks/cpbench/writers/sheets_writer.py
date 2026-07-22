from __future__ import annotations

from typing import Any


class SheetsWriter:
    def __init__(
        self,
        spreadsheet_id: str,
        credentials_path: str,
    ) -> None:
        self.spreadsheet_id = spreadsheet_id
        self.credentials_path = credentials_path


    def _open_worksheet(self, worksheet_name, columns: int = 26) -> Any:
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
            return spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            return spreadsheet.add_worksheet(
                title=worksheet_name, rows=1, cols=max(columns, 26)
            )


    def append(self, worksheet_name, rows: list[list[Any]], columns: list[str]) -> int:
        if not rows:
            return 0

        duplicates = {column for column in columns if columns.count(column) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate columns would collapse into one: {sorted(duplicates)}"
            )

        worksheet = self._open_worksheet(worksheet_name, len(columns))
        header = worksheet.row_values(1)
        merged = header + [column for column in columns if column not in header]
        # A sheet that predates these columns may be too narrow to hold the header.
        if len(merged) > worksheet.col_count:
            worksheet.resize(cols=len(merged))
        if merged != header:
            worksheet.update([merged], "A1")

        worksheet.append_rows(
            [self._align(row, columns, merged) for row in rows],
            value_input_option="USER_ENTERED",
        )

        return len(rows)


    @staticmethod
    def _align(row: list[Any], columns: list[str], merged: list[str]) -> list[Any]:
        """Reorder one row to the worksheet's header, blanking columns it lacks.
        """
        if len(row) > len(columns):
            raise ValueError(
                f"Row has {len(row)} values but only {len(columns)} columns are named"
            )

        by_column = dict(zip(columns, row))

        return [by_column.get(column, "") for column in merged]
