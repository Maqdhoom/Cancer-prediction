import pandas as pd
from pathlib import Path

folder = Path("data/raw")
output = []

for file in folder.glob("*.xlsx"):
    output.append("=" * 80)
    output.append(f"FILE: {file.name}")
    output.append("=" * 80)

    excel = pd.ExcelFile(file)

    output.append(f"Sheets: {excel.sheet_names}")

    for sheet in excel.sheet_names:
        df = pd.read_excel(file, sheet_name=sheet)

        output.append("")
        output.append(f"--- SHEET: {sheet} ---")
        output.append(f"Rows: {df.shape[0]}")
        output.append(f"Columns: {df.shape[1]}")
        output.append("")
        output.append("COLUMN NAMES:")

        for i, col in enumerate(df.columns, 1):
            output.append(f"{i}. {col}")

        output.append("")
        output.append("DATA TYPES:")
        output.append(str(df.dtypes))

        output.append("")
        output.append("FIRST 3 ROWS:")
        output.append(str(df.head(3)))

        output.append("")

Path("dataset_structure.txt").write_text(
    "\n".join(output),
    encoding="utf-8"
)

print("Dataset structure saved to dataset_structure.txt")
