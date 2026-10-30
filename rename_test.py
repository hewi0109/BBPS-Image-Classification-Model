import os
import pandas as pd

SPREADSHEET_PATH = 'TestAnswers.xlsx'
FILENAME_COLUMN = 'Test name'
SCORE_COLUMN = 'BBPS'
TEST_DIR = 'BBPS Pictures/Test'


def rename_test_files():
    if not os.path.exists(SPREADSHEET_PATH):
        print(f"Error: Spreadsheet not found at '{SPREADSHEET_PATH}'")
        return

    df = pd.read_excel(SPREADSHEET_PATH)
    print(f"Loaded spreadsheet. Found {len(df)} rows.")

    success_count = 0
    mismatch_count = 0

    for index, row in df.iterrows():
        # Get raw entry from spreadsheet (e.g., 'Test_V1')
        excel_entry = str(row[FILENAME_COLUMN]).strip()
        score = str(row[SCORE_COLUMN]).strip()

        # Format it to match your folder files exactly (e.g., 'TEST_V1.PNG')
        target_filename = f"{excel_entry.upper()}.PNG"
        old_path = os.path.join(TEST_DIR, target_filename)

        if os.path.exists(old_path):
            # Format to your standard model pipeline structure: BBPS_X_TEST_V1.PNG
            new_filename = f"BBPS_{score}_{target_filename}"
            new_path = os.path.join(TEST_DIR, new_filename)

            os.rename(old_path, new_path)
            success_count += 1
        else:
            mismatch_count += 1

    print(f"\nDone! Successfully renamed {success_count} files inside your Test folder.")
    if mismatch_count > 0:
        print(f"Skipped {mismatch_count} rows where the file wasn't found in the directory.")


if __name__ == '__main__':
    rename_test_files()
