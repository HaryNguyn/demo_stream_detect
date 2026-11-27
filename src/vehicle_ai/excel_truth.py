"""Read image ground truth from the first worksheet containing PLATE_NUMBER."""


def read_excel_truth(source):
    from openpyxl import load_workbook

    book = load_workbook(source, read_only=True, data_only=True)
    try:
        for sheet in book.worksheets:
            rows = iter(sheet.values)
            for row in rows:
                headers = [str(value or "").strip().upper() for value in row]
                if "PLATE_NUMBER" in headers:
                    break
            else:
                continue
            plate = headers.index("PLATE_NUMBER")
            name = next((headers.index(key) for key in ("FILENAME", "IMAGE_PATH", "IMAGE_NAME")
                         if key in headers), None)
            labels = []
            seen = set()
            for row in rows:
                if all(value is None for value in row):
                    continue
                value = row[plate] if plate < len(row) else None
                label = {"plate": str(value).strip() if value is not None else ""}
                if name is not None:
                    filename = str(row[name] or "").strip().replace("\\", "/") if name < len(row) else ""
                    if not filename or filename.lower() in seen:
                        raise ValueError("Tên ảnh trong Excel bị trống hoặc trùng.")
                    seen.add(filename.lower())
                    label["filename"] = filename
                labels.append(label)
            if not labels:
                raise ValueError("Cột PLATE_NUMBER chưa có dòng dữ liệu.")
            return labels
        raise ValueError("Không tìm thấy cột PLATE_NUMBER trong Excel.")
    finally:
        book.close()
