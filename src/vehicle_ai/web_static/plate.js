const $ = (selector) => document.querySelector(selector);
let busy = false;
let stopRequested = false;
const queue = [];
const imageInputs = [$("#image-files"), $("#folder-files")];
const supportedImage = /\.(jpe?g|png|bmp|webp)$/i;

function selectedFiles() {
  return imageInputs.flatMap((input) => Array.from(input.files || []));
}

function updateFileSummary() {
  $("#file-summary").textContent = queue.length
    ? `Đã thêm ${queue.length} ảnh vào hàng đợi.`
    : "Chưa chọn ảnh.";
}

const node = (tag, text, className) => {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
};

const badge = (text) => node("span", text, "badge");
const meta = (label, value) => node("span", `${label}: ${value}`, "metric");

const syncMode = () => {
  const yaml = $("#plate-mode").value === "yaml";
  $("#plate-yaml").hidden = !yaml;
  $("#plate-manual").hidden = yaml;
};
$("#plate-mode").onchange = syncMode;
syncMode();

async function json(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = response.status === 413
      ? "Lô ảnh vượt giới hạn tải lên của máy chủ. Hãy giảm kích thước ảnh."
      : body.error || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return body;
}

async function loadConfig() {
  try {
    const config = await json("/api/plate/config");
    const missing = Object.entries(config.available || {})
      .filter(([, exists]) => !exists).map(([name]) => name);
    if (missing.length) {
      const warning = $("#model-warning");
      warning.textContent = `Thiếu model/config: ${missing.join(", ")}. Hãy kiểm tra đường dẫn trong phần cấu hình.`;
      warning.hidden = false;
    }
  } catch (error) {
    $("#error").textContent = `Không tải được cấu hình model: ${error.message}`;
  }
}

function vehicleCard(vehicle) {
  const card = node("article", undefined, "vehicle");
  const head = node("div", undefined, "vehicle-head");
  if (vehicle.plate_image) {
    const image = node("img", undefined, "thumb");
    image.src = vehicle.plate_image;
    image.alt = "Ảnh biển số";
    image.loading = "lazy";
    head.append(image);
  }
  head.append(node("div", vehicle.plate_text || (vehicle.plate_detected ? "Chưa đọc được" : "Không có biển"), "plate"));
  if (vehicle.plate_text) {
    const copy = node("button", "Copy", "copy");
    copy.onclick = async () => {
      try {
        await navigator.clipboard.writeText(vehicle.plate_text);
        copy.textContent = "Đã copy";
      } catch {
        copy.textContent = "Không copy được";
      }
    };
    head.append(copy);
  }
  card.append(head);
  card.append(node("p", vehicle.status || "", "metric"));
  const facts = node("div", undefined, "metadata");
  facts.append(
    badge(`Xe #${vehicle.index}`),
    badge(vehicle.vehicle_type || "Loại xe ?"),
  );
  if (vehicle.brand) facts.append(badge(vehicle.brand));
  facts.append(badge(vehicle.accepted ? "Biển đã đọc" : "Chưa chốt"));
  card.append(facts);
  const details = node("div", undefined, "metric");
  details.append(
    meta("Xe", `${(Number(vehicle.vehicle_confidence) * 100).toFixed(0)}%`),
    meta("Hãng", vehicle.brand ? `${(Number(vehicle.brand_confidence) * 100).toFixed(0)}%` : "—"),
    meta("Biển", vehicle.plate_detected ? `${(Number(vehicle.plate_confidence) * 100).toFixed(0)}%` : "—"),
    meta("OCR", vehicle.plate_text ? `${(Number(vehicle.plate_score) * 100).toFixed(0)}%` : "—"),
  );
  card.append(details);
  if (vehicle.plate_quality?.quality_factors) {
    const factors = vehicle.plate_quality.quality_factors;
    card.append(node("p", `Chất lượng ảnh biển: ${(Number(vehicle.plate_quality.quality_score) * 100).toFixed(0)}/100 · ${factors.plate_width}×${factors.plate_height}px · Nét ${factors.sharpness}`, "metric"));
  }
  return card;
}

function updateQueueSummary() {
  const completed = queue.filter((item) => item.status === "done" || item.status === "error").length;
  const errors = queue.filter((item) => item.status === "error").length;
  const plates = queue.reduce((total, item) => total + Number(item.summary?.plates_accepted || 0), 0);
  const progress = $("#queue-progress");
  progress.max = Math.max(1, queue.length);
  progress.value = completed;
  $("#queue-summary").textContent = queue.length
    ? `${completed}/${queue.length} đã xử lý · ${plates} biển đọc được · ${errors} lỗi`
    : "Chưa có ảnh.";
  updateFileSummary();
}

function updateQueueRow(item) {
  const labels = {
    pending: "Chờ xử lý",
    running: "Đang xử lý",
    done: "Hoàn tất",
    error: "Lỗi",
  };
  item.statusCell.textContent = labels[item.status];
  item.statusCell.className = `queue-status queue-status-${item.status}`;
  item.platesCell.textContent = item.plateTexts?.join(", ")
    || (item.status === "done" ? "Không có biển đọc được" : item.error || "—");
  item.platesCell.title = item.error || "";
}

function addSelectedFiles() {
  const files = selectedFiles();
  if (!files.length) return;
  const known = new Set(queue.map((item) => item.key));
  const body = $("#queue-items");
  const fragment = document.createDocumentFragment();
  let added = 0;
  let skipped = 0;

  for (const file of files) {
    const name = file.webkitRelativePath || file.name;
    if (!supportedImage.test(file.name)) {
      skipped += 1;
      continue;
    }
    const key = `${name.toLowerCase()}\u0000${file.size}\u0000${file.lastModified}`;
    if (known.has(key)) continue;
    known.add(key);

    const row = document.createElement("tr");
    const numberCell = node("td", String(queue.length + 1));
    const nameCell = node("td", name);
    nameCell.title = name;
    const statusCell = node("td", "Chờ xử lý", "queue-status queue-status-pending");
    const platesCell = node("td", "—");
    row.append(numberCell, nameCell, statusCell, platesCell);
    fragment.append(row);
    queue.push({
      file, name, key, status: "pending", summary: null, plateTexts: [],
      error: "", previewUrl: URL.createObjectURL(file), statusCell, platesCell,
    });
    added += 1;
  }

  if (added && queue.length === added) body.replaceChildren();
  body.append(fragment);
  if (skipped) {
    $("#error").textContent = `${skipped} tệp không phải ảnh JPG/PNG/BMP/WebP đã bị bỏ qua.`;
  } else if (added) {
    $("#error").textContent = "";
  }
  updateQueueSummary();
}

function renderResult(item, result) {
  const section = node("section", undefined, "panel image-result");
  const head = node("div", undefined, "result-head");
  head.append(
    node("h2", item.name),
    badge(`${result.summary?.vehicles || 0} xe có biển`),
    badge(`${result.summary?.plates_detected || 0} biển`),
    badge(`${result.summary?.plates_accepted || 0} biển đọc được`),
  );
  if (result.elapsed_ms != null) head.append(badge(`${Number(result.elapsed_ms).toFixed(0)} ms`));
  section.append(head);

  if (result.error) {
    section.append(node("p", result.error, "low-confidence"));
  } else {
    const previews = node("div", undefined, "preview-pair");
    const original = node("figure");
    original.append(node("figcaption", "Ảnh gốc"));
    const originalImage = node("img");
    originalImage.src = item.previewUrl;
    originalImage.alt = `Ảnh gốc ${item.name}`;
    originalImage.loading = "lazy";
    original.append(originalImage);
    previews.append(original);

    if (result.annotated) {
      const annotated = node("figure");
      annotated.append(node("figcaption", "Kết quả nhận diện"));
      const annotatedImage = node("img");
      annotatedImage.src = `data:image/jpeg;base64,${result.annotated}`;
      annotatedImage.alt = `Ảnh đã nhận diện ${item.name}`;
      annotatedImage.loading = "lazy";
      annotated.append(annotatedImage);
      previews.append(annotated);
    }
    section.append(previews);
    section.append(node("p", `Kích thước xử lý: ${result.width}×${result.height}px`, "metric"));
    const vehicles = result.vehicles || [];
    if (!vehicles.length) {
      section.append(node("p", "Không tìm thấy xe có biển số trong ảnh.", "metric"));
    } else {
      const grid = node("div", undefined, "vehicle-grid");
      vehicles.forEach((vehicle) => grid.append(vehicleCard(vehicle)));
      section.append(grid);
    }
  }
  $("#results").append(section);
}

function setBusy(value) {
  busy = value;
  $("#recognize").disabled = value;
  $("#stop-recognition").disabled = !value;
  $("#clear-queue").disabled = value;
  imageInputs.forEach((input) => { input.disabled = value; });
  $("#plate-form").querySelectorAll("input:not([type=file]), select")
    .forEach((input) => { input.disabled = value; });
}

async function recognizeQueue(event) {
  event.preventDefault();
  if (busy) return;
  addSelectedFiles();
  if (!queue.length) {
    $("#error").textContent = "Hãy chọn một ảnh hoặc một thư mục ảnh.";
    return;
  }
  const pending = queue.filter((item) => item.status !== "done");
  if (!pending.length) {
    $("#upload-progress").textContent = "Tất cả ảnh trong hàng đợi đã xử lý.";
    return;
  }

  const fields = Array.from(new FormData(event.currentTarget).entries())
    .filter(([key]) => key !== "image");
  stopRequested = false;
  setBusy(true);
  $("#error").textContent = "";
  $("#upload-progress").textContent = "";

  try {
    for (const item of pending) {
      if (stopRequested) break;
      item.status = "running";
      item.error = "";
      updateQueueRow(item);
      const completed = queue.filter((entry) => entry.status === "done" || entry.status === "error").length;
      $("#upload-progress").textContent = `Đang xử lý ${completed + 1}/${queue.length}: ${item.name}`;

      const formData = new FormData();
      fields.forEach(([key, value]) => formData.append(key, value));
      formData.append("image", item.file, item.name);
      try {
        const payload = await json("/api/plate/recognize", {
          method: "POST",
          body: formData,
        });
        const result = payload.results?.[0];
        if (!result) throw new Error("Máy chủ không trả kết quả cho ảnh này.");
        item.summary = result.summary || {};
        item.plateTexts = (result.vehicles || [])
          .map((vehicle) => vehicle.plate_text).filter(Boolean);
        if (result.error) {
          item.status = "error";
          item.error = result.error;
        } else {
          item.status = "done";
        }
        renderResult(item, result);
      } catch (error) {
        item.status = "error";
        item.error = error.message || String(error);
      }
      updateQueueRow(item);
      updateQueueSummary();
    }
  } finally {
    setBusy(false);
    $("#upload-progress").textContent = stopRequested
      ? "Đã dừng sau khi hoàn tất ảnh hiện tại."
      : `Hoàn tất xử lý ${queue.filter((item) => item.status === "done" || item.status === "error").length}/${queue.length} ảnh.`;
  }
}

function stopRecognition() {
  if (!busy) return;
  stopRequested = true;
  $("#stop-recognition").disabled = true;
  $("#upload-progress").textContent = "Đang dừng sau khi hoàn tất ảnh hiện tại…";
}

function clearQueue() {
  if (busy) return;
  queue.forEach((item) => URL.revokeObjectURL(item.previewUrl));
  queue.length = 0;
  imageInputs.forEach((input) => { input.value = ""; });
  const emptyRow = document.createElement("tr");
  const emptyCell = node("td", "Chưa có ảnh trong hàng đợi.");
  emptyCell.colSpan = 4;
  emptyRow.append(emptyCell);
  $("#queue-items").replaceChildren(emptyRow);
  $("#results").replaceChildren();
  $("#error").textContent = "";
  $("#upload-progress").textContent = "";
  updateQueueSummary();
}

imageInputs.forEach((input) => input.addEventListener("change", addSelectedFiles));
$("#plate-form").addEventListener("submit", recognizeQueue);
$("#stop-recognition").addEventListener("click", stopRecognition);
$("#clear-queue").addEventListener("click", clearQueue);
updateQueueSummary();

loadConfig();
