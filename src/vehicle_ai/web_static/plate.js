const $ = (selector) => document.querySelector(selector);
let busy = false;
let stopRequested = false;
const queue = [];
const imageInputs = [$("#image-files"), $("#folder-files")];
const supportedImage = /\.(jpe?g|png|bmp|webp)$/i;

function selectedFiles() {
  return imageInputs.flatMap((input) => Array.from(input.files || []));
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

const pageSize = 50;
const categories = {plate: "Biển số", vehicle_type: "Loại xe", brand: "Hãng xe"};
const labels = {
  pending: "Chờ xử lý", running: "Đang xử lý", correct: "Đúng", wrong: "Sai",
  unread: "Không đọc được", unlabeled: "Thiếu nhãn", error: "Lỗi xử lý",
};
const percent = new Intl.NumberFormat("vi-VN", {style: "percent", maximumFractionDigits: 1});
const recentPreviews = new Map();
let truth = new Map();
let page = 0;
let loadingTruth = false;
let previewUrl;
const tableUrls = new Map();

const pathKey = (name) => name.trim().replace(/\\/g, "/").replace(/^\.\//, "").toLowerCase();
const plateKey = (value) => value.toUpperCase().replace(/[^A-Z0-9]/g, "");
const labelKey = (value) => value.normalize("NFD").replace(/\p{M}/gu, "")
  .toLowerCase().replace(/đ/g, "d").trim().replace(/\s+/g, " ");
const basename = (path) => path.split("/").at(-1);
const normalize = (field, value) => field === "plate" ? plateKey(value) : labelKey(value);

function readTruth(text) {
  const entries = new Map();
  for (const [index, line] of text.replace(/^\uFEFF/, "").split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    let row;
    try {
      row = JSON.parse(line);
    } catch {
      throw new Error(`JSON không hợp lệ ở dòng ${index + 1}.`);
    }
    if (typeof row?.filename !== "string" || !row.filename.trim()
        || Object.keys(categories).some((key) => row[key] != null && typeof row[key] !== "string")) {
      throw new Error(`Tên tệp hoặc nhãn không hợp lệ ở dòng ${index + 1}.`);
    }
    const key = pathKey(row.filename);
    if (entries.has(key)) throw new Error(`Trùng tên tệp ở dòng ${index + 1}.`);
    entries.set(key, row);
  }
  if (!entries.size) throw new Error("Tệp nhãn chuẩn không có dữ liệu.");
  return entries;
}

function applyTruth() {
  const counts = new Map();
  for (const item of queue) counts.set(item.basename, (counts.get(item.basename) || 0) + 1);
  for (const item of queue) {
    item.expected = truth.get(item.path)
      || truth.get(item.relativePath)
      || (counts.get(item.basename) === 1 ? truth.get(item.basename) : null);
    // A basename-only label is ambiguous when several selected files share it.
    if (item.path === item.basename && counts.get(item.basename) > 1) item.expected = null;
  }
}

function expectedValue(item, field) {
  if (field !== "plate") return item.expected?.[field] || "";
  if (item.manualPlate !== undefined) return item.manualPlate;
  if (item.expected?.plate) return item.expected.plate;
  if (!$("#filename-truth").checked) return "";
  const stem = (item.basename || "").split("_")[0].replace(/\.(jpe?g|png|bmp|webp)$/i, "");
  if (!/^[a-z0-9 .-]+$/i.test(stem)) return "";
  const plate = plateKey(stem);
  return /^\d{2}[A-Z]{1,2}\d{4,6}$/.test(plate) ? plate : "";
}

function classify(item, field) {
  if (item.status !== "done") return item.status;
  const expected = normalize(field, expectedValue(item, field));
  const predicted = normalize(field, item.predicted?.[field] || "");
  if (field === "plate" && !predicted) return "unread";
  if (!expected) return "unlabeled";
  if (!predicted) return "unread";
  return expected === predicted ? "correct" : "wrong";
}

function primaryPrediction(vehicles) {
  const best = vehicles.reduce((previous, vehicle) => {
    if (!vehicle.accepted || !plateKey(vehicle.plate_text || "")) return previous;
    return !previous || Number(vehicle.plate_score || 0) > Number(previous.plate_score || 0)
      ? vehicle : previous;
  }, null);
  const vehicle = best || vehicles[0];
  return {
    plate: best?.plate_text || "",
    vehicle_type: vehicle?.vehicle_type || "",
    brand: vehicle?.brand || "",
    score: best?.plate_score,
    crop: best?.plate_image || vehicles.find((entry) => entry.plate_image)?.plate_image || "",
  };
}

function updateQueueSummary() {
  const scores = Object.fromEntries(Object.keys(categories).map((field) =>
    [field, {correct: 0, wrong: 0, unread: 0}]));
  let done = 0;
  let errors = 0;
  let reads = 0;
  let elapsed = 0;
  for (const item of queue) {
    if (item.status === "error") errors += 1;
    if (item.status !== "done") continue;
    done += 1;
    elapsed += Number(item.result?.elapsed_ms || 0);
    if (item.predicted.plate) reads += 1;
    for (const field of Object.keys(categories)) {
      if (normalize(field, expectedValue(item, field))) scores[field][classify(item, field)] += 1;
    }
  }

  const progress = $("#queue-progress");
  progress.max = Math.max(1, queue.length);
  progress.value = done + errors;
  $("#queue-summary").textContent = `${done + errors}/${queue.length} đã xử lý · ${reads}/${done} ảnh đọc được biển (${done ? percent.format(reads / done) : "—"}) · ${errors} lỗi`;
  const {correct, wrong, unread} = scores.plate;
  const graded = correct + wrong + unread;
  const kpis = [
    ["Ảnh đã xử lý", `${done + errors}/${queue.length}`],
    ["Tỉ lệ đọc được", done ? percent.format(reads / done) : "—"],
    ["Đúng / ảnh đã chấm", graded ? percent.format(correct / graded) : "—"],
    ["Chính xác khi đọc được", correct + wrong ? percent.format(correct / (correct + wrong)) : "—"],
    ["Đọc sai", wrong],
    ["Không đọc được", done - reads],
    ["Lỗi xử lý", errors],
    ["ms/ảnh trung bình", done ? Math.round(elapsed / done) : "—"],
  ];
  $("#batch-kpis").replaceChildren(...kpis.map(([label, value]) => {
    const card = node("div", undefined, "batch-kpi");
    card.append(node("strong", value), node("span", label));
    return card;
  }));
  $("#export-csv").disabled = done + errors === 0;
  const rows = Object.entries(categories).map(([field, name]) => {
    const row = node("tr");
    const label = node("th", name);
    label.scope = "row";
    const {correct, wrong, unread} = scores[field];
    const read = correct + wrong;
    const total = read + unread;
    row.append(label, node("td", total));
    for (const count of [correct, wrong, unread]) {
      row.append(node("td", total ? `${count} · ${percent.format(count / total)}` : "—"));
    }
    row.append(node("td", read ? percent.format(correct / read) : "—"));
    return row;
  });
  $("#score-items").replaceChildren(...rows);
  renderQueue();
}

function renderQueue() {
  if (document.activeElement?.classList.contains("manual-plate")) return;
  const filter = $("#result-filter").value;
  const items = filter === "all" ? queue : queue.filter((item) =>
    filter === "unlabeled" ? item.status === "done" && !plateKey(expectedValue(item, "plate"))
      : filter === "issues" ? ["wrong", "unread"].includes(classify(item, "plate"))
      : classify(item, "plate") === filter);
  const pages = Math.max(1, Math.ceil(items.length / pageSize));
  page = Math.min(page, pages - 1);
  $("#previous-page").disabled = page === 0;
  $("#next-page").disabled = page + 1 === pages;
  $("#page-status").textContent = `${page + 1}/${pages}`;

  const visible = items.slice(page * pageSize, (page + 1) * pageSize);
  const visibleItems = new Set(visible);
  for (const [item, url] of tableUrls) {
    if (visibleItems.has(item)) continue;
    URL.revokeObjectURL(url);
    tableUrls.delete(item);
  }
  const rows = visible.map((item) => {
    const row = node("tr");
    const photo = node("td");
    const thumbnail = node("button", undefined, "thumbnail-button");
    thumbnail.type = "button";
    thumbnail.setAttribute("aria-label", `Xem ảnh ${item.name}`);
    if (!tableUrls.has(item)) tableUrls.set(item, URL.createObjectURL(item.file));
    const image = node("img", undefined, "batch-photo");
    image.src = tableUrls.get(item);
    image.alt = "";
    image.loading = "lazy";
    thumbnail.append(image);
    thumbnail.onclick = () => showResult(item);
    photo.append(thumbnail);

    const crop = node("td", "—");
    if (item.predicted?.crop) {
      const plate = node("img", undefined, "batch-crop");
      plate.src = item.predicted.crop;
      plate.alt = "Ảnh crop biển số";
      plate.loading = "lazy";
      crop.replaceChildren(plate);
    }
    const prediction = node("td");
    prediction.append(node("strong", item.predicted?.plate || "—"));
    if (item.predicted?.score != null) prediction.append(node("div", percent.format(item.predicted.score), "metric"));
    const vehicle = [item.predicted?.vehicle_type, item.predicted?.brand].filter(Boolean).join(" · ");
    if (vehicle) prediction.append(node("div", vehicle, "metric"));

    const expected = node("td");
    const input = node("input", undefined, "manual-plate");
    input.type = "text";
    input.placeholder = "Nhập biển";
    input.autocomplete = "off";
    input.value = expectedValue(item, "plate");
    input.setAttribute("aria-label", `Biển chuẩn cho ảnh ${item.name}`);
    input.oninput = () => { item.manualPlate = input.value.trim(); };
    input.onblur = () => setTimeout(updateQueueSummary, 0);
    expected.append(input);

    const status = classify(item, "plate");
    const statusCell = node("td", labels[status], `queue-status queue-status-${status}`);
    statusCell.title = item.error || "";
    if (status === "unlabeled") statusCell.textContent = "Đọc được · thiếu nhãn";
    row.append(node("td", item.number), photo, node("td", item.name),
      crop, prediction, expected, statusCell,
      node("td", item.result?.elapsed_ms == null ? "—" : Math.round(item.result.elapsed_ms)));
    return row;
  });
  if (!rows.length) {
    const row = node("tr");
    const cell = node("td", queue.length ? "Không có ảnh phù hợp." : "Chưa chọn ảnh.");
    cell.colSpan = 8;
    row.append(cell);
    rows.push(row);
  }
  $("#queue-items").replaceChildren(...rows);
}

function addSelectedFiles() {
  const files = selectedFiles();
  if (!files.length) return;
  const known = new Set(queue.map((item) => item.key));
  let skipped = 0;

  for (const file of files) {
    if (!supportedImage.test(file.name)) {
      skipped += 1;
      continue;
    }
    const name = file.webkitRelativePath || file.name;
    const path = pathKey(name);
    const key = JSON.stringify([path, file.size, file.lastModified]);
    if (known.has(key)) continue;
    known.add(key);
    queue.push({
      file, name, path, key, number: queue.length + 1, status: "pending",
      relativePath: file.webkitRelativePath ? path.slice(path.indexOf("/") + 1) : null,
      basename: basename(path),
    });
  }
  imageInputs.forEach((input) => { input.value = ""; });
  $("#error").textContent = skipped ? `${skipped} tệp ảnh không được hỗ trợ đã bị bỏ qua.` : "";
  applyTruth();
  updateQueueSummary();
}

function showResult(item) {
  const dialog = $("#result-dialog");
  previewUrl = URL.createObjectURL(item.file);
  renderResult(item, recentPreviews.get(item) || item.result || {});
  dialog.showModal();
}
function renderResult(item, result, regenerated = false) {
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

  if (item.error) {
    section.append(node("p", item.error, "low-confidence"));
  }


  const previews = node("div", undefined, "preview-pair");
  const original = node("figure");
  original.append(node("figcaption", "Ảnh gốc"));
  const originalImage = node("img");
  originalImage.src = previewUrl;
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
  if (item.status === "done" && !result.annotated && !regenerated && !recentPreviews.has(item)) {
    const button = node("button", "Tạo lại ảnh đánh dấu");
    const status = node("p", "", "metric");
    status.setAttribute("role", "status");
    button.onclick = async () => {
      button.disabled = true;
      status.textContent = "Đang tạo ảnh…";
      try {
        const regenerated = await requestRecognition(item, item.fields);
        if (button.isConnected) renderResult(item, regenerated, true);
      } catch (error) {
        status.textContent = error.message;
        button.disabled = false;
      }
    };
    section.append(button, status);
  }
  if (result.width) section.append(node("p", `${result.width} × ${result.height} px`, "metric"));
  const vehicles = result.vehicles || [];
  if (!vehicles.length && item.status === "done") {
    section.append(node("p", "Không tìm thấy xe có biển số trong ảnh.", "metric"));
  } else if (vehicles.length) {
    const grid = node("div", undefined, "vehicle-grid");
    vehicles.forEach((vehicle) => grid.append(vehicleCard(vehicle)));
    section.append(grid);
  }
  $("#results").replaceChildren(section);
}

function setBusy(value) {
  busy = value;
  $("#recognize").disabled = value;
  $("#stop-recognition").disabled = !value;
  $("#clear-queue").disabled = value;
  $("#truth-file").disabled = value;
  imageInputs.forEach((input) => { input.disabled = value; });
  $("#plate-form").querySelectorAll("input:not([type=file]), select")
    .forEach((input) => { input.disabled = value; });
}

async function requestRecognition(item, fields) {
  const data = new FormData();
  fields.forEach(([key, value]) => data.append(key, value));
  data.append("image", item.file, item.name);
  const payload = await json("/api/plate/recognize", {method: "POST", body: data});
  const result = payload.results?.[0];
  if (!result) throw new Error("Máy chủ không trả kết quả cho ảnh này.");
  if (result.error) throw new Error(result.error);
  return result;
}

async function recognizeQueue(event) {
  event.preventDefault();
  if (busy || loadingTruth) return;
  addSelectedFiles();
  const pending = queue.filter((item) => item.status !== "done");
  if (!pending.length) {
    $("#upload-progress").textContent = queue.length ? "Đã xử lý hết ảnh." : "Hãy chọn ảnh hoặc thư mục.";
    return;
  }

  const fields = [...new FormData(event.currentTarget)];
  stopRequested = false;
  setBusy(true);
  $("#error").textContent = "";
  let nextRefresh = 0;
  try {
    for (const item of pending) {
      if (stopRequested) break;
      item.status = "running";
      item.error = "";
      if (performance.now() >= nextRefresh && !stopRequested) {
        $("#upload-progress").textContent = `Đang xử lý ảnh #${item.number}`;
        updateQueueSummary();
        nextRefresh = performance.now() + 150;
      }

      try {
        const result = await requestRecognition(item, fields);
        item.fields = fields;
        const {annotated, vehicles = [], ...summary} = result;
        item.predicted = primaryPrediction(vehicles);
        item.result = {...summary, vehicles: vehicles.map(({plate_image, ...vehicle}) => vehicle)};
        item.status = "done";
        recentPreviews.set(item, result);
        if (recentPreviews.size > 12) recentPreviews.delete(recentPreviews.keys().next().value);
      } catch (error) {
        item.status = "error";
        item.error = error.message || String(error);
      }
    }
  } finally {
    setBusy(false);
    updateQueueSummary();
    $("#upload-progress").textContent = stopRequested ? "Đã dừng." : "Đã xử lý xong lô ảnh.";
  }
}

function stopRecognition() {
  if (!busy) return;
  stopRequested = true;
  $("#stop-recognition").disabled = true;
  $("#upload-progress").textContent = "Đang hoàn tất ảnh hiện tại…";
}

function clearQueue() {
  if (busy) return;
  queue.length = 0;
  page = 0;
  recentPreviews.clear();
  imageInputs.forEach((input) => { input.value = ""; });
  $("#error").textContent = "";
  $("#upload-progress").textContent = "";
  updateQueueSummary();
}

$("#filename-truth").onchange = updateQueueSummary;
$("#export-csv").onclick = () => {
  const field = (value) => {
    const text = String(value ?? "");
    const safe = /^\s*[=+@-]|^[\t\r\n]/.test(text) ? "'" + text : text;
    return '"' + safe.replace(/"/g, '""') + '"';
  };
  const rows = [["Tên tệp", "Biển chuẩn", "Biển OCR", "Loại xe", "Hãng", "Kết quả", "ms", "Lỗi"]];
  for (const item of queue) {
    if (!["done", "error"].includes(item.status)) continue;
    rows.push([item.name, expectedValue(item, "plate"), item.predicted?.plate,
      item.predicted?.vehicle_type, item.predicted?.brand, labels[classify(item, "plate")],
      item.result?.elapsed_ms, item.error]);
  }
  const url = URL.createObjectURL(new Blob(["\uFEFF", rows.map((row) => row.map(field).join(",")).join("\r\n")],
    {type: "text/csv;charset=utf-8"}));
  const link = node("a");
  link.href = url;
  link.download = "ket-qua-nhan-dien.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

imageInputs.forEach((input) => input.addEventListener("change", addSelectedFiles));
$("#truth-file").addEventListener("change", async (event) => {
  const input = event.currentTarget;
  loadingTruth = true;
  input.disabled = true;
  $("#recognize").disabled = true;
  const status = $("#truth-status");
  try {
    const file = input.files[0];
    truth = file ? readTruth(await file.text()) : new Map();
    status.textContent = file ? `${file.name} · ${truth.size} nhãn` : "";
  } catch (error) {
    input.value = "";
    truth = new Map();
    status.textContent = error.message;
  } finally {
    loadingTruth = false;
    input.disabled = busy;
    $("#recognize").disabled = busy;
    applyTruth();
    updateQueueSummary();
  }
});
$("#result-filter").onchange = () => { page = 0; renderQueue(); };
$("#previous-page").onclick = () => { page -= 1; renderQueue(); };
$("#next-page").onclick = () => { page += 1; renderQueue(); };
$("#result-dialog").addEventListener("close", () => {
  $("#results").replaceChildren();
  URL.revokeObjectURL(previewUrl);
});
$("#plate-form").addEventListener("submit", recognizeQueue);
$("#stop-recognition").addEventListener("click", stopRecognition);
$("#clear-queue").addEventListener("click", clearQueue);
updateQueueSummary();
loadConfig();
