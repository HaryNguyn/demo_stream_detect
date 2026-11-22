const $ = (selector) => document.querySelector(selector);
let eventsSignature = "", revision = -1;
let submitting = false;
$("#mode").onchange = () => {
  const yaml = $("#mode").value === "yaml";
  $("#yaml-settings").hidden = !yaml;
  $("#manual-settings").hidden = yaml;
};
$("#ocr-choice").onchange = () => {
  const vn = $("#ocr-choice").value === "vn";
  $('[name="ocr_weights"]').value = vn ? "models/ocr_model.onnx" : "models/cct_s_v1_vn.onnx";
  $('[name="ocr_config"]').value = vn ? "models/ocr_model.yaml" : "models/cct_s_v1_vn_plate_config.yaml";
};
const labels = {idle: "Sẵn sàng", running: "Đang xử lý", stopping: "Đang dừng…", stopped: "Đã dừng", done: "Hoàn tất", error: "Có lỗi"};
const rejectionLabels = {
  vehicle_too_small: "xe quá nhỏ", roi_rejected: "ngoài ROI",
  no_plate_detected: "không thấy biển", plate_model_disabled: "chưa bật model biển",
  ocr_disabled: "chưa bật OCR", empty_vehicle_crop: "ảnh xe rỗng",
  empty_plate_crop: "vùng biển rỗng", plate_too_small: "biển quá nhỏ",
  plate_too_blurry: "biển mờ", ocr_budget_skipped: "bỏ qua do ngân sách OCR",
  ocr_rejected: "OCR không chấp nhận", quality_below_threshold: "ảnh dưới ngưỡng",
  plate_not_confirmed: "chưa đủ phiếu biển", no_plate_read: "không có lần đọc biển",
  merged_duplicate: "đã gộp trùng", low_dropped: "bỏ nhãn vàng",
  no_qualified_evidence: "không có ảnh đạt ngưỡng",
  hidden_by_recognized_only: "ẩn do chỉ lấy xe đọc được biển"
};
function reasonParts(reasons, details = {}) {
  return Object.entries(reasons || {}).filter(([, value]) => Number(value) > 0).map(([key, value]) => {
    const detail = details?.[key];
    return detail ? `${rejectionLabels[key] || key} (${value} · ${detail})`
      : `${rejectionLabels[key] || key} (${value})`;
  });
}
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function showPhoto(url) {
  $("#photo img").src = url;
  $("#photo").showModal();
}
$("#close-photo").onclick = () => $("#photo").close();
function renderEvents(rows) {
  const signature = JSON.stringify(rows);
  if (signature === eventsSignature) return;
  eventsSignature = signature;
  $("#count").textContent = `${rows.length} xe`;
  $("#cards").replaceChildren();
  if (!rows.length) $("#cards").append(node("p", "Chưa có xe đọc được biển số. Kết quả xuất hiện khi xe mất dấu hoặc phiên kết thúc."));
  for (const row of rows) {
    const card = node("article", undefined, "card");
    const head = node("div", undefined, "card-head");
    const photos = row.photos || {};
    if (photos.plate || photos.detected) {
      const image = node("img", undefined, "thumb");
      image.src = photos.plate || photos.detected;
      image.alt = "Ảnh xe / biển số"; image.loading = "lazy";
      image.onerror = () => { image.hidden = true; };
      head.append(image);
    }
    head.append(node("div", row["Biển số"] || "Chưa đọc được biển", "plate"));
    if (row["Biển số"]) {
      const copy = node("button", "Copy", "copy");
      copy.onclick = async () => {
        try { await navigator.clipboard.writeText(row["Biển số"]); copy.textContent = "Đã copy"; }
        catch { copy.textContent = "Không copy được"; }
      };
      head.append(copy);
    }
    card.append(head);
    if (row.confidence_status === "low") {
      card.append(node("p", "⚠ Độ tin cậy thấp · Cần kiểm tra ảnh, biển số chưa được xác nhận", "low-confidence"));
    }
    const meta = node("div", undefined, "metadata");
    let time = `${Number(row["Xuất hiện (giây)"] || 0).toFixed(1)}s`;
    if (row["Kết thúc (giây)"] != null) time += ` → ${Number(row["Kết thúc (giây)"]).toFixed(1)}s`;
    meta.append(node("span", `Xe #${row["Track ID"]}`, "badge"), node("span", time, "badge"));
    card.append(meta, node("p", row.Camera || ""), node("div", `Loại xe: ${row["Loại xe"] || "Chưa xác định"} · Hãng: ${row["Hãng"] || "Chưa xác định"}`, "details"));
    const decisions = row.classification_status || {};
    card.append(node("p", `Loại xe: ${decisions.vehicle_type?.confirmed ? "đã chốt" : "chưa chốt"} · Hãng: ${decisions.brand?.confirmed ? "đồng thuận trong các lớp model biết" : "chưa đồng thuận"}`, "details"));
    if (row.plate_decision?.best) {
      const decision = row.plate_decision;
      const reasons = {reads: "số lần đọc", score: "tổng điểm", mean: "điểm trung bình", margin: "cách biệt", share: "tỷ lệ phiếu", image_quality: "chất lượng ảnh"};
      card.append(node("p", `Phiếu biển: ${decision.reads} lần · Tổng ${decision.score} · TB ${decision.mean} · Cách biệt ${decision.margin} · Tỷ lệ ${(decision.share * 100).toFixed(0)}%`
        + (decision.unmet?.length ? ` · Chưa đạt: ${decision.unmet.map(key => reasons[key] || key).join(", ")}` : ""), "details"));
    }
    if (row.image_scores?.plate > 0) {
      card.append(node("p", `Chất lượng ảnh biển: ${(row.image_scores.plate * 100).toFixed(0)}/100`, "details"));
    }
    const foundReasons = reasonParts(row.rejection_reasons, row.rejection_detail);
    if (foundReasons.length) {
      card.append(node("p", `Lý do trong track: ${foundReasons.join(" · ")}`, "details"));
    }
    for (const [kind, label] of [["plate", "biển"], ["detected", "xe"]]) {
      const candidates = row.photo_candidates?.[kind] || [];
      if (!candidates.length) continue;
      const gallery = node("details");
      gallery.append(node("summary", `${candidates.length} ảnh ${label} tốt nhất`));
      const images = node("div", undefined, "metadata");
      for (const candidate of candidates) {
        const button = node("button", undefined, "candidate-photo");
        const image = node("img", undefined, "thumb");
        image.src = candidate.url; image.loading = "lazy"; image.alt = `Ảnh ${label}`;
        const factors = candidate.quality_factors || {};
        button.title = `Chất lượng: ${(candidate.quality_score * 100).toFixed(1)}/100 · OCR: ${(candidate.ocr_score * 100).toFixed(1)}%`
          + (factors.area_pixels ? ` · ${factors.plate_width}×${factors.plate_height}px · Nét: ${factors.sharpness} · Cách mép: ${factors.edge_margin_pixels}px · W/H: ${factors.aspect_ratio}` : "");
        button.append(image, node("span", `${(candidate.quality_score * 100).toFixed(0)}/100`));
        button.onclick = () => showPhoto(candidate.url);
        images.append(button);
      }
      gallery.append(images); card.append(gallery);
    }
    if (row["Biển số"]) {
      const score = Math.min(1, Math.max(0, Number(row["Điểm biển"] || 0)));
      const confidence = node("div", undefined, "confidence");
      confidence.append(node("span", "Điểm OCR tốt nhất"), node("strong", `${(score * 100).toFixed(1)}%`));
      const bar = node("progress"); bar.max = 1; bar.value = score;
      card.append(confidence, bar);
    }
    const actions = node("div", undefined, "actions");
    for (const [kind, label] of [["plate", "biển"], ["detected", "xe"]]) {
      if (!photos[kind]) continue;
      const view = node("button", `Xem ảnh ${label}`);
      view.onclick = () => showPhoto(photos[kind]);
      const download = node("a", `↓ Tải ảnh ${label}`, "action");
      download.href = `${photos[kind]}${photos[kind].includes("?") ? "&" : "?"}download=1`;
      actions.append(view, download);
    }
    card.append(actions);
    $("#cards").append(card);
  }
}
async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${response.status}`);
  }
  return response.json();
}
async function refresh() {
  try {
    const [status, rows] = await Promise.all([json("/api/status"), json("/api/events")]);
    $("#status").textContent = labels[status.status];
    $("#start").disabled = submitting || ["running", "stopping"].includes(status.status);
    $("#stop").disabled = status.status !== "running";
    $("#stop").textContent = status.status === "stopping" ? "Đang dừng…" : "■ Stop";
    $("#camera").textContent = status.camera || "Chọn nguồn video để bắt đầu";
    $("#frames").textContent = `Nguồn ${status.source_index || 0}/${status.source_count || 0} · Khung hình ${status.frame} / ${status.total || "—"}`;
    const diagnostic = status.diagnostic || {};
    const counts = status.result_counts || {confirmed: 0, low: 0};
    const totalResults = counts.confirmed + counts.low;
    $("#quality-summary").textContent = `Phiên hiện tại: ${counts.confirmed} đã chốt · ${counts.low} nhãn vàng`
      + (totalResults ? ` · Tỷ lệ chốt ${(counts.confirmed / totalResults * 100).toFixed(1)}%` : "")
      + (status.plates_only
        ? ` · Chỉ trả xe có biển số (đã ẩn ${status.hidden_by_plates_only || 0})`
        : "");
    $("#brand-coverage").textContent = status.brand_classes?.length
      ? `Model hãng có ${status.brand_classes.length} lớp: ${status.brand_classes.join(", ")}. Đồng thuận không bảo đảm đúng với hãng ngoài danh sách.`
      : "Chưa có thông tin lớp của model hãng.";
    $("#diagnostic").textContent = diagnostic.status
      ? `Biển số: ${diagnostic.status} · Mẫu xe: ${diagnostic.samples ?? 0} · Phát hiện biển: ${diagnostic.plate_detections ?? 0} · OCR đạt: ${diagnostic.accepted_reads ?? 0} · Chuỗi OCR: ${diagnostic.text || "—"} · Điểm: ${Number(diagnostic.score || 0).toFixed(2)}`
      : "Chưa có dữ liệu";
    if (diagnostic.frame_size) {
      $("#diagnostic").textContent += ` · Khung ${diagnostic.frame_size} · Track detector ${diagnostic.detector_boxes || 0} · Ngoài ROI ${diagnostic.roi_rejected_this_frame || 0} · Xe nhỏ ${diagnostic.vehicle_too_small_this_frame || 0}`;
    }
    const timing = diagnostic.timings_ms;
    if (timing) {
      $("#diagnostic").textContent += ` · Detect xe: ${timing.vehicle} ms · Biển/crop/logo: ${timing.plate_and_crops} ms · OCR: ${timing.ocr} ms · Batch: ${diagnostic.ocr_batch || 0} · Track đã chốt: ${diagnostic.confirmed_tracks || 0}/${diagnostic.active_tracks || 0}`;
    }
    const threshold = diagnostic.thresholds || {};
    $("#threshold-summary").textContent = threshold.detector_confidence != null
      ? `Ngưỡng: conf xe ${threshold.detector_confidence} · xe ≥ ${threshold.min_vehicle_width}px · detector biển ${threshold.plate_imgsz}px · vùng dưới ${threshold.plate_lower_fraction} · quét toàn xe mỗi ${threshold.plate_full_scan_interval} lần · biển ≥ ${threshold.min_plate_width}px · nét ≥ ${threshold.min_plate_sharpness} · ảnh ≥ ${threshold.min_image_quality} · detect ≥ ${threshold.min_evidence_detection} · OCR ≥ ${threshold.min_evidence_ocr} · chốt ${threshold.plate_confirm_reads}/${threshold.plate_confirm_score}/${threshold.plate_confirm_mean}/${threshold.plate_confirm_margin}. Các ngưỡng này chưa được hiệu chỉnh bằng ground truth.`
      : "Chưa có thông số ngưỡng.";
    const seenReasons = reasonParts(diagnostic.rejections, diagnostic.rejection_detail);
    $("#rejection-summary").textContent = seenReasons.length
      ? `Lý do loại theo track: ${seenReasons.join(" · ")}`
      : "Chưa có track bị loại ở khâu lõi.";
    const filterStats = {...(status.filter_stats || {}), hidden_by_recognized_only: diagnostic.hidden_by_recognized_only || 0};
    const filterReasons = reasonParts(filterStats);
    $("#filter-summary").textContent = filterReasons.length
      ? `Bỏ ở khâu lọc/gộp: ${filterReasons.join(" · ")}`
      : "Chưa có thống kê lọc kết quả.";
    $("#progress").value = status.status === "done" ? 100 : status.total ? status.frame / status.total * 100 : 0;
    if (status.error) $("#error").textContent = status.error;
    if (status.revision && status.revision !== revision) {
      revision = status.revision;
      if ($("#frame").hidden) $("#frame").src = `/api/stream?t=${Date.now()}`;
      $("#frame").hidden = false; $("#empty").hidden = true;
    }
    renderEvents(rows);
  } catch (error) { $("#error").textContent = `Không kết nối được máy chủ: ${error.message}`; }
  finally { setTimeout(refresh, 1000); }
}
$("#stop").onclick = async () => {
  $("#stop").disabled = true;
  try {
    await json("/api/stop", {method: "POST"});
    $("#status").textContent = labels.stopping;
  } catch (error) {
    $("#error").textContent = error.message;
    $("#stop").disabled = false;
  }
};
$("#source").onsubmit = async (event) => {
  event.preventDefault();
  submitting = true;
  $("#start").disabled = true; $("#error").textContent = "";
  try {
    await json("/api/analyze", {method: "POST", body: new FormData(event.target)});
    revision = -1; $("#frame").removeAttribute("src"); $("#frame").hidden = true; $("#empty").hidden = false;
  } catch (error) { $("#error").textContent = error.message; $("#start").disabled = false; }
  finally { submitting = false; }
};
refresh();
