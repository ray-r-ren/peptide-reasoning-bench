const MODE_LABELS = {
  base: "Base",
  tools_high_reasoning: "Tools + high reasoning",
  baseline: "Baseline",
  sanity_check: "Sanity check"
};

const state = {
  rows: [],
  filter: "competitive",
  sortKey: "coverage_adjusted_score",
  sortDir: "desc"
};

function numberOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function coverageFraction(row) {
  const coverage = row.coverage || {};
  let attempted = 0;
  let completed = 0;
  Object.values(coverage).forEach((item) => {
    attempted += Number(item.attempted || 0);
    completed += Number(item.completed || 0);
  });
  if (!attempted) {
    attempted = Number(row.total_cases_attempted || 0);
    completed = Number(row.total_cases_completed || 0);
  }
  return attempted ? Math.max(0, Math.min(1, completed / attempted)) : 0;
}

function metric(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function statusTags(row) {
  const tags = [];
  if (row.row_status && row.row_status.startsWith("excluded")) tags.push(row.row_status);
  if (
    Number(row.api_error_count || 0) ||
    Number(row.invalid_json_count || 0) ||
    Number(row.unresolved_provider_error_count || 0) ||
    Number(row.unresolved_invalid_json_count || 0)
  ) tags.push("errors");
  if (row.coverage < 1) tags.push("incomplete");
  if (row.is_baseline) tags.push("baseline");
  const modelId = String(row.model_id || "").toLowerCase();
  const oracleLike = modelId.includes("oracle") && !modelId.includes("non_oracle") && !modelId.includes("non-oracle");
  if (row.competitive === false || oracleLike || row.mode === "sanity_check") tags.push("non-competitive");
  if (!tags.length) tags.push("clean");
  return tags;
}

function normalize(rawRows) {
  const rows = rawRows.map((row) => {
    const coverage = coverageFraction(row);
    const score = numberOrNull(row.coverage_adjusted_score);
    const mean = numberOrNull(row.mean_score);
    const normalized = {
      rank: null,
      model_id: row.model_id,
      provider: row.provider,
      mode: row.mode,
      mode_label: MODE_LABELS[row.mode] || row.mode || "",
      mean_score: mean,
      coverage_adjusted_score: score === null && mean !== null ? mean * coverage : score,
      human_effect: metric(row.human_effect_category_macro_f1),
      binding_rank: metric(row.binding_rank_kendall) ?? metric(row.binding_rank_spearman),
      pose: metric(row.pose_contact_f1),
      structure: metric(row.structure_score) ?? "not computed",
      coverage,
      api_error_count: Number(row.api_error_count || 0),
      invalid_json_count: Number(row.invalid_json_count || 0),
      unresolved_provider_error_count: Number(row.unresolved_provider_error_count || 0),
      unresolved_invalid_json_count: Number(row.unresolved_invalid_json_count || 0),
      fallback_prediction_count: Number(row.fallback_prediction_count || 0),
      row_status: row.row_status || "",
      scored: row.scored === true,
      is_baseline: Boolean(row.is_baseline),
      competitive: row.competitive === true && row.row_status === "clean_completed",
      status: ""
    };
    normalized.status = statusTags(normalized).join(", ");
    return normalized;
  });
  const ranked = rows
    .filter((row) => row.competitive && row.scored && row.row_status === "clean_completed")
    .sort((a, b) => (b.coverage_adjusted_score ?? -Infinity) - (a.coverage_adjusted_score ?? -Infinity));
  ranked.forEach((row, index) => {
    row.rank = index + 1;
  });
  return rows;
}

function formatScore(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value.toFixed(3);
  return value || "not computed";
}

function formatPercent(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function filteredRows() {
  return state.rows.filter((row) => {
    if (state.filter === "all") return true;
    if (state.filter === "base") return row.mode === "base";
    if (state.filter === "tools_high_reasoning") return row.mode === "tools_high_reasoning";
    if (state.filter === "baselines") return row.is_baseline || row.competitive === false;
    return row.competitive && row.scored && row.row_status === "clean_completed";
  });
}

function sortRows(rows) {
  const dir = state.sortDir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const left = a[state.sortKey];
    const right = b[state.sortKey];
    if (typeof left === "number" || typeof right === "number") {
      return ((left ?? -Infinity) - (right ?? -Infinity)) * dir;
    }
    return String(left ?? "").localeCompare(String(right ?? "")) * dir;
  });
}

function renderStatus(status) {
  return `<span class="status">${status.split(", ").map((tag) => `<span class="tag ${tag}">${tag}</span>`).join("")}</span>`;
}

function render() {
  const rows = sortRows(filteredRows());
  document.querySelector("#summary").textContent = `${rows.length} rows shown. Default score: coverage-adjusted mean.`;
  const body = document.querySelector("#leaderboard-table tbody");
  body.innerHTML = rows.map((row) => `
    <tr>
      <td>${row.rank ?? "—"}</td>
      <td>${row.model_id}</td>
      <td>${row.mode_label}</td>
      <td>${formatScore(row.mean_score)}</td>
      <td>${formatScore(row.coverage_adjusted_score)}</td>
      <td>${formatScore(row.human_effect)}</td>
      <td>${formatScore(row.binding_rank)}</td>
      <td>${formatScore(row.pose)}</td>
      <td>${formatScore(row.structure)}</td>
      <td>${formatPercent(row.coverage)}</td>
      <td>${row.api_error_count}</td>
      <td>${row.invalid_json_count}</td>
      <td>${renderStatus(row.status)}</td>
    </tr>
  `).join("");
}

document.querySelectorAll("[data-filter]").forEach((button) => {
  button.addEventListener("click", () => {
    state.filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    render();
  });
});

document.querySelectorAll("th[data-sort]").forEach((header) => {
  header.addEventListener("click", () => {
    const key = header.dataset.sort;
    state.sortDir = state.sortKey === key && state.sortDir === "desc" ? "asc" : "desc";
    state.sortKey = key;
    render();
  });
});

fetch("leaderboard.json")
  .then((response) => response.json())
  .then((rows) => {
    state.rows = normalize(rows);
    render();
  })
  .catch((error) => {
    document.querySelector("#summary").textContent = `Could not load leaderboard.json: ${error.message}`;
  });
