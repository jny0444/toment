const THRESHOLD = 0.7;

const LABELS = [
  ["toxic", "Toxic"],
  ["severe_toxic", "Severe"],
  ["obscene", "Obscene"],
  ["threat", "Threat"],
  ["insult", "Insult"],
  ["identity_hate", "Identity"],
];

const state = {
  archive: null,
  collectionId: "political",
  sliceId: "early",
  nodeId: null,
  slip: null,
};

const $ = (id) => document.getElementById(id);

async function boot() {
  $("stage").innerHTML = `<p class="empty">Opening the archive…</p>`;
  const response = await fetch("/api/archive");
  state.archive = await response.json();
  render();
}

function collection() {
  return state.archive.collections.find((item) => item.id === state.collectionId);
}

function snapshot() {
  const current = collection();
  return current.slices.find((item) => item.id === state.sliceId) || current.slices[0];
}

function replaceCollection(next) {
  const index = state.archive.collections.findIndex((item) => item.id === next.id);
  state.archive.collections[index] = next;
  render();
}

function render() {
  renderMetrics();
  renderIndex();
  renderStage();
  renderDesk();
}

function renderMetrics() {
  const metrics = state.archive.metrics;
  const cards = [
    ["English hold-out", metrics.english_holdout.macro_auc],
    ["Multilingual", metrics.multilingual_challenge.macro_auc],
    ["N-gram only", metrics.multilingual_ngram_only.macro_auc],
  ];
  $("metrics").innerHTML = cards
    .map(
      ([label, value]) =>
        `<div class="metric"><span>${label} AUC</span><strong>${value.toFixed(2)}</strong></div>`
    )
    .join("");
}

function renderIndex() {
  const cards = state.archive.collections
    .map(
      (item) => `<button class="collection ${item.id === state.collectionId ? "active" : ""}" data-id="${item.id}">
        <small>${item.accession}</small>
        <strong>${item.title}</strong>
        <em>${item.kicker}</em>
      </button>`
    )
    .join("");
  const current = collection();
  $("index").innerHTML = `
    <p class="eyebrow">Collections</p>
    ${cards}
    <p class="summary">${current.summary}</p>
    <details class="method">
      <summary>Reading method</summary>
      <p>Each deposit is a graph. People are nodes, replies, reshares, or co-authorships are edges. Louvain splits the snapshot into communities. The next deposit is compared with the last one for birth, growth, merge, split, shrink, and dissolve.</p>
      <p>Under the graph, a PyTorch model scores each comment on six labels. It is trained on English only. Shared lexicon features are the stand-in for the cross-lingual transfer that XLM-RoBERTa and multilingual DistilBERT provide in the full experiment. ROC-AUC is the ranking metric.</p>
    </details>
    <div class="row">
      <button class="ghost" id="reset" type="button">Reset deposits</button>
    </div>`;
  $("index").querySelectorAll(".collection").forEach((button) => {
    button.addEventListener("click", () => {
      state.collectionId = button.dataset.id;
      state.sliceId = collection().slices[0].id;
      state.nodeId = null;
      state.slip = null;
      render();
    });
  });
  $("reset").addEventListener("click", async () => {
    const response = await fetch("/api/reset", { method: "POST" });
    state.archive = await response.json();
    state.nodeId = null;
    state.slip = null;
    render();
  });
}

function renderStage() {
  const current = collection();
  const slice = snapshot();
  const film = current.slices
    .map(
      (item) =>
        `<button type="button" data-slice="${item.id}" class="${item.id === slice.id ? "active" : ""}">${item.label}</button>`
    )
    .join("");
  const stats = [
    ["Nodes", slice.stats.nodes],
    ["Edges", slice.stats.edges],
    ["Communities", slice.stats.communities],
    ["Modularity", slice.stats.modularity.toFixed(2)],
    ["Toxic share", `${Math.round(slice.stats.toxic_share * 100)}%`],
  ]
    .map(([label, value]) => `<div class="stat"><span>${label}</span><b>${value}</b></div>`)
    .join("");
  const events = slice.events
    .map(
      (event) =>
        `<div class="stamp ${event.type}">${event.type}<small>${event.detail}</small></div>`
    )
    .join("");
  $("stage").innerHTML = `
    <div class="stage-head">
      <div>
        <h2>${current.title}</h2>
        <p class="when">${current.accession} · ${slice.when}</p>
      </div>
      <div class="film">${film}</div>
    </div>
    <p class="note">${slice.note}</p>
    <div class="graph-wrap">${drawGraph(slice, current.palette)}</div>
    <div class="legend">
      <span><i class="swatch" style="background:#5c564c"></i>reply</span>
      <span><i class="swatch" style="background:#1f4e79"></i>co-author</span>
      <span><i class="swatch" style="background:#8a6232"></i>bridge</span>
      <span><i class="swatch" style="background:#9a3412"></i>reshare</span>
      <span><i class="swatch cut"></i>cut by moderation</span>
    </div>
    <div class="stats">${stats}</div>
    <div class="events">${events}</div>`;
  $("stage").querySelectorAll(".film button").forEach((button) => {
    button.addEventListener("click", () => {
      state.sliceId = button.dataset.slice;
      state.nodeId = null;
      state.slip = null;
      render();
    });
  });
  $("stage").querySelectorAll(".node").forEach((node) => {
    node.addEventListener("click", () => {
      state.nodeId = node.dataset.id;
      render();
    });
  });
}

function drawGraph(slice, palette) {
  const width = 800;
  const height = 460;
  const edges = slice.edges
    .map((edge) => {
      const source = slice.nodes.find((node) => node.id === edge.source);
      const target = slice.nodes.find((node) => node.id === edge.target);
      const x1 = source.x * width;
      const y1 = source.y * height;
      const x2 = target.x * width;
      const y2 = target.y * height;
      const color = { reply: "#5c564c", coauthor: "#1f4e79", bridge: "#8a6232", reshare: "#9a3412" }[edge.kind] || "#5c564c";
      const dash = edge.cut || edge.kind === "reshare" ? 'stroke-dasharray="6 5"' : "";
      const opacity = edge.cut ? 0.35 : 0.85;
      return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${edge.cut ? "#b4331a" : color}" stroke-width="${edge.kind === "bridge" ? 2.6 : 1.6}" ${dash} opacity="${opacity}"></line>`;
    })
    .join("");
  const nodes = slice.nodes
    .map((node) => {
      const cx = node.x * width;
      const cy = node.y * height;
      const fill = node.community < 0 ? "#8d8478" : palette[node.community % palette.length];
      const ring = node.toxic ? `stroke="#b4331a" stroke-width="3"` : `stroke="#1c1915" stroke-width="1.2"`;
      const selected = node.id === state.nodeId ? "selected" : "";
      return `<g class="node ${selected}" data-id="${node.id}" transform="translate(${cx} ${cy})">
        <circle r="${node.id === state.nodeId ? 16 : 13}" fill="${fill}" ${ring}></circle>
        <text y="28" text-anchor="middle">${escapeHtml(node.name.split(" ")[0])}</text>
      </g>`;
    })
    .join("");
  return `<svg class="graph" viewBox="0 0 ${width} ${height}" role="img" aria-label="Community graph">${edges}${nodes}</svg>`;
}

function renderDesk() {
  const slice = snapshot();
  const selected = slice.nodes.find((node) => node.id === state.nodeId);
  const comments = slice.comments.filter((comment) => !selected || comment.author === selected.id);
  const clips = comments
    .map((comment) => {
      const person = slice.nodes.find((node) => node.id === comment.author);
      const verdict = comment.toxic ? "Toxic" : "Civil";
      return `<button class="clip ${comment.toxic ? "hot" : ""}" data-author="${comment.author}">
        <header><span>${escapeHtml(person.name)} · ${comment.lang}</span><span class="verdict ${comment.toxic ? "" : "civil"}">${verdict}</span></header>
        <p>${escapeHtml(comment.text)}</p>
      </button>`;
    })
    .join("");
  const authorOptions = slice.nodes
    .map((node) => `<option value="${node.id}" ${node.id === state.nodeId ? "selected" : ""}>${escapeHtml(node.name)}</option>`)
    .join("");
  $("desk").innerHTML = `
    <div class="desk-head">
      <h2>${selected ? selected.name : "Thread"}</h2>
      <p>${selected ? selected.role : `${slice.comments.length} archived comments`}</p>
    </div>
    <div class="comments">${clips || `<p class="empty">No comments for this person in the deposit.</p>`}</div>
    <form class="composer" id="composer">
      <label for="author">File a comment as</label>
      <select id="author">${authorOptions}</select>
      <label for="draft">Comment</label>
      <textarea id="draft" placeholder="English or another language. The model was trained on English only."></textarea>
      <div class="row">
        <button class="score-btn" type="submit">Score and file</button>
        <button class="ghost danger" id="moderate" type="button">Cut toxic paths</button>
      </div>
      <div id="slip">${state.slip || ""}</div>
    </form>`;
  $("desk").querySelectorAll(".clip").forEach((clip) => {
    clip.addEventListener("click", () => {
      state.nodeId = clip.dataset.author;
      render();
    });
  });
  $("author").addEventListener("change", (event) => {
    state.nodeId = event.target.value;
    render();
  });
  $("composer").addEventListener("submit", fileComment);
  $("moderate").addEventListener("click", moderate);
}

async function fileComment(event) {
  event.preventDefault();
  const text = $("draft").value.trim();
  if (!text) return;
  const author = $("author").value;
  const response = await fetch(`/api/collections/${state.collectionId}/slices/${state.sliceId}/comments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ author, text }),
  });
  const next = await response.json();
  const slice = next.slices.find((item) => item.id === state.sliceId);
  const filed = slice.comments[slice.comments.length - 1];
  state.slip = slipHtml(filed);
  state.nodeId = author;
  replaceCollection(next);
}

async function moderate() {
  const response = await fetch(`/api/collections/${state.collectionId}/slices/${state.sliceId}/moderate`, {
    method: "POST",
  });
  replaceCollection(await response.json());
}

function slipHtml(comment) {
  const bars = LABELS.map(([key, label]) => {
    const score = comment.scores[key];
    const width = Math.round(score * 100);
    return `<div class="bar"><span>${label}</span><div class="track"><i class="${score >= THRESHOLD ? "warn" : ""}" style="width:${width}%"></i></div><b>${score.toFixed(2)}</b></div>`;
  }).join("");
  const ngram = comment.ngram_scores.toxic.toFixed(2);
  const how = comment.signals.lexicon ? "lexicon slot fired" : "n-grams only";
  return `<div class="slip">Filed. N-gram-only toxic score ${ngram}. Decision uses the full head (${how}).</div><div class="bars">${bars}</div>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

boot();
