// Results table: pick a horizon and outcome definition, then a model, and read the top-decile result.
import { findModel, caught } from "./data.js";
import { int, fixed, escapeHtml } from "./format.js";

const VIEWS = [
  { id: "h365", label: "1년 안 부실", sub: "기본 정의", horizon: 365, source: "metrics" },
  { id: "h730", label: "2년 안 부실", sub: "기본 정의", horizon: 730, source: "metrics" },
  { id: "r365", label: "1년 안 부실", sub: "엄격한 정의", horizon: 365, source: "robust" },
  { id: "r730", label: "2년 안 부실", sub: "엄격한 정의", horizon: 730, source: "robust" }
];
const FINAL = "combined_market_calibrated_random_forest";

function readout(view, entry, row){
  const years = view.horizon === 365 ? "1년" : "2년";
  return `${entry.name} 모형으로 점수가 높은 상위 10%만 검토하면, ${years} 안에 부실이 난 ${int(row.test_positives)}곳 가운데 ${caught(row)}곳이 그 안에 있습니다. 무작위로 같은 수를 고를 때보다 ${fixed(row.top_10pct.lift, 2)}배 많고, ROC AUC는 ${fixed(row.roc_auc, 4)}입니다.`;
}

function rowHtml(entry, row, selected){
  return `<tr class="${entry.model === FINAL ? "final" : ""}" data-model="${entry.model}" aria-selected="${selected}" tabindex="0">
    <td>${escapeHtml(entry.name)}<small>${escapeHtml(entry.note)}</small></td>
    <td class="num">${caught(row)} / ${int(row.test_positives)}곳</td>
    <td class="num lift">${fixed(row.top_10pct.lift, 2)}배</td>
  </tr>`;
}

export function mountExplorer(data){
  const tabs = document.getElementById("viewTabs");
  const body = document.getElementById("modelRows");
  const out = document.getElementById("readout");
  document.getElementById("robustNote").textContent = data.content.paper.robust_note;
  const entries = data.content.explorer_models;
  let state = { view: VIEWS[0], model: FINAL };

  tabs.innerHTML = VIEWS.map(view => `
    <button type="button" role="tab" class="view" id="tab-${view.id}" data-view="${view.id}" aria-selected="false">
      ${view.label}<small>${view.sub}</small>
    </button>`).join("");

  function draw(){
    const rows = data[state.view.source];
    tabs.querySelectorAll(".view").forEach(button => {
      const active = button.dataset.view === state.view.id;
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    const available = entries
      .map(entry => ({ entry, row: findModel(rows, state.view.horizon, entry.model) }))
      .filter(item => item.row);
    body.innerHTML = available.map(({ entry, row }) => rowHtml(entry, row, entry.model === state.model)).join("");
    const selected = available.find(item => item.entry.model === state.model) || available[available.length - 1];
    out.textContent = readout(state.view, selected.entry, selected.row);
  }

  function pickView(index){
    state = { ...state, view: VIEWS[(index + VIEWS.length) % VIEWS.length] };
    draw();
  }

  tabs.addEventListener("click", event => {
    const button = event.target.closest(".view");
    if (button) pickView(VIEWS.findIndex(view => view.id === button.dataset.view));
  });
  tabs.addEventListener("keydown", event => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    pickView(VIEWS.indexOf(state.view) + step);
    document.getElementById(`tab-${state.view.id}`).focus();
  });
  const pickRow = target => {
    const row = target.closest("tr[data-model]");
    if (!row) return;
    state = { ...state, model: row.dataset.model };
    draw();
  };
  body.addEventListener("click", event => pickRow(event.target));
  body.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); pickRow(event.target); }
  });
  draw();
}
