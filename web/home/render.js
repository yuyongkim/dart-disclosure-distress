// Static sections: bound numbers and the publication record.
import { findModel, caught } from "./data.js";
import { int, fixed, percent, escapeHtml } from "./format.js";

export function bindings({ content, metrics, panel }){
  const p = content.paper;
  const base = findModel(metrics, 365, "sequence_logit");
  const head = findModel(metrics, content.headline_model.horizon, content.headline_model.model);
  return {
    panel_rows: int(panel.panel_rows),
    panel_firms: int(panel.panel_firms),
    market_features: int(panel.market_feature_count),
    price_rate: percent(panel.price_record_rate),
    event_rows: int(p.event_rows),
    event_firms: int(p.event_firms),
    n_test_365: int(base.n_test),
    n_train_365: int(base.n_train),
    pos_365: int(head.test_positives),
    caught_365: int(caught(head)),
    lift_365: fixed(head.top_10pct.lift, 2),
    base_rate_365: percent(base.test_positive_rate, 2),
    split: p.split,
    title_en: p.title_en,
    manuscript_id: p.manuscript_id,
    doi: p.doi,
    journal: p.journal,
    author: p.author,
    affiliation: p.affiliation,
    sample_period: p.sample_period
  };
}

export function applyBindings(values){
  document.querySelectorAll("[data-bind]").forEach(node => {
    const value = values[node.dataset.bind];
    if (value !== undefined) node.textContent = value;
  });
}

export function renderTimeline({ content }){
  document.getElementById("timeline").innerHTML = content.timeline.map(row => `
    <li class="reveal"><time>${escapeHtml(row.date)}</time><b>${escapeHtml(row.event)}</b><span>${escapeHtml(row.result)}</span></li>`).join("");
}
