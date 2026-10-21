// Loads the front-page content file and the Paper 4 result files copied under ./paper4/.
const SOURCES = {
  content: "./home/content.json",
  metrics: "./paper4/market_model_metrics.json",
  robust: "./paper4/market_robust_model_metrics.json",
  panel: "./paper4/market_model_panel_summary.json",
  cases: "./paper4/cases.json"
};

async function loadJson(url){
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

export async function loadAll(){
  const entries = await Promise.all(
    Object.entries(SOURCES).map(async ([key, url]) => [key, await loadJson(url)])
  );
  return Object.fromEntries(entries);
}

export function findModel(rows, horizon, model){
  return rows.find(row => Number(row.horizon_days) === horizon && row.model === model && row.status === "ok");
}

// Number of distress cases inside the top decile, from recall and test positives.
export function caught(row){
  return Math.round(Number(row.top_10pct.recall) * Number(row.test_positives));
}
