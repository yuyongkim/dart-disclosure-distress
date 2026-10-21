// Interface fragments built from one Paper 4 case record: the disclosure sequence and the score card.
import { int, escapeHtml } from "./format.js";

const dateLabel = iso => iso.replaceAll("-", ".");

export function sequenceList(events, className = ""){
  return `<ul class="seq">${events.map(event => `
    <li class="${className}"><time datetime="${event.date}">${dateLabel(event.date)}</time><span class="chip ${event.code}">${event.code}</span><span>${escapeHtml(event.title)}</span></li>`).join("")}</ul>`;
}

// Share of test firm-years that scored higher, written as "상위 n%".
export function topShare(record){
  const share = record.share_scored_higher * 100;
  return share < 1 ? share.toFixed(1) : share.toFixed(0);
}

export function outcomeText(record, labels){
  const label = labels[record.outcome_label] || record.outcome_label;
  return `${dateLabel(record.outcome_date)} ${label} · 기준일 ${int(record.days_to_outcome)}일 뒤`;
}

function codeTally(events){
  const counts = events.reduce((acc, event) => ({ ...acc, [event.code]: (acc[event.code] || 0) + 1 }), {});
  const parts = Object.entries(counts).map(([code, count]) => `${code} ${count}건`);
  return parts.length ? ` · ${parts.join(", ")}` : "";
}

export function sequenceFrame(record){
  return `<span class="ui-head">${escapeHtml(record.firm)}<span>${dateLabel(record.anchor_date)} 기준 공시</span></span>${sequenceList(record.visible)}`;
}

export function scoreFrame(record, labels){
  const years = record.horizon_days === 365 ? "1년" : "2년";
  return `<span class="ui-head">${escapeHtml(record.firm)}<span>${years} 안 부실 점수</span></span>
    <span class="score">
      <span class="big">${record.predicted_risk.toFixed(3)}</span>
      <span class="row"><span>검증 표본 ${int(record.test_rows)}곳 가운데</span><b>상위 ${topShare(record)}%</b></span>
      <span class="meter" title="점수 순위"><span style="left:calc(${(record.share_scored_higher * 100).toFixed(2)}% - 2px)"></span></span>
      <span class="row"><span>모형이 본 공시</span><span>${record.visible.length}건${codeTally(record.visible)}</span></span>
      <span class="row"><span>실제 결과</span><span>${escapeHtml(outcomeText(record, labels))}</span></span>
    </span>`;
}
