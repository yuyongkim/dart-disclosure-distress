// Number formatting shared by the front-page renderers.
const integer = new Intl.NumberFormat("ko-KR");

export const int = value => integer.format(Number(value));
export const fixed = (value, digits) => Number(value).toFixed(digits);
export const percent = (value, digits = 1) => `${(Number(value) * 100).toFixed(digits)}%`;

// Splits a figure into whole part and decimal part for the decimal-aligned ledger.
export function splitDecimal(text){
  const index = text.indexOf(".");
  return index < 0 ? [text, ""] : [text.slice(0, index), text.slice(index)];
}

export function escapeHtml(value){
  return String(value).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
}
