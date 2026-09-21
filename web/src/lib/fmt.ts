// Number formatting. Pure functions over numbers that came from data.json; no fact lives here.
export const pct = (v: number, dp = 2): string => `${(v * 100).toFixed(dp)}%`;
export const pctNum = (v: number, dp = 2): string => (v * 100).toFixed(dp);
export const usd = (v: number, dp = 2): string => `$${v.toFixed(dp)}`;
export const int = (v: number): string => Math.round(v).toLocaleString("en-US");
export const dec = (v: number, dp = 1): string => v.toFixed(dp);
export const seconds = (v: number, dp = 1): string => `${v.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp })} s`;
export const hours = (v: number, dp = 2): string => `${v.toFixed(dp)} h`;
export const times = (v: number, dp = 1): string => `${v.toFixed(dp)}x`;
/** Percentage-point gap between two fractions, e.g. gap(0.9274, 0.6689) = "25.9". */
export const gap = (a: number, b: number, dp = 1): string => ((a - b) * 100).toFixed(dp);
export const ci = (x: { lo: number; hi: number }, dp = 1): string => `${pctNum(x.lo, dp)} to ${pctNum(x.hi, dp)}`;
export const clock = (secs: number): string => (secs >= 3600 ? hours(secs / 3600) : seconds(secs));
