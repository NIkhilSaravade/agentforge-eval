// Series colors and the per-problem ramps. Both were checked with the dataviz palette validator on the site's paper surface (#f2eee4):
//   categorical  #C4411D #2B58A6 #A56E00  -> all checks pass (worst adjacent CVD dE 21.3 protan / 22.0 tritan; contrast >= 3:1)
//   ordinal ramp (3 steps per hue, light to dark; class "never" is drawn hollow, not as a step) -> all checks pass:
//     vermilion  #d98f77 #ce6749 #c4411d     (blend alphas .55 .78 1.0)
//     cobalt     #859cc2 #5779b4 #2b58a6     (blend alphas .55 .78 1.0)
//     ochre      #c4a15b #b4882e #a56e00     (blend alphas .60 .80 1.0; a lighter light-end failed the 2:1 floor)
// A first attempt at five equal opacity steps FAILED (light end under 2:1, dark steps too close), which is why there are three.
import type { ArmId } from "./data";

export const RAMP: Record<ArmId, readonly [string, string, string]> = {
  self: ["#d98f77", "#ce6749", "#c4411d"],
  greedy: ["#d98f77", "#ce6749", "#c4411d"],
  haiku: ["#859cc2", "#5779b4", "#2b58a6"],
  opus: ["#c4a15b", "#b4882e", "#a56e00"],
};

export const HEX: Record<ArmId, string> = { self: "#c4411d", greedy: "#c4411d", haiku: "#2b58a6", opus: "#a56e00" };

/** never -> null (hollow); some but under half -> step 0; half or more but not all -> step 1; every sample -> step 2. */
export function rampStep(c: number, n: number): 0 | 1 | 2 | null {
  if (c === 0) return null;
  if (c === n) return 2;
  return c / n < 0.5 ? 0 : 1;
}
