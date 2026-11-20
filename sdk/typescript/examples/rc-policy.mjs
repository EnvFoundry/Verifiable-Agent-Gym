import { serveBatchPolicy } from "../dist/index.js";

// The noiseless RC model can be identified from one complex response.
// This is an analytic policy, not a model-performance result.
await serveBatchPolicy(({ observations }) => observations.map(({ slot, observation }) => {
  if (observation.type === "start") return { slot, action: { tool: "measure_response", arguments: { frequency_hz: 1000 } } };
  if (observation.type !== "tool_result" || observation.tool !== "measure_response") throw new Error("RC policy expected a measured complex response");
  const { frequency_hz: f, real: real, imag: imaginary } = observation.result;
  const cutoff = -f * real / imaginary;
  const gain = real * (1 + (f / cutoff) ** 2);
  return { slot, action: { submit: { gain, cutoff_hz: cutoff } }, policy_data: { method: "one-complex-measurement" } };
}));
