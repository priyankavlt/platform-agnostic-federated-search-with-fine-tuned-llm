// CopilotKit runtime: forwards chat runs to the Python agent over AG-UI (no LLM calls happen here).
import { HttpAgent } from "@ag-ui/client";
import { CopilotRuntime, createCopilotRuntimeHandler } from "@copilotkit/runtime/v2";

const runtime = new CopilotRuntime({
  agents: {
    default: new HttpAgent({ url: process.env.FEDSEARCH_AGUI_URL ?? "http://127.0.0.1:8080/agui" }),
  },
});

const handler = createCopilotRuntimeHandler({ runtime, basePath: "/api/copilotkit" });

export const GET = handler;
export const POST = handler;
