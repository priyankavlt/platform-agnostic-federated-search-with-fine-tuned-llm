"use client";

// Generative UI: how each agent tool call is drawn in the chat.
import { useRenderTool } from "@copilotkit/react-core/v2";
import { z } from "zod";
import { useState } from "react";

type SqlResult = {
  title: string; sql: string; columns: string[]; rows: unknown[][]; row_count: number;
  truncated: boolean; systems: string[]; elapsed_ms: number;
};
type SearchResult = { hits: { id: string; system: string; title: string; url?: string; snippet: string }[] };

function parse<T>(result: unknown): T | null {
  if (typeof result !== "string") return (result as T) ?? null;
  try { return JSON.parse(result) as T; } catch { return null; }
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
  return String(v);
}

function toCsv(r: SqlResult): string {
  const esc = (v: unknown) => `"${fmt(v).replaceAll('"', '""')}"`;
  return [r.columns.map(esc).join(","), ...r.rows.map((row) => row.map(esc).join(","))].join("\n");
}

function ResultTable({ r }: { r: SqlResult }) {
  const [showSql, setShowSql] = useState(false);
  const download = () => {
    const url = URL.createObjectURL(new Blob([toCsv(r)], { type: "text/csv" }));
    Object.assign(document.createElement("a"), { href: url, download: `${r.title || "result"}.csv` }).click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="card">
      <div className="card-head">
        <strong>{r.title}</strong>
        <span className="muted">{r.row_count}{r.truncated ? "+" : ""} rows · {r.elapsed_ms} ms</span>
      </div>
      <div className="badges">{r.systems.map((s) => <span key={s} className="badge">{s}</span>)}</div>
      <div className="table-wrap">
        <table>
          <thead><tr>{r.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {r.rows.slice(0, 50).map((row, i) => (
              <tr key={i}>{row.map((v, j) => <td key={j} className={typeof v === "number" ? "num" : ""}>{fmt(v)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="card-foot">
        <button onClick={() => setShowSql(!showSql)}>{showSql ? "Hide" : "Show"} SQL</button>
        <button onClick={download}>Download CSV</button>
      </div>
      {showSql && <pre className="sql">{r.sql}</pre>}
    </div>
  );
}

export function ToolRenderers() {
  useRenderTool({
    name: "run_sql",
    parameters: z.object({ sql: z.string(), title: z.string() }),
    render: (props) => {
      if (props.status !== "complete") {
        return <div className="card pending">Querying systems: {props.parameters?.title ?? "…"}</div>;
      }
      const r = parse<SqlResult>(props.result);
      return r ? <ResultTable r={r} /> : <div className="card error">{String(props.result)}</div>;
    },
  }, []);

  useRenderTool({
    name: "search_text",
    parameters: z.object({ query: z.string(), k: z.number().optional() }),
    render: (props) => {
      if (props.status !== "complete") {
        return <div className="card pending">Searching docs, tickets and chat: “{props.parameters?.query ?? "…"}”</div>;
      }
      const r = parse<SearchResult>(props.result);
      return (
        <div className="card">
          <div className="card-head"><strong>Text search</strong><span className="muted">“{props.parameters.query}”</span></div>
          <ol className="hits">
            {(r?.hits ?? []).map((h) => (
              <li key={h.id}>
                <span className="badge">{h.system}</span> <code>{h.id}</code> {h.title}
                <div className="muted snippet">{h.snippet.slice(0, 180)}…</div>
              </li>
            ))}
          </ol>
        </div>
      );
    },
  }, []);

  return null;
}
