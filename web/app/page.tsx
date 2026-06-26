"use client";

import { CopilotChat, CopilotKitProvider } from "@copilotkit/react-core/v2";
import { ToolRenderers } from "@/components/ToolRenderers";

const EXAMPLES = [
  "Which enterprise customers have overdue invoices, and how many open tickets do they have?",
  "Total ARR by region, and how much of it is at risk or churn risk?",
  "Does billed MRR match the ARR in the CRM? Show the biggest mismatches.",
  "Why are Globex users bounced back to the login page?",
];

export default function Page() {
  return (
    <CopilotKitProvider runtimeUrl="/api/copilotkit">
      <ToolRenderers />
      <main className="shell">
        <aside className="side">
          <h1>fedsearch</h1>
          <p>One question, every system. Federated search &amp; aggregation across:</p>
          <ul className="systems">
            <li><b>CRM</b> Postgres: customers, deals</li>
            <li><b>Billing</b> Supabase/Postgres: subscriptions, invoices</li>
            <li><b>Support</b> Zendesk-style tickets</li>
            <li><b>Docs &amp; chat</b> runbooks, product docs, Slack-style chat</li>
          </ul>
          <p className="muted">Try:</p>
          <ul className="examples">
            {EXAMPLES.map((q) => <li key={q}>{q}</li>)}
          </ul>
        </aside>
        <section className="chat">
          <CopilotChat
            agentId="default"
            labels={{
              chatInputPlaceholder: "Ask across CRM, billing, support and docs...",
              welcomeMessageText: "Ask a question; I'll query each system and join the results.",
            }}
          />
        </section>
      </main>
    </CopilotKitProvider>
  );
}
