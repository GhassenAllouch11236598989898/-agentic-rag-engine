# Billing Copilot: client demo guide

## Who this is for

Subscription software businesses whose support and finance teams repeatedly answer questions about invoices, renewal payments, plan changes, cancellations, and refunds.

Position the service as a company-specific billing knowledge assistant with human review. Its value is finding the correct policy quickly, drafting consistent guidance, and preserving evidence for the reviewer.

Customer-support automation is an active area of investment. Zendesk's [2026 customer-experience research](https://cxtrends.zendesk.com/) reports increasing expectations for faster responses. This supports the broad support use case; it is not proof of demand for this particular product or a guarantee of sales.

## A five-minute walkthrough

1. Open Billing Assistant. Explain: “These are fictional Northstar billing policies. The local demo retrieves actual documents and assembles source excerpts. Connected mode can use an AI provider.”
2. Choose **Switch from monthly to annual**. Show the policy-based draft, then click a citation and **Open original document**. Point out the exact sentence behind the answer.
3. Choose **Request an annual refund**. Explain that the assistant can state the policy; a billing employee must decide whether to approve an individual request.
4. Choose **An account-specific billing question**. Show that an invoice balance requires a billing-system lookup. The assistant does not invent a number.
5. Upload a sample policy provided with the client's permission. Ask about a unique rule from that policy. Show that the new source is searchable immediately.
6. Run **Quality Checks**. These measure expected policy retrieval, reference IDs, and escalation behavior. They do not measure factual accuracy or guarantee a correct reply.
7. Open **Overview** and adjust the pilot calculator with the client's numbers. Describe its output as estimated capacity value, not guaranteed cash savings.

## A simple pitch

“I build billing support assistants around your actual policies. Your team can prepare replies about invoices, subscriptions, and refunds, inspect the evidence, and escalate exceptions. We start with a small pilot and measure whether it saves review time while keeping answers consistent.”

## What to offer in a paid pilot

- One company workspace, a defined policy collection, and a limited set of billing question types.
- Import and cleanup of approved documents, plus a client-specific evaluation set.
- Configuration of the client's selected AI provider and a private deployment.
- Agent/admin access keys, documented operating procedures, and a handover session.
- Measurement of handling time, reviewer corrections, and escalation rate against a baseline.

Agree on scope and acceptance criteria before pricing. Hosting, model usage, document updates, integrations, and ongoing support should be explicit parts of the proposal.

## Be clear about the current boundaries

This application drafts responses. It does not connect to Stripe, QuickBooks, a bank, an email inbox, or a payment processor. It cannot confirm account balances, issue refunds, collect payments, or send messages. Those would require separately scoped integrations and access controls.

The local demo uses SQLite keyword retrieval and extractive answers. The connected mode uses PostgreSQL with pgvector and the configured AI provider. Do not present extractive demo answers as live model output, sample policies as real client data, or the calculator as measured ROI.
