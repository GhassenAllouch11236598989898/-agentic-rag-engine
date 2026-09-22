"""Transparent workflow checks, not a synthetic model leaderboard."""

from datetime import datetime, timezone

CASES = [
    (
        "Annual upgrade credit",
        "Will an annual upgrade credit the unused monthly payment?",
        "Subscription Changes",
        False,
    ),
    (
        "Invoice correction",
        "How can we correct the company name on an issued invoice?",
        "Invoices And Billing Details",
        False,
    ),
    (
        "Failed payment retry",
        "When does a failed subscription payment retry and how long is the grace period?",
        "Payment Failures",
        False,
    ),
    (
        "First annual refund",
        "We purchased our first annual subscription 8 days ago. Can we request a refund?",
        "Refund Policy",
        False,
    ),
    (
        "Cancel renewal",
        "How can we cancel renewal and how long does our access continue?",
        "Subscription Changes",
        False,
    ),
    (
        "Account balance escalation",
        "What is the exact outstanding balance on invoice INV-2026-8841?",
        None,
        True,
    ),
]


async def run_checks(service, owner):
    cases = []
    for name, question, expected, escalate in CASES:
        result = await service.draft(owner, question, persist=False)
        titles = [s["title"] for s in result["sources"]]
        retrieval_ok = expected in titles if expected else True
        behavior_ok = (
            result["status"] == "needs_escalation"
            if escalate
            else result["status"] == "draft"
        )
        references_ok = (
            result["citation_check"]["references_valid"] if not escalate else True
        )
        cases.append(
            {
                "name": name,
                "question": question,
                "passed": retrieval_ok and behavior_ok and references_ok,
                "retrieval_ok": retrieval_ok,
                "behavior_ok": behavior_ok,
                "references_ok": references_ok,
                "expected_source": expected,
                "actual_sources": titles,
                "message": result["message"],
                "elapsed_ms": result["elapsed_ms"],
            }
        )
    return {
        "cases": cases,
        "passed": sum(c["passed"] for c in cases),
        "total": len(cases),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "method": "Checks billing-policy retrieval, reference IDs, and escalation behavior. Human review is still required for answer accuracy.",
    }
