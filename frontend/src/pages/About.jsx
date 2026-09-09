export default function About() {
  return (
    <div className="page">
      <div className="about-content">
        <h2>About Veriscribe</h2>
        <p>
          Veriscribe is a Fact Knowledge Layer for financial documents. It ingests PDFs —
          annual reports, earnings presentations, prospectuses — and automatically extracts
          structured facts from text, tables, and charts. Every fact is grounded to its
          precise source: the exact quote, page number, and visual element it came from.
        </p>

        <h2>What counts as a fact?</h2>
        <p>
          A fact is a discrete, attributable claim: a revenue figure for a specific period,
          a percentage margin, a unit count, or a qualitative statement that can be compared
          across documents. Each extracted fact carries a raw attribute name (e.g. "Revenue from
          services"), a temporal scope (e.g. FY24), a normalized value, and a confidence score.
        </p>

        <h2>How are relationships decided?</h2>
        <p>
          When a new fact is ingested, it is compared against semantically similar facts
          already in the knowledge base. A deterministic reconciler handles clear cases —
          two facts with numerically identical values (within 1% rounding tolerance) and
          equivalent temporal scopes are labelled <strong>corroborates</strong> without an
          LLM call. Genuinely ambiguous or contradictory pairs are routed to an LLM judge,
          which reads the full evidence context for both facts and returns one of four
          verdicts: <strong>corroborates</strong>, <strong>contradicts</strong>,{' '}
          <strong>reconciled</strong> (same metric, different period/scope), or{' '}
          <strong>ambiguous</strong>. Every verdict includes a plain-English justification.
        </p>

        <h2>Tech stack</h2>
        <p>
          Backend: Python · pdfplumber · OpenAI-compatible LLM API · ChromaDB (vector store) ·
          SQLite (persistent store) · FastAPI.<br />
          Frontend: React · Vite · react-router · plain CSS design system extracted from Stitch mockups.<br />
          Built for Superjoin VIT 2026.
        </p>
      </div>
    </div>
  );
}
