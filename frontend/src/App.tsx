import { ChangeEvent, useMemo, useState } from "react";

type Item = { name: string; qty: number; unit_price: number; total: number; confidence?: number };
type Charge = { type: string; amount: number };
type Receipt = { items: Item[]; charges: Charge[]; subtotal: number; grand_total: number };
type Extraction = {
  status: string;
  quality?: { guidance?: string[] };
  receipt?: Receipt;
  repair?: { warnings?: string[]; consistent?: boolean };
};
type SplitResult = {
  summary: string;
  people: { name: string; amount: string; explanation: string }[];
  warnings?: string[];
};

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

function money(value: number) {
  return Number(value || 0).toFixed(2);
}

function itemNeedsReview(item: Item) {
  const name = item.name.trim().toLowerCase();
  const expectedTotal = item.qty * item.unit_price;
  return !name
    || name === "unknown"
    || name === "unknown item"
    || !Number.isFinite(item.qty)
    || !Number.isFinite(item.unit_price)
    || !Number.isFinite(item.total)
    || item.qty <= 0
    || item.unit_price < 0
    || item.total < 0
    || Math.abs(expectedTotal - item.total) > 0.02
    || (item.confidence !== undefined && item.confidence < 95);
}

export default function App() {
  const [file, setFile] = useState<File>();
  const [receipt, setReceipt] = useState<Receipt>();
  const [status, setStatus] = useState("Add the receipt and people sharing the bill.");
  const [statusKind, setStatusKind] = useState<"neutral" | "ok" | "warn" | "error">("neutral");
  const [peopleCount, setPeopleCount] = useState(2);
  const [peopleNames, setPeopleNames] = useState(["", ""]);
  const [assignments, setAssignments] = useState<Record<number, string[]>>({});
  const [split, setSplit] = useState<SplitResult>();
  const [reviewWarnings, setReviewWarnings] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const people = useMemo(() => peopleNames.map((name) => name.trim()).filter(Boolean), [peopleNames]);

  function changePeopleCount(value: string) {
    const count = Math.max(1, Math.min(20, Number(value) || 1));
    setPeopleCount(count);
    setPeopleNames((current) => Array.from({ length: count }, (_, index) => current[index] || ""));
  }

  function changePerson(index: number, value: string) {
    setPeopleNames((current) => current.map((name, itemIndex) => itemIndex === index ? value : name));
  }

  function updateReceipt(next: Receipt) {
    setReceipt({ ...next, subtotal: Number(next.subtotal), grand_total: Number(next.grand_total) });
  }

  async function extract() {
    if (!file || people.length !== peopleCount) {
      setStatus("Add a name for every person before extracting the receipt.");
      setStatusKind("warn");
      return;
    }
    setBusy(true);
    setSplit(undefined);
    setStatus("Checking image quality and extracting…");
    setStatusKind("neutral");
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${API_URL}/extract`, { method: "POST", body });
      const payload = (await response.json()) as Extraction & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "Extraction failed");
      if (payload.status === "retake_photo") {
        setStatus((payload.quality?.guidance || ["Please take a clearer photo."]).join(" "));
        setStatusKind("warn");
        setReceipt(undefined);
        return;
      }
      if (!payload.receipt) throw new Error("The model could not produce a receipt. Please try again.");
      setReceipt(payload.receipt);
      setAssignments({});
      setReviewWarnings(payload.repair?.warnings || []);
      setStatus(payload.repair?.consistent ? "Receipt extracted and validated." : "Receipt extracted. Review the highlighted values.");
      setStatusKind(payload.repair?.consistent ? "ok" : "warn");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Something went wrong.");
      setStatusKind("error");
    } finally {
      setBusy(false);
    }
  }

  function updateItem(index: number, field: keyof Item, value: string) {
    if (!receipt) return;
    const numericValue = Number(value);
    const items = receipt.items.map((item, itemIndex) => {
      if (itemIndex !== index) return item;
      if (field === "name") return { ...item, name: value };
      if (field === "total") {
        return {
          ...item,
          total: numericValue,
          unit_price: item.qty > 0 ? numericValue / item.qty : 0,
        };
      }
      if (field === "qty") {
        return {
          ...item,
          qty: numericValue,
          total: numericValue * item.unit_price,
        };
      }
      if (field === "unit_price") {
        return {
          ...item,
          unit_price: numericValue,
          total: numericValue * item.qty,
        };
      }
      return item;
    });
    updateReceipt({ ...receipt, items });
  }

  function updateCharge(index: number, field: keyof Charge, value: string) {
    if (!receipt) return;
    const charges = receipt.charges.map((charge, chargeIndex) => chargeIndex === index
      ? { ...charge, [field]: field === "type" ? value : Number(value) }
      : charge);
    updateReceipt({ ...receipt, charges });
  }

  function toggleAssignment(index: number, name: string) {
    setAssignments((current) => {
      const existing = current[index] || [];
      return { ...current, [index]: existing.includes(name) ? existing.filter((item) => item !== name) : [...existing, name] };
    });
  }

  async function calculateSplit() {
    if (!receipt || people.length !== peopleCount) {
      setStatus("Add a name for every person before calculating the split.");
      setStatusKind("warn");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${API_URL}/split`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ receipt, people, assignments }),
      });
      const payload = await response.json() as SplitResult & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "Could not calculate split");
      setSplit(payload);
      setStatus("Split calculated. Review each person’s explanation.");
      setStatusKind("ok");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not calculate split.");
      setStatusKind("error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <header>
        <div className="logo">SS</div>
        <div><p className="eyebrow">RECEIPT FAIRNESS</p><h1>SplitSnap</h1></div>
        <span className="pill">India ready</span>
      </header>
      <section className="hero">
        <p className="eyebrow">SCAN · REVIEW · SPLIT</p>
        <h2>Make every rupee count.</h2>
        <p>Upload once, name your group, and share the bill without the awkward math.</p>
      </section>

      {!receipt && <section className="panel setup-panel">
        <div className="section-heading"><div><p className="eyebrow">STEP 1</p><h3>Set up your group</h3></div><span className="badge">Before scanning</span></div>
        <label>Number of people<input type="number" min="1" max="20" value={peopleCount} onChange={(event) => changePeopleCount(event.target.value)} /></label>
        <div className="people-grid">
          {peopleNames.map((name, index) => <label key={index}>Person {index + 1}<input placeholder={`Name ${index + 1}`} value={name} onChange={(event) => changePerson(index, event.target.value)} /></label>)}
        </div>
        <label className="upload"><input type="file" accept="image/*" capture="environment" onChange={(event: ChangeEvent<HTMLInputElement>) => setFile(event.target.files?.[0])} /><span className="upload-icon">↥</span><strong>{file ? file.name : "Upload or take a receipt photo"}</strong><small>Clear, well-lit photos work best</small></label>
        <button className="primary wide" onClick={extract} disabled={!file || busy}>{busy ? "Working…" : "Scan receipt"}</button>
        <div className={`status ${statusKind}`}>{status}</div>
      </section>}

      {receipt && <section className="workspace">
        <div className="workspace-title">
          <div><p className="eyebrow">STEP 2 · SHARED BILL</p><h3>Review items & assign the bill</h3></div>
          <button className="text-button" onClick={() => { setReceipt(undefined); setSplit(undefined); setStatus("Add the receipt and people sharing the bill."); }}>Start over</button>
        </div>
        {(receipt.items.some(itemNeedsReview) || reviewWarnings.length > 0) && <div className="review-banner"><strong>Review highlighted items in orange.</strong><span>Items below 95% estimated confidence need review. You can overwrite the total amount; price per piece is recalculated automatically.</span></div>}
        <div className="workspace-grid">
          <section className="panel item-panel">
            <div className="section-heading"><div><h3>Extracted items</h3><p className="muted">Tap names or amounts to correct them.</p></div><span className="badge">{receipt.items.length} items</span></div>
            {receipt.items.map((item, index) => {
              const flagged = itemNeedsReview(item);
              const confidence = item.confidence;
              return <article className={`item-card ${flagged ? "flagged" : ""}`} key={index}>
                <div className="item-card-top"><input className="item-name" value={item.name} onChange={(event) => updateItem(index, "name", event.target.value)} /><span className={`confidence ${flagged ? "low" : ""}`}>{confidence !== undefined ? `${confidence.toFixed(0)}% ${flagged ? "⚠ Review" : "✓ Confident"}` : flagged ? "⚠ Review" : "✓ Ready"}</span></div>
                <div className="item-fields"><label>Qty<input type="number" value={item.qty} onChange={(event) => updateItem(index, "qty", event.target.value)} /></label><label>Price / piece (₹)<input type="number" value={item.unit_price} onChange={(event) => updateItem(index, "unit_price", event.target.value)} /></label><label>Total (₹)<input type="number" value={item.total} onChange={(event) => updateItem(index, "total", event.target.value)} /></label></div>
                <div className="assignment-row"><span>Assigned to:</span>{people.map((person) => <button className={`chip ${(assignments[index] || []).includes(person) ? "selected" : ""}`} key={person} onClick={() => toggleAssignment(index, person)}>{(assignments[index] || []).includes(person) ? "✓ " : "+ "}{person}</button>)}</div>
              </article>;
            })}
            <button className="text-button" onClick={() => updateReceipt({ ...receipt, items: [...receipt.items, { name: "", qty: 1, unit_price: 0, total: 0 }] })}>+ Add item</button>
            <h3 className="subheading">Separate non-item charges</h3>
            {receipt.charges.map((charge, index) => <div className="charge-grid" key={index}><input value={charge.type} onChange={(event) => updateCharge(index, "type", event.target.value)} /><input type="number" value={charge.amount} onChange={(event) => updateCharge(index, "amount", event.target.value)} /></div>)}
            <button className="text-button" onClick={() => updateReceipt({ ...receipt, charges: [...receipt.charges, { type: "", amount: 0 }] })}>+ Add charge</button>
            <div className="totals"><label>Subtotal<input type="number" value={receipt.subtotal} onChange={(event) => updateReceipt({ ...receipt, subtotal: Number(event.target.value) })} /></label><label>Grand total<input type="number" value={receipt.grand_total} onChange={(event) => updateReceipt({ ...receipt, grand_total: Number(event.target.value) })} /></label></div>
          </section>
          <aside className="panel people-panel">
            <div className="section-heading"><div><h3>Named members</h3><p className="muted">Choose who shared each item.</p></div></div>
            {people.map((person) => <div className="member-card" key={person}><strong>{person}</strong><span>{Object.values(assignments).filter((names) => names.includes(person)).length} assigned item(s)</span>{split && <b>₹{split.people.find((entry) => entry.name === person)?.amount || "0.00"}</b>}</div>)}
            <button className="primary wide" onClick={calculateSplit} disabled={busy}>{busy ? "Calculating…" : "Calculate split"}</button>
          </aside>
        </div>
      </section>}

      {split && <section className="panel result-panel"><p className="eyebrow">YOUR SPLIT</p><h3>{split.summary}</h3>{split.people.map((person) => <article className="person-result" key={person.name}><div><strong>{person.name}</strong><p>{person.explanation}</p></div><b>₹{person.amount}</b></article>)}{split.warnings?.map((warning) => <div className="status warn" key={warning}>{warning}</div>)}</section>}
      <footer>Built for shared meals in India · Review before you pay</footer>
    </main>
  );
}
