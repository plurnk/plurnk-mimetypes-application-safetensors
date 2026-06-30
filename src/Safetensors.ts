import { BaseHandler } from "@plurnk/plurnk-mimetypes";
import type { HandlerContent, MimeSymbol } from "@plurnk/plurnk-mimetypes";

// application/x-safetensors (safetensors model weights) handler — Tier 4
// binary, no dep.
//
// A safetensors file is `<uint64 header-length><JSON header><raw tensor bytes>`.
// The JSON header is a tensor inventory: `{ name: { dtype, shape, data_offsets },
// …, "__metadata__": {…} }`. This handler reads ONLY that header — never the
// weight bytes — so an agent can answer "what's in this checkpoint" (tensor
// names, dtypes, shapes) without loading the file.
//
// Symbols are the tensor names; deepJson is the `{ name: { dtype, shape } }`
// view (a jsonpath target); toText renders `name: dtype [shape]` lines (the
// regex/glob + embed source). `data_offsets` are dropped from the model-facing
// views as byte-layout noise. References are not applicable.
export default class Safetensors extends BaseHandler {
    override extractRaw(content: HandlerContent): MimeSymbol[] {
        const header = readSafetensors(toBytes(content));
        if (!header) return [];
        return header.tensors.map((t, i) => ({ name: t.name, kind: "field", line: i + 1, endLine: i + 1 }));
    }

    override deepJson(content: HandlerContent): unknown {
        const header = readSafetensors(toBytes(content));
        if (!header) return null;
        const out: Record<string, unknown> = {};
        for (const t of header.tensors) out[t.name] = { dtype: t.dtype, shape: t.shape };
        if (Object.keys(header.metadata).length > 0) out.__metadata__ = header.metadata;
        return out;
    }

    override extent(content: HandlerContent): number {
        const header = readSafetensors(toBytes(content));
        return header ? header.tensors.length : 0;
    }

    override validate(content: HandlerContent): void {
        if (!readSafetensors(toBytes(content))) throw new Error("not a valid safetensors file (bad header)");
    }

    protected override toText(content: HandlerContent): string {
        const header = readSafetensors(toBytes(content));
        if (!header) return "";
        return header.tensors.map((t) => `${t.name}: ${t.dtype} [${t.shape.join(", ")}]`).join("\n");
    }
}

export interface SafetensorsTensor {
    name: string;
    dtype: string;
    shape: number[];
}

export interface SafetensorsHeader {
    tensors: SafetensorsTensor[];
    metadata: Record<string, string>;
}

export function readSafetensors(bytes: Uint8Array): SafetensorsHeader | null {
    if (bytes.length < 8) return null;
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const len = Number(dv.getBigUint64(0, true));
    if (len <= 0 || 8 + len > bytes.length) return null;

    let obj: unknown;
    try {
        obj = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + len)));
    } catch {
        return null;
    }
    if (typeof obj !== "object" || obj === null) return null;

    const tensors: SafetensorsTensor[] = [];
    let metadata: Record<string, string> = {};
    for (const [name, info] of Object.entries(obj as Record<string, unknown>)) {
        if (name === "__metadata__") {
            if (typeof info === "object" && info !== null) metadata = info as Record<string, string>;
            continue;
        }
        // A tensor entry without a string dtype and array shape is a malformed
        // header — fail the whole parse (→ validate() throws) rather than
        // coercing to `""`/`[]` and rendering a phantom tensor.
        if (typeof info !== "object" || info === null) return null;
        const rec = info as Record<string, unknown>;
        if (typeof rec.dtype !== "string" || !Array.isArray(rec.shape)) return null;
        tensors.push({
            name,
            dtype: rec.dtype,
            shape: rec.shape.filter((n): n is number => typeof n === "number"),
        });
    }
    return { tensors, metadata };
}

function toBytes(content: HandlerContent): Uint8Array {
    return typeof content === "string" ? new TextEncoder().encode(content) : content;
}
