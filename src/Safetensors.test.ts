import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { Mimetypes } from "@plurnk/plurnk-mimetypes";
import Safetensors, { readSafetensors } from "./Safetensors.ts";

const META = { mimetype: "application/x-safetensors", glyph: "🔢", extensions: [".safetensors"] };
const h = () => new Safetensors(META);

const enc = new TextEncoder();
function buildST(header: object): Uint8Array {
    const json = enc.encode(JSON.stringify(header));
    const len = new Uint8Array(8);
    new DataView(len.buffer).setBigUint64(0, BigInt(json.length), true);
    const out = new Uint8Array(8 + json.length);
    out.set(len, 0);
    out.set(json, 8);
    return out;
}

const ST = buildST({
    "model.embed.weight": { dtype: "F16", shape: [32000, 4096], data_offsets: [0, 262144000] },
    "model.norm.weight": { dtype: "F32", shape: [4096], data_offsets: [262144000, 262160384] },
    __metadata__: { format: "pt" },
});

it("{§mimetype-content} the declared built handler produces a readable projection through the public framework", async () => {
    const root = resolve(import.meta.dirname, "..");
    const mimetypes = new Mimetypes({ discoverOptions: {
        cwd: root,
        packageDirs: [root],
        env: { PLURNK_EXTENSIONS_TRUSTED_ONLY: "1" },
    } });
    try {
        const projection = await mimetypes.projectReadable({ content: ST, hint: META.mimetype });
        assert.equal(projection?.content, "model.embed.weight: F16 [32000, 4096]\nmodel.norm.weight: F32 [4096]");
        assert.equal(projection?.sourceMimetype, META.mimetype);
        assert.match(projection?.projectionIdentity ?? "", /^[a-f0-9]{64}$/);
        assert.equal(await mimetypes.projectReadable({ content: enc.encode("not a model"), hint: META.mimetype }), null);
    } finally {
        await mimetypes.dispose();
    }
});

describe("Safetensors — header parse", () => {
    it("reads the tensor inventory (name, dtype, shape) without the weights", () => {
        const header = readSafetensors(ST)!;
        assert.equal(header.tensors.length, 2);
        assert.deepEqual(header.tensors[0], { name: "model.embed.weight", dtype: "F16", shape: [32000, 4096] });
        assert.equal(header.metadata.format, "pt");
    });

    it("rejects non-safetensors bytes", () => {
        assert.equal(readSafetensors(new TextEncoder().encode("nope")), null);
    });
});

describe("Safetensors — channels", () => {
    it("publishes the readable tensor inventory through the content channel", () => {
        assert.equal(h().content(ST), "model.embed.weight: F16 [32000, 4096]\nmodel.norm.weight: F32 [4096]");
        assert.equal(h().content(new Uint8Array()), undefined);
    });

    it("symbols are the tensor names; __metadata__ is not a tensor", () => {
        assert.deepEqual(h().extractRaw(ST).map((s) => s.name), ["model.embed.weight", "model.norm.weight"]);
        assert.deepEqual(h().extractRaw(ST).map(({ line, endLine }) => [line, endLine]), [[1, 1], [2, 2]]);
    });

    it("deepJson is the dtype/shape view, data_offsets dropped", () => {
        const tree = h().deepJson(ST) as Record<string, { dtype: string; shape: number[] }>;
        assert.deepEqual(tree["model.norm.weight"], { dtype: "F32", shape: [4096] });
        assert.equal("data_offsets" in (tree["model.norm.weight"] as object), false);
    });

    it("regex searches the readable tensor inventory", async () => {
        const matches = await h().query(ST, "regex", "embed\\.weight: (\\w+) \\[(\\d+)");
        const caps = matches[0]?.matched as string[];
        assert.equal(caps[0], "F16");
        assert.equal(caps[1], "32000");
    });

    it("validate throws on bad header; other channels degrade", () => {
        const bad = new TextEncoder().encode("nope");
        assert.throws(() => h().validate(bad), { name: "SyntaxError", message: "not a valid safetensors file (bad header)" });
        assert.deepEqual(h().extractRaw(bad), []);
        assert.doesNotThrow(() => h().validate(ST));
    });
});
