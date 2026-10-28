"""Rebuild the kernel containers lib/kernels/k<cc>.pt from kernels.py and specs/sm<cc>.json.

Needs Linux, Triton 3.3.1 and PyTorch; no GPU (Triton compiles for the target compute capability with its bundled ptxas).

  pip install triton==3.3.1 torch
  python build.py --out built                       # every architecture in specs/
  python build.py --out built --archs 86 --compare ../lib/kernels

Each spec fixes one specialization the runtime launches: the kernel, its constexpr values (tile sizes, unit strides),
which arguments are multiples of 16, and num_warps / num_stages. --compare checks every rebuilt cubin against the
shipped container: same entry keys, launch parameters and shared memory, and identical machine code (the .text
sections; debug line tables carry file paths and line numbers and are not compared).
"""

import argparse
import hashlib
import io
import json
import re
import struct
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LAUNCH_FIELDS = (
    'key',
    'name',
    'num_warps',
    'shared',
    'params',
    'hidden',
    'div16',
    'expect',
    'offset',
    'size',
    'sha256',
)


def ptx_entry_params(ptx, name):
    m = re.search(r'\.entry\s+' + re.escape(name) + r'\s*\((.*?)\)', ptx, re.S)
    if not m:
        raise RuntimeError(f'{name}: no .entry in PTX')
    return re.findall(r'\.param\s+\.(\w+)', m.group(1))


def compile_one(triton, fn, spec, cc):
    """One cubin for one spec, plus the launcher fields the runtime reads."""
    from triton.backends.compiler import GPUTarget

    names = list(fn.arg_names)
    types = {p['name']: p['type'] for p in spec['params']}
    sig = {a: 'constexpr' if a in spec['constexprs'] else types[a] for a in names}
    attrs = {(names.index(a),): [['tt.divisibility', 16]] for a in spec['div16']}
    src = triton.compiler.ASTSource(fn=fn, signature=sig, constexprs=dict(spec['constexprs']), attrs=attrs)
    ck = triton.compile(
        src, target=GPUTarget('cuda', cc, 32), options=dict(num_warps=spec['num_warps'], num_stages=spec['num_stages'])
    )
    md = ck.metadata
    runtime = [(a, t) for a, t in sig.items() if t != 'constexpr']
    ptx_types = ptx_entry_params(ck.asm['ptx'], md.name)
    n_hidden = len(ptx_types) - len(runtime)
    if n_hidden < 0:
        raise RuntimeError(f'{spec["key"]}: PTX has fewer params than the runtime arguments')
    for (a, t), pt in zip(runtime, ptx_types):  # pointers are .u64, i32 scalars .u32/.s32/.b32
        if not (pt in ('u64', 'b64') if t.startswith('*') else pt in ('u32', 's32', 'b32')):
            raise RuntimeError(f'{spec["key"]}: argument {a} ({t}) maps to PTX .{pt}')
    hidden = [
        dict(
            name='global_scratch' if i == 0 else f'hidden_{i}',
            size=int(getattr(md, 'global_scratch_size', 0) or 0) if i == 0 else 0,
            align=int(getattr(md, 'global_scratch_align', 1) or 1) if i == 0 else 1,
        )
        for i in range(n_hidden)
    ]
    entry = dict(
        key=spec['key'],
        name=md.name,
        num_warps=int(md.num_warps),
        shared=int(md.shared),
        params=[dict(name=a, type=t) for a, t in runtime],
        hidden=hidden,
        div16=spec['div16'],
        expect=spec['expect'],
        constexprs={k: spec['constexprs'][k] for k in ('BM', 'BN') if k in spec['constexprs']},
    )
    return entry, bytes(ck.asm['cubin'])


def pack(cc, arch_specific, entries, blob):
    """The runtime container: the launcher index and the concatenated cubins, each zlib-compressed into a uint8 tensor."""
    import torch

    index = {
        'cc': cc,
        'arch_specific': arch_specific,
        'entries': [{k: e[k] for k in LAUNCH_FIELDS} | {'constexprs': e['constexprs']} for e in entries],
    }
    as_u8 = lambda raw: torch.frombuffer(bytearray(zlib.compress(raw, 9)), dtype=torch.uint8).clone()
    return {'i': as_u8(json.dumps(index).encode('utf-8')), 'b': as_u8(bytes(blob))}


def unpack(path):
    import torch

    data = torch.load(str(path), map_location='cpu', weights_only=True)
    index = json.loads(zlib.decompress(bytes(data['i'].tolist())).decode('utf-8'))
    return index, zlib.decompress(bytes(data['b'].tolist()))


def text_sections(cubin):
    """sha256 of every .text.* section of an ELF64 cubin (the machine code)."""
    (shoff,) = struct.unpack_from('<Q', cubin, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', cubin, 0x3A)
    sections = [struct.unpack_from('<IIQQQQIIQQ', cubin, shoff + i * shentsize) for i in range(shnum)]
    names_off = sections[shstrndx][4]
    out = {}
    for name_off, _, _, _, off, size, *_ in sections:
        end = cubin.index(b'\0', names_off + name_off)
        name = cubin[names_off + name_off : end].decode()
        if name.startswith('.text.'):
            out[name] = hashlib.sha256(cubin[off : off + size]).hexdigest()
    return out


def build_arch(triton, doc, out):
    import torch
    from kernels import KERNELS

    entries, blob = [], bytearray()
    for spec in doc['specs']:
        entry, cubin = compile_one(triton, KERNELS[spec['kernel']], spec, doc['cc'])
        while len(blob) % 256:
            blob.append(0)
        entry.update(offset=len(blob), size=len(cubin), sha256=hashlib.sha256(cubin).hexdigest())
        blob += cubin
        entries.append(entry)
    path = out / f'k{doc["cc"]}.pt'
    torch.save(pack(doc['cc'], doc['arch_specific'], entries, blob), path)
    return path


def compare(built, shipped):
    (a, blob_a), (b, blob_b) = unpack(built), unpack(shipped)
    problems = []
    if [e['key'] for e in a['entries']] != [e['key'] for e in b['entries']]:
        return ['entry keys differ']
    for x, y in zip(a['entries'], b['entries']):
        for field in ('name', 'num_warps', 'shared', 'params', 'hidden', 'div16', 'expect', 'constexprs'):
            if x[field] != y[field]:
                problems.append(f'{x["key"]}: {field} differs')
        code_a = text_sections(blob_a[x['offset'] : x['offset'] + x['size']])
        code_b = text_sections(blob_b[y['offset'] : y['offset'] + y['size']])
        if code_a != code_b:
            problems.append(f'{x["key"]}: machine code differs')
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--archs', default=None, help='comma-separated compute capabilities (default: every spec file)')
    ap.add_argument('--compare', default=None, help='folder of shipped k<cc>.pt containers to check against')
    a = ap.parse_args()
    import triton

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((HERE / 'specs').glob('sm*.json'))]
    if a.archs:
        docs = [d for d in docs if str(d['cc']) in a.archs.split(',')]
    failed = False
    for doc in docs:
        if not triton.__version__.startswith(doc['triton']):
            print(f'warning: specs were built with Triton {doc["triton"]}, this is {triton.__version__}')
        path = build_arch(triton, doc, out)
        line = f'sm_{doc["cc"]}: {len(doc["specs"])} kernels -> {path}'
        if a.compare:
            problems = compare(path, Path(a.compare) / path.name)
            failed |= bool(problems)
            line += '; ' + (
                'identical machine code' if not problems else f'{len(problems)} differences: {problems[:3]}'
            )
        print(line, flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
