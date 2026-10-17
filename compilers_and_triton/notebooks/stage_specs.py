"""What each notebook says and does, separate from how it is built.

`make_notebooks.py` is machinery: it embeds sources, strips guards and
writes JSON. This file is content — the prose around each stage and the
cells that let a reader take it apart rather than only run it. Keeping
them apart means editing the teaching does not risk the generator.

Each entry:
    module    key into make_notebooks.SOURCES
    stem      output file name, without the extension
    title     the notebook's heading
    level     FUNDAMENTAL / INTERMEDIATE / ADVANCED / CAPSTONE
    needs     modules embedded ahead of this one, in order
    gpu       whether the stage needs a GPU to verify anything
    runtime   roughly how long the run cell takes
    summary   one paragraph under the heading
    explore   (heading, blurb, code) cells after the run
"""

from typing import Dict, List

SPECS: List[Dict] = [
    {
        'module': 'a1_lexer_parser',
        'stem': 'a1_lexer_parser',
        'title': 'A1 — Text to tree: lexing and Pratt parsing',
        'level': 'FUNDAMENTAL',
        'needs': [],
        'gpu': False,
        'runtime': 'under a second',
        'summary':
            'A lexer that turns characters into tokens, and a Pratt '
            'parser that turns\ntokens into a tree. The idea worth '
            'carrying forward is that **precedence\nis data**: it lives '
            'in one dictionary rather than in the shape of the\ncode, so '
            'adding an operator is a line instead of a refactor.',
        'explore': [
            ('Tokens come before trees',
             'The lexer does nothing clever, and that is deliberate. '
             'Every decision\nabout meaning belongs to the parser; a '
             'lexer that tries to be smart is a\nlexer you cannot reason '
             'about.',
             "for source in ('1 + 2 * 3',\n"
             "               'let x = 4 in x * x',\n"
             "               'max(a, -b) / 2'):\n"
             "    print(f'{source:<24}', end=' ')\n"
             "    print(' '.join(str(t) for t in tokenize(source)))"),
            ('Precedence is data, not control flow',
             'This is the claim, so test it. Demoting `*` to the level '
             'of `+` changes\nhow everything parses without touching a '
             'line of parser code. In a\nrecursive-descent parser written '
             'as a tower of functions, the same change\nis a rewrite.',
             "print('normal:   ', render(parse('2 + 3 * 4')))\n"
             "\n"
             "original = dict(BINDING_POWER)\n"
             "BINDING_POWER['*'] = BINDING_POWER['+']\n"
             "print('* demoted:', render(parse('2 + 3 * 4')))\n"
             "\n"
             "BINDING_POWER.clear()\n"
             "BINDING_POWER.update(original)\n"
             "print('restored: ', render(parse('2 + 3 * 4')))"),
            ('What a usable error message looks like',
             'A parser spends most of its working life being handed '
             'input that is\nwrong, so the error path is not an '
             'afterthought. Each of these reports a\nposition, because '
             '"syntax error" on its own tells the reader nothing.',
             "for bad in ('1 + * 2', '(1 + 2', 'let x 4 in x',\n"
             "            'max(1,', '@', '1 2'):\n"
             "    try:\n"
             "        parse(bad)\n"
             "        print(f'{bad!r:<16} parsed (unexpectedly)')\n"
             "    except CompileError as error:\n"
             "        print(f'{bad!r:<16} '\n"
             "              f'{str(error).splitlines()[0]}')"),
            ('Try it yourself: add an operator',
             'Adding `^` takes **two** edits, not one, and the first '
             'draft of this cell\ngot it wrong — it set a binding power '
             'and the lexer then rejected the\ncharacter outright, so '
             'the parser never saw a `^` to be clever about.\nThat is the '
             'honest shape of the design: the lexer decides what '
             'characters\nexist, the table decides what they mean. Two '
             'lines, two places, no new\nparser code.\n\n'
             'Note what is still missing afterwards. The expression '
             '*parses*, but it\ncannot be evaluated until `^` has a row '
             'in A2\'s `BINARY_OPS`, and it\ncannot be compiled until it '
             'has one in the capstone\'s `ELEMENTWISE`. An\noperator is '
             'not one change; it is one change per layer that has to '
             'know.',
             "import re\n"
             "\n"
             "TOKEN_SPEC[2] = ('OP', r'<=|>=|==|!=|[-+*/%<>()=,^]')\n"
             "TOKEN_RE = re.compile('|'.join(f'(?P<{name}>{pattern})'\n"
             "                               for name, pattern\n"
             "                               in TOKEN_SPEC))\n"
             "BINDING_POWER['^'] = 40\n"
             "\n"
             "print('tokens:     ',\n"
             "      ' '.join(str(t) for t in tokenize('2 ^ 3 ^ 2')))\n"
             "print('left-assoc: ', render(parse('2 ^ 3 ^ 2')))\n"
             "RIGHT_ASSOCIATIVE.add('^')\n"
             "print('right-assoc:', render(parse('2 ^ 3 ^ 2')))\n"
             "print('\\nand it parses inside a larger expression:')\n"
             "print('            ', render(parse('1 + 2 ^ 3 * 4')))\n"
             "\n"
             "RIGHT_ASSOCIATIVE.discard('^')\n"
             "del BINDING_POWER['^']\n"
             "TOKEN_SPEC[2] = ('OP', r'<=|>=|==|!=|[-+*/%<>()=,]')\n"
             "TOKEN_RE = re.compile('|'.join(f'(?P<{name}>{pattern})'\n"
             "                               for name, pattern\n"
             "                               in TOKEN_SPEC))"),
        ],
    },
    {
        'module': 'a2_ir',
        'stem': 'a2_ir',
        'title': 'A2 — Tree to three-address code, and an interpreter',
        'level': 'FUNDAMENTAL',
        'needs': ['a1_lexer_parser'],
        'gpu': False,
        'runtime': 'under a second',
        'summary':
            'Lowering the tree to a flat list of three-address '
            'instructions in which\nevery temporary is assigned exactly '
            'once. `let` disappears — scoping was\na front-end concern. '
            'The interpreter here is the **specification**:\neverything '
            'downstream has to preserve what it computes.',
        'explore': [
            ('Lowering, one sample at a time',
             'Watch `let` vanish. By the time the IR exists there are '
             'only names and\noperations, and every temporary is written '
             'once — which is exactly what\nmakes the passes in A3 '
             'possible to write correctly.',
             "for source, env in SAMPLES:\n"
             "    program = compile_source(source)\n"
             "    print(f'--- {source}')\n"
             "    print(program)\n"
             "    print(f'    = {execute(program, env):g}   '\n"
             "          f'({program.count()} instructions)\\n')"),
            ('Two independent routes to the same number',
             'The tree walker and the IR interpreter are separate '
             'implementations of\nthe same language. Agreement between '
             'them is the only reason to believe\neither one, and it is '
             'the baseline every later stage is measured against.',
             "import random\n"
             "random.seed(20260101)\n"
             "worst = 0.0\n"
             "for source, _ in SAMPLES:\n"
             "    tree = parse(source)\n"
             "    program = compile_source(source)\n"
             "    for _ in range(50):\n"
             "        env = {name: random.uniform(-5, 5)\n"
             "               for name in program.free_names}\n"
             "        worst = max(worst, abs(execute(program, env)\n"
             "                               - evaluate_tree(tree, env)))\n"
             "print(f'worst disagreement over 300 runs: {worst:.1e}')"),
            ('Shadowing, and why single assignment ends the argument',
             '`let x = 2 in let x = x + 1 in x * 10` has two different '
             'things called\n`x`. The lowerer gives each a fresh name, so '
             'the IR has no shadowing left\nto get wrong — and neither '
             'does any pass written against it.',
             "program = compile_source(\n"
             "    'let x = 2 in let x = x + 1 in x * 10')\n"
             "print(program)\n"
             "print(f'\\nresult {execute(program, {}):g} — the two x '\n"
             "      f'values have different IR names')"),
            ('Try it yourself',
             'Lower an expression of your own and count the '
             'instructions. Then carry\nthe same expression into A3 and '
             'see how much of it survives.',
             "yours = 'let g = (a + b) / 2 in max(g * g, exp(0)) - "
             "abs(-3)'\n"
             "program = compile_source(yours)\n"
             "print(program)\n"
             "print(f'\\n{program.count()} instructions; free names '\n"
             "      f'{sorted(program.free_names)}')"),
        ],
    },
    {
        'module': 'a3_passes',
        'stem': 'a3_passes',
        'title': 'A3 — Folding, propagation, CSE and DCE, to a fixpoint',
        'level': 'INTERMEDIATE',
        'needs': ['a1_lexer_parser', 'a2_ir'],
        'gpu': False,
        'runtime': 'a second or two',
        'summary':
            'Four classical passes — constant folding, copy '
            'propagation, common\nsubexpression elimination and '
            'dead-code elimination — run to a fixpoint,\nbecause each one '
            'creates work for the others. Correctness is established '
            'by\n**differential testing**: the optimised program is run '
            'against the\noriginal on hundreds of random inputs.',
        'explore': [
            ('One program, pass by pass',
             'The trace shows why a single round is never enough: '
             'folding creates\ncopies, propagation makes subexpressions '
             'identical, CSE makes\ninstructions dead, DCE removes them — '
             'and that can expose more folding.\nThe loop stops when a '
             'whole round changes nothing, which is the only\nhonest '
             'termination condition.',
             "program = compile_source(\n"
             "    'let s = a + b in s * s + (a + b) * (a + b)')\n"
             "print('BEFORE')\n"
             "print(program)\n"
             "optimised, trace = optimise(program, trace=True)\n"
             "print('\\nTRACE')\n"
             "for line in trace:\n"
             "    print(' ', line)\n"
             "print('\\nAFTER')\n"
             "print(optimised)"),
            ('The fixpoint is not decoration',
             'Cap the rounds at one and work is left on the table. The '
             'middle column\nis what a pipeline that runs each pass once '
             'achieves; the right-hand one\nis convergence.',
             "for label, source in BENCHMARKS:\n"
             "    base = compile_source(source)\n"
             "    one, _ = optimise(compile_source(source), max_rounds=1)\n"
             "    full, _ = optimise(compile_source(source))\n"
             "    print(f'{label:<22} {base.count():>3} -> '\n"
             "          f'{one.count():>3} (one round) -> '\n"
             "          f'{full.count():>3} (fixpoint)')"),
            ('Differential testing, which is how this is really checked',
             'Proving a pass correct is hard; running the optimised '
             'program against the\noriginal on a thousand random inputs '
             'is easy, and it is what real\ncompiler test suites do. '
             'Raise the trial count and it stays silent —\nthat silence '
             'is the evidence.',
             "for label, source in BENCHMARKS:\n"
             "    before = compile_source(source)\n"
             "    after, _ = optimise(compile_source(source))\n"
             "    ok, detail = differential_check(before, after,\n"
             "                                    trials=1000)\n"
             "    verdict = 'agree' if ok else 'DISAGREE'\n"
             "    print(f'{label:<22} {verdict:<9} {detail}')"),
            ('The optimisation that must not happen',
             'Folding `1 / 0` would turn a runtime error into a '
             'compile-time crash.\nThat is a change in behaviour, not an '
             'optimisation — the compiler would be\nwrong even though the '
             'program it produced was smaller.',
             "program = compile_source('1 / 0')\n"
             "folded, _ = optimise(program)\n"
             "print('IR after optimisation:')\n"
             "print(folded)\n"
             "print('\\nthe division survived, so the error happens at '\n"
             "      'run time where it belongs:')\n"
             "try:\n"
             "    execute(folded, {})\n"
             "except ZeroDivisionError as error:\n"
             "    print(f'  ZeroDivisionError: {error}')"),
        ],
    },
    {
        'module': 'a4_schedule',
        'stem': 'a4_schedule',
        'title': 'A4 — Algorithm versus schedule, cost models, autotuning',
        'level': 'ADVANCED',
        'needs': [],
        'gpu': False,
        'runtime': 'a second',
        'summary':
            'The separation that organises Halide, TVM and every tensor '
            'compiler\nsince: the **algorithm** says what to compute, the '
            '**schedule** says in\nwhat order and in what tile sizes. A '
            'readable cost model stands in for\nthe hardware, a search '
            'over it finds a tiling, and the arithmetic never\nchanges.',
        'explore': [
            ('Five schedules, one algorithm',
             'Every row computes exactly the same arithmetic. Only the '
             'order and the\ntile sizes differ, and the traffic differs '
             'by two orders of magnitude.\nThat gap is the entire reason '
             'tensor compilers exist.',
             "m = n = k = 1024\n"
             "candidates = [\n"
             "    ('naive triple loop', None),\n"
             "    ('thin tiles', Schedule(16, 256, 32, 'mnk')),\n"
             "    ('square tiles', Schedule(64, 64, 32, 'mnk')),\n"
             "    ('k outermost', Schedule(64, 64, 32, 'kmn')),\n"
             "    ('big square', Schedule(128, 128, 8, 'mnk')),\n"
             "]\n"
             "for label, schedule in candidates:\n"
             "    if schedule is None:\n"
             "        cost, extra = naive_traffic(m, n, k), ''\n"
             "    else:\n"
             "        cost = estimate_traffic(schedule, m, n, k)\n"
             "        extra = (f'   working set '\n"
             "                 f'{working_set(schedule) / 1024:6.1f} KB')\n"
             "    print(f'{label:<20} {cost / 1e9:8.2f} GB{extra}')"),
            ('The cache budget changes the answer',
             'An autotuner is not looking for "the best tiling" — it is '
             'looking for the\nbest tiling *on this machine*. Shrink the '
             'fast memory and the winner\nchanges, which is why tuned '
             'kernels do not transfer between GPUs.',
             "for budget_kb in (16, 32, 64, 128, 256, 512):\n"
             "    best, cost, tried = autotune(1024, 1024, 1024,\n"
             "                                 budget_kb * 1024)\n"
             "    print(f'{budget_kb:>4} KB cache   {best}  '\n"
             "          f'{cost / 1e9:7.3f} GB   ({tried} feasible)')"),
            ('Shape matters as much as the machine',
             'A tall-thin multiply and a square one do not want the same '
             'schedule. This\nis why `triton.autotune` keys its cache on '
             'the shapes as well as the\nconfiguration.',
             "for m, n, k in ((1024, 1024, 1024), (4096, 64, 1024),\n"
             "                (64, 4096, 1024), (128, 128, 8192)):\n"
             "    best, cost, _ = autotune(m, n, k)\n"
             "    print(f'{m:>5} x {n:<5} k={k:<5} -> {best}  '\n"
             "          f'{cost / 1e9:7.3f} GB')"),
            ('Try it yourself, then ask where the model lies',
             'The ranking below is everything this model can see. It '
             'ignores vector\nwidths, tensor-core shapes and occupancy, '
             'so it ranks tilings sensibly and\nwould mislead you about '
             'absolute speed. That is the honest limit of a\ncost model, '
             'and why B3 writes the kernel and `triton.autotune` times '
             'it.',
             "ranked = sorted(((estimate_traffic(s, 1024, 1024, 1024), s)\n"
             "                 for s in search_space()),\n"
             "                key=lambda pair: pair[0])\n"
             "print('the ten best schedules the model can see:')\n"
             "for rank, (score, schedule) in enumerate(ranked[:10], 1):\n"
             "    print(f'  {rank:>2}. {schedule}   "
             "{score / 1e9:7.3f} GB')\n"
             "print(f'\\nworst feasible: {ranked[-1][1]}   '\n"
             "      f'{ranked[-1][0] / 1e9:.3f} GB')"),
        ],
    },
    {
        'module': 'b1_vector_add',
        'stem': 'b1_vector_add',
        'title': 'B1 — The first kernel, and the three ideas in it',
        'level': 'FUNDAMENTAL',
        'needs': ['runtime'],
        'gpu': True,
        'runtime': 'a few seconds, mostly compilation',
        'summary':
            'Vector addition, which is worth more attention than it '
            'usually gets. It\ncontains the three ideas you use in every '
            'kernel afterwards: you write\nfor a **block** rather than an '
            'element, `program_id` tells you which block\nyou are, and '
            '**masks** handle the ragged edge without branching.',
        'explore': [
            ('What the grid actually is',
             'There is no loop over blocks. The grid *is* the loop, '
             'executed in\nparallel, and `program_id` is how an instance '
             'finds out which iteration\nit is. This is the arithmetic '
             'the launch does, in Python, so you can see\nwhere the last '
             'partial block comes from.',
             "for n, block in ((1024, 256), (1000, 256), (1, 256),\n"
             "                 (70000, 1024)):\n"
             "    grid = -(-n // block)\n"
             "    tail = n - (grid - 1) * block\n"
             "    print(f'n={n:<7} BLOCK={block:<5} grid={grid:<4} '\n"
             "          f'last block holds {tail} of {block} lanes')"),
            ('The mask is not a detail',
             'Without a mask the final instance reads and writes past '
             'the end of the\nbuffer. On a GPU that is not a crash — it '
             'is silently wrong data, or\nsomeone else\'s. The mask is '
             'what makes a block-shaped kernel safe on\narbitrary '
             'lengths, which is why every load and store in this '
             'repository\ncarries one.',
             "import numpy as np\n"
             "n, block = 10, 4\n"
             "for pid in range(-(-n // block)):\n"
             "    offsets = pid * block + np.arange(block)\n"
             "    mask = offsets < n\n"
             "    print(f'pid={pid}  offsets={offsets}  mask={mask}')"),
            ('Ragged lengths, checked',
             'Run the reference against lengths that are deliberately '
             'awkward. If a\nkernel is going to be wrong about the edge, '
             'it will be wrong here — and\nthese are the sizes real '
             'tensors have.',
             "import numpy as np\n"
             "rng = np.random.default_rng(1)\n"
             "for n in (1, 2, 255, 256, 257, 1000, 4095):\n"
             "    x, y = rng.standard_normal(n), rng.standard_normal(n)\n"
             "    expected = reference_add(x, y)\n"
             "    if CAN_RUN_KERNELS:\n"
             "        import torch\n"
             "        got = vector_add(torch.from_numpy(x).cuda(),\n"
             "                         torch.from_numpy(y).cuda())\n"
             "        gap = float((got.cpu().numpy() - expected).max())\n"
             "        print(f'n={n:<6} kernel max error {gap:.2e}')\n"
             "    else:\n"
             "        print(f'n={n:<6} reference only (no GPU here)')"),
            ('Try it yourself',
             'Change the block size and watch nothing break. The kernel '
             'is written for\na block of unknown width, so 64, 256 and '
             '4096 are all correct — they\ndiffer only in speed, which is '
             'the distinction A4 was about.',
             "if CAN_RUN_KERNELS:\n"
             "    import numpy as np\n"
             "    import torch\n"
             "    x = torch.randn(1_000_003, device='cuda')\n"
             "    y = torch.randn(1_000_003, device='cuda')\n"
             "    for block in (64, 256, 1024, 4096):\n"
             "        got = vector_add(x, y, block_size=block)\n"
             "        ok = torch.allclose(got, x + y, atol=1e-6)\n"
             "        print(f'BLOCK={block:<5} matches torch: {ok}')\n"
             "else:\n"
             "    print('needs a GPU: Runtime -> Change runtime type -> "
             "T4 GPU')"),
        ],
    },
    {
        'module': 'b2_softmax',
        'stem': 'b2_softmax',
        'title': 'B2 — Reductions, and why softmax is written oddly',
        'level': 'INTERMEDIATE',
        'needs': ['runtime'],
        'gpu': True,
        'runtime': 'a few seconds, mostly compilation',
        'summary':
            'A reduction is the first thing that breaks the elementwise '
            'picture:\nsoftmax needs the maximum and the sum of a row '
            'before it can emit\nanything. Two lessons — give each '
            'program instance a whole row, and\n**subtract the maximum**, '
            'because `exp(x)` overflows float32 above about\n88 and '
            'logits of 100 are ordinary.',
        'explore': [
            ('Watch float32 run out of room',
             'The identity `softmax(x) = softmax(x - c)` is exact: the '
             'constant cancels\nbetween numerator and denominator. What '
             'is not exact is `exp` in\nfloat32, which overflows a little '
             'above 88. Find the point where the\nnaive version stops '
             'being arithmetic and starts being NaN.',
             "import numpy as np\n"
             "for scale in (1, 10, 50, 80, 88, 90, 100, 1000):\n"
             "    row = np.array([[scale - 1.0, scale, scale + 1.0]])\n"
             "    naive = naive_softmax(row)[0]\n"
             "    stable = reference_softmax(row)[0]\n"
             "    state = 'ok' if np.all(np.isfinite(naive)) else 'NaN'\n"
             "    print(f'logits around {scale:<5} naive {state:<4} '\n"
             "          f'stable {np.round(stable, 4)}')"),
            ('Why masked lanes load minus infinity',
             'A padded lane must lose the maximum and contribute nothing '
             'to the sum.\nLoading `0.0` would do neither: on an '
             'all-negative row, zero *is* the\nmaximum, and the answer '
             'comes out wrong with no error anywhere.',
             "import numpy as np\n"
             "row = np.array([[-5.0, -3.0, -8.0]])\n"
             "padded_with_zero = np.array([[-5.0, -3.0, -8.0, 0.0]])\n"
             "print('true softmax of the row       ',\n"
             "      np.round(reference_softmax(row)[0], 4))\n"
             "print('with a zero-padded lane       ',\n"
             "      np.round(reference_softmax(padded_with_zero)[0][:3],\n"
             "               4))\n"
             "print('\\nthe padding changed the answer, which is why the '\n"
             "      'kernel loads -inf')"),
            ('One row per program, and where that stops working',
             'The whole reduction happens in registers when a row fits a '
             'block, so no\ninstance ever has to talk to another. The '
             'launcher refuses rows that do\nnot fit rather than silently '
             'truncating — the point at which you need a\ntwo-pass or '
             'online algorithm is a real boundary, and it should be '
             'visible.',
             "import numpy as np\n"
             "if CAN_RUN_KERNELS:\n"
             "    import torch\n"
             "    for cols in (128, 781, 2048, 65536, 131072):\n"
             "        try:\n"
             "            got = softmax(torch.randn(4, cols,\n"
             "                                      device='cuda'))\n"
             "            print(f'{cols:>7} columns  ok')\n"
             "        except ValueError as error:\n"
             "            print(f'{cols:>7} columns  refused: {error}')\n"
             "else:\n"
             "    print('needs a GPU; the reference handles any width')\n"
             "    wide = np.random.default_rng(0).standard_normal(\n"
             "        (2, 131072))\n"
             "    print('reference row sums:',\n"
             "          reference_softmax(wide).sum(axis=1))"),
            ('Try it yourself',
             'Compare against PyTorch on a GPU. The interesting number '
             'is not the\nspeedup — it is that a thirty-line kernel '
             'matches a library that has had\nyears of work, because '
             'softmax is memory-bound and there is only so much\nto win.',
             "if CAN_RUN_KERNELS:\n"
             "    import torch\n"
             "    x = torch.randn(4096, 1024, device='cuda')\n"
             "    mine = softmax(x)\n"
             "    theirs = torch.softmax(x, axis=1)\n"
             "    print('max difference:',\n"
             "          float((mine - theirs).abs().max()))\n"
             "else:\n"
             "    print('needs a GPU: Runtime -> Change runtime type -> "
             "T4 GPU')"),
        ],
    },
    {
        'module': 'b3_matmul',
        'stem': 'b3_matmul',
        'title': 'B3 — The tiled matrix multiply A4 described',
        'level': 'ADVANCED',
        'needs': ['runtime'],
        'gpu': True,
        'runtime': 'ten seconds or so, mostly compilation',
        'summary':
            'A4 searched schedules and concluded: square tiles, k '
            'innermost,\naccumulate in registers. This is that kernel. '
            'The accumulator never\nleaves registers during the k loop, '
            'so C is written exactly once, and\ngrouped ordering permutes '
            'the launch so neighbouring instances share '
            'both\noperands in L2.',
        'explore': [
            ('Why the accumulator is float32 even for float16 inputs',
             'Accumulating a long sum in half precision loses accuracy '
             'badly, and a k\nloop of 4096 is ordinary. The error lands '
             'in every element of C.\nfloat32 accumulation costs nothing '
             'on tensor cores, which is why every\nserious kernel does '
             'it.',
             "import numpy as np\n"
             "rng = np.random.default_rng(7)\n"
             "for length in (64, 256, 1024, 4096, 16384):\n"
             "    values = rng.standard_normal(length).astype(np.float16)\n"
             "    half = np.float16(0)\n"
             "    for value in values:\n"
             "        half = np.float16(half + value)\n"
             "    exact = float(values.astype(np.float64).sum())\n"
             "    error = abs(float(half) - exact) / max(abs(exact), 1e-9)\n"
             "    print(f'k={length:<6} float16 accumulation relative '\n"
             "          f'error {error:.2e}')"),
            ('Grouped ordering, worked out on paper',
             'The permutation is the whole of it. Row-major ordering '
             'means\nconsecutive instances share a strip of A and stride '
             'across the whole of\nB, so L2 sees no reuse; grouping makes '
             'neighbours share both. Print the\ntwo orders and the '
             'difference is obvious.',
             "tiles_m = tiles_n = 4\n"
             "group_m = 2\n"
             "print('pid -> (row, col)      row-major      grouped')\n"
             "for pid in range(tiles_m * tiles_n):\n"
             "    plain = (pid // tiles_n, pid % tiles_n)\n"
             "    per_group = group_m * tiles_n\n"
             "    group = pid // per_group\n"
             "    first_m = group * group_m\n"
             "    rows = min(tiles_m - first_m, group_m)\n"
             "    grouped = (first_m + (pid % per_group) % rows,\n"
             "               (pid % per_group) // rows)\n"
             "    print(f'{pid:>3}                  {plain}        '\n"
             "          f'{grouped}')"),
            ('Ragged shapes, which is where kernels are wrong',
             'A multiply whose dimensions are not multiples of the tile '
             'sizes is the\nnormal case, and masks are the whole answer — '
             'the same idiom as B1. The\nreference runs in float64 so it '
             'is the trustworthy number, not a second\nopinion of equal '
             'standing.',
             "import numpy as np\n"
             "rng = np.random.default_rng(3)\n"
             "for m, n, k in ((256, 256, 256), (300, 200, 101),\n"
             "                (17, 129, 65), (1, 1, 1024)):\n"
             "    a = rng.standard_normal((m, k)).astype(np.float32)\n"
             "    b = rng.standard_normal((k, n)).astype(np.float32)\n"
             "    expected = reference_matmul(a, b)\n"
             "    if CAN_RUN_KERNELS:\n"
             "        import torch\n"
             "        got = matmul(torch.from_numpy(a).cuda(),\n"
             "                     torch.from_numpy(b).cuda())\n"
             "        scale = max(abs(expected).max(), 1e-9)\n"
             "        gap = abs(got.cpu().numpy() - expected).max() / scale\n"
             "        print(f'{m:>4}x{k:<4} @ {k:>4}x{n:<4} relative '\n"
             "              f'error {gap:.2e}')\n"
             "    else:\n"
             "        print(f'{m:>4}x{k:<4} @ {k:>4}x{n:<4} reference '\n"
             "              f'only (no GPU here)')"),
            ('Try it yourself: tune against the clock, not the model',
             'This is `triton.autotune` by hand, and it is where A4 gets '
             'marked.\nWatch for two things.\n\n'
             'First, the ranking. A4 predicted square tiles and a large '
             '`BLOCK_M`\nwould win, and the clock broadly agrees — but '
             'not in the same order,\nbecause the model knows nothing '
             'about tensor-core shapes, occupancy or\npipelining.\n\n'
             'Second, `OutOfResources`. A4 modelled a 128 KB fast-memory '
             'budget, and\nthe GPU you are on may have less: a 128x128 '
             'tile with `BLOCK_K=64` wants\n128 KB of shared memory, and '
             'an RTX 5070 offers 99 KB. The feasibility\nconstraint in '
             'A4\'s `search_space` was not a modelling convenience. It '
             'is\nthis error, predicted.',
             "if CAN_RUN_KERNELS:\n"
             "    import torch\n"
             "    import triton\n"
             "    from triton.runtime.errors import OutOfResources\n"
             "\n"
             "    a = torch.randn(1024, 1024, device='cuda')\n"
             "    b = torch.randn(1024, 1024, device='cuda')\n"
             "    limit = torch.cuda.get_device_properties(\n"
             "        0).shared_memory_per_block_optin\n"
             "    print(f'shared memory per block here: "
             "{limit / 1024:.0f} KB')\n"
             "    print(f\"A4's assumed budget:          \"\n"
             "          f'{128 * 1024 / 1024:.0f} KB\\n')\n"
             "\n"
             "    flops = 2 * 1024 ** 3\n"
             "    for bm, bn, bk in ((32, 32, 32), (64, 64, 32),\n"
             "                       (128, 64, 32), (128, 128, 32),\n"
             "                       (128, 128, 64), (256, 256, 64)):\n"
             "        want = (bm * bk + bk * bn) * 4 * 2\n"
             "        try:\n"
             "            ms = triton.testing.do_bench(\n"
             "                lambda m=bm, n=bn, k=bk: matmul(a, b, m, n, k))\n"
             "        except OutOfResources:\n"
             "            print(f'{bm:>4}x{bn:<4} k={bk:<4} does not fit: '\n"
             "                  f'wants ~{want / 1024:.0f} KB of "
             "{limit / 1024:.0f} KB')\n"
             "            continue\n"
             "        print(f'{bm:>4}x{bn:<4} k={bk:<4} {ms:7.3f} ms   '\n"
             "              f'{flops / (ms * 1e-3) / 1e12:5.2f} TFLOP/s')\n"
             "\n"
             "    ms = triton.testing.do_bench(lambda: a @ b)\n"
             "    print(f'\\ntorch.matmul        {ms:7.3f} ms   '\n"
             "          f'{flops / (ms * 1e-3) / 1e12:5.2f} TFLOP/s')\n"
             "else:\n"
             "    print('needs a GPU: Runtime -> Change runtime type -> "
             "T4 GPU')"),
        ],
    },
    {
        'module': 'tensor_compiler',
        'stem': 'capstone_tensor_compiler',
        'title': 'Capstone — A tensor compiler that emits fused Triton',
        'level': 'CAPSTONE',
        'needs': ['a1_lexer_parser', 'a2_ir', 'a3_passes', 'runtime'],
        'gpu': False,
        'runtime': 'a second or two',
        'summary':
            'The two tracks meeting. The front end from A1-A2 and the '
            'passes from A3\nare reused **unchanged**, and one `emit` '
            'walks the same IR to produce\neither NumPy or a fused Triton '
            'kernel. That the IR needed no changes to\ngo from scalars to '
            'tensors is the entire argument for having an IR.',
        'explore': [
            ('One expression, both backends',
             'Read the two emitted functions side by side. They differ '
             'in a table of\nexpression templates and a prologue; the '
             'parser, the lowerer and all four\npasses are the same code '
             'that ran in A1 to A3.',
             "program = optimise(compile_source('max(x * w + b, 0)'))[0]\n"
             "print('OPTIMISED IR')\n"
             "print(program)\n"
             "for backend in ('numpy', 'triton'):\n"
             "    print(f'\\n===== {backend} '\n"
             "          f'{\"=\" * (60 - len(backend))}')\n"
             "    print(emit(program, backend, 'relu_affine'))"),
            ('What fusion is actually worth',
             'Unfused, each operation reads its operands and writes a '
             'temporary, so an\nexpression of four operations makes four '
             'kernel launches and twelve\nmemory passes. Fused, it reads '
             'each input once and writes once. For\nmemory-bound '
             'elementwise work that ratio *is* the speedup.',
             "for label, expression in EXPRESSIONS:\n"
             "    program = optimise(compile_source(expression))[0]\n"
             "    unfused, fused = count_memory_passes(program)\n"
             "    print(f'{label:<22} {expression:<34} '\n"
             "          f'{unfused:>3} passes -> {fused:>2}  '\n"
             "          f'({unfused / fused:.1f}x less traffic)')"),
            ('The optimiser earns its place here',
             'CSE is not a tidiness measure once the operands are '
             'tensors: a repeated\nsubexpression is a repeated pass over '
             'memory. Compare the instruction\ncount and the emitted body '
             'before and after.',
             "expression = '(x + y) * (x + y) + exp(x + y)'\n"
             "raw = compile_source(expression)\n"
             "tuned, _ = optimise(compile_source(expression))\n"
             "print(f'{expression}\\n')\n"
             "print(f'unoptimised: {raw.count()} instructions')\n"
             "print(f'optimised:   {tuned.count()} instructions\\n')\n"
             "print(emit(tuned, 'triton', 'shared_subexpression'))"),
            ('A compiler that refuses is better than one that guesses',
             'The emission table is the whole of what this backend '
             'knows. Take an\noperation out of it and the compiler '
             'refuses the program rather than\nemitting something '
             'plausible and wrong — which is the behaviour you want\nfrom '
             'anything that generates code you are not going to read.',
             "program = optimise(compile_source('max(x, 0) + 1'))[0]\n"
             "removed = ELEMENTWISE.pop('max')\n"
             "try:\n"
             "    emit(program, 'triton')\n"
             "    print('emitted anyway — that would be a bug')\n"
             "except UnsupportedOperation as error:\n"
             "    print(f'refused: {error}')\n"
             "finally:\n"
             "    ELEMENTWISE['max'] = removed\n"
             "print('\\nrestored, and it compiles again:')\n"
             "print(emit(program, 'triton').splitlines()[0])"),
            ('Try it yourself',
             'Compile an expression of your own, check the emitted NumPy '
             'against a\ndirect evaluation, and read the Triton. If you '
             'have a GPU, the emitted\nkernel is valid Triton — paste it '
             'into a cell with the launcher from B1\nand run it.',
             "import numpy as np\n"
             "yours = 'exp(x) / (exp(x) + 1) * w'\n"
             "program = optimise(compile_source(yours))[0]\n"
             "source = emit(program, 'numpy', 'yours')\n"
             "kernel = build_numpy_kernel(source, 'yours')\n"
             "rng = np.random.default_rng(11)\n"
             "inputs = {name: rng.standard_normal(1000)\n"
             "          for name in sorted(program.free_names)}\n"
             "got = kernel(**inputs)\n"
             "expected = evaluate_directly(yours, inputs)\n"
             "print(f'max difference from a direct evaluation: '\n"
             "      f'{np.abs(got - expected).max():.2e}\\n')\n"
             "print(emit(program, 'triton', 'yours'))"),
        ],
    },
]
