"""Build the self-contained Colab notebook from canonical source files."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "jax_addition_transformer"
OUTPUT = ROOT / "notebooks" / "01_build_train_exact_10m.ipynb"
SOURCE_ORDER = [
    "config.py",
    "tokenizer.py",
    "data.py",
    "sampling.py",
    "layers.py",
    "normalization.py",
    "positional.py",
    "attention.py",
    "ffn.py",
    "model.py",
    "losses.py",
    "optimizers.py",
    "generation.py",
    "evaluation.py",
    "checkpointing.py",
    "reporting.py",
    "training.py",
    "cli.py",
    "__init__.py",
]
SOURCE_FILES = [PACKAGE / name for name in SOURCE_ORDER]


def source_cell(path: Path):
    relative = path.relative_to(ROOT)
    cell = nbf.v4.new_code_cell(f"%%writefile {relative}\n{path.read_text()}")
    cell.metadata["canonical_source"] = str(relative)
    return cell


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": OUTPUT.name, "provenance": []},
        }
    )
    badge = "https://colab.research.google.com/assets/colab-badge.svg"
    url = "https://colab.research.google.com/github/marcoharuni/jax-addition-transformer/blob/main/notebooks/01_build_train_exact_10m.ipynb"
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            f"# JAX Addition Transformer: exact 10M\n\n[![Open In Colab]({badge})]({url})\n\nA complete decoder-only arithmetic-language-model experiment. Numerical outputs are produced only when this notebook runs."
        ),
        nbf.v4.new_markdown_cell("## 1. Runtime and device verification"),
        nbf.v4.new_code_cell(
            "import sys, subprocess, importlib.metadata\nprint(sys.version)\nimport jax\nprint('JAX',jax.__version__,'devices',jax.devices())\nFULL_RUN=any(d.platform=='gpu' for d in jax.devices())\nif not FULL_RUN: print('The full default run requires Runtime > Change runtime type > T4 GPU.')\ndef require_gpu_for_full_run():\n    assert any(d.platform=='gpu' for d in jax.devices()), 'Select a T4 GPU runtime before starting the full default run.'\nsupport={'flax':'0.12.0','optax':'0.2.6','orbax-checkpoint':'0.11.28','matplotlib':'3.10.7'}\nneeded=[]\nfor package,version in support.items():\n    try: installed=importlib.metadata.version(package)\n    except importlib.metadata.PackageNotFoundError: installed=None\n    if installed!=version: needed.append(f'{package}=={version}')\nif needed: subprocess.check_call([sys.executable,'-m','pip','install',*needed])\nfor package in ['jax','jaxlib',*support]: print(package,importlib.metadata.version(package))"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Reproducibility and task representation\n\n`AAA + BBB = RRRR` uses a reversed four-digit answer so carries flow in generation order. The 15-token input predicts the shifted sequence, with loss only at target positions 11–14."
        ),
        nbf.v4.new_code_cell(
            "import matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(12, 2)); ax.axis('off')\nfor i, token in enumerate('123 + 456 = 9750'):\n    ax.add_patch(plt.Rectangle((i,0), .9,.8, fill=False)); ax.text(i+.45,.4,repr(token)[1:-1] or 'space',ha='center',va='center')\nax.set_xlim(0,16); ax.set_ylim(0,1); ax.set_title('Fixed-width token layout'); plt.show()"
        ),
        nbf.v4.new_code_cell(
            "import matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(8,2)); ax.axis('off'); ax.text(.1,.7,'normal answer: 0579',fontsize=16); ax.annotate('',(.75,.4),(.4,.4),arrowprops={'arrowstyle':'->'}); ax.text(.1,.15,'generation: 9750',fontsize=16); ax.set_title('Reverse the answer for least-significant-digit-first generation'); plt.show()"
        ),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(10,2)); ax.axis('off')\nlabels=['token + position','5 decoder blocks','final LayerNorm','tied token head']\nfor i,label in enumerate(labels):\n ax.add_patch(plt.Rectangle((i*2.4,.2),2,.6,fill=False)); ax.text(i*2.4+1,.5,label,ha='center',va='center')\n if i<len(labels)-1: ax.annotate('',(i*2.4+2.35,.5),(i*2.4+2,.5),arrowprops={'arrowstyle':'->'})\nax.set_xlim(0,9.5); ax.set_ylim(0,1); ax.set_title('Decoder-only transformer architecture'); plt.show()"
        ),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(11,2)); ax.axis('off')\nlabels=['x','LayerNorm','causal MHA','residual +','LayerNorm','GELU FFN','residual +']\nfor i,label in enumerate(labels): ax.text(i/7+.06,.5,label,ha='center',bbox={'fill':False,'boxstyle':'round'})\nfor i in range(6): ax.annotate('',((i+1)/7+.01,.5),(i/7+.11,.5),arrowprops={'arrowstyle':'->'})\nax.set_title('One Pre-LayerNorm block data flow'); plt.show()"
        ),
        nbf.v4.new_code_cell(
            "shape_rows=[['input','B × 15 × 320'],['Q, K, V','B × 15 × 5 × 64'],['scores','B × 5 × 15 × 15'],['attention output','B × 15 × 5 × 64'],['projection','B × 15 × 320']]\nfig,ax=plt.subplots(figsize=(7,2.5)); ax.axis('off'); ax.table(cellText=shape_rows,colLabels=['tensor','shape'],loc='center'); ax.set_title('MHA tensor-shape walkthrough'); plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 3. Canonical implementation (visible and written locally)"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nPath('src/jax_addition_transformer').mkdir(parents=True, exist_ok=True)"
        ),
        *[
            item
            for path in SOURCE_FILES
            for item in (
                nbf.v4.new_markdown_cell(f"### Canonical source: `{path.name}`"),
                source_cell(path),
            )
        ],
        nbf.v4.new_code_cell(
            "import sys\nsys.path.insert(0, 'src')\nfrom jax_addition_transformer.config import ExperimentConfig\nfrom jax_addition_transformer.model import AdditionTransformer, assert_parameter_count, parameter_table\nfrom flax import nnx\nconfig = ExperimentConfig.load('configs/exact_10m_t4.json') if __import__('pathlib').Path('configs/exact_10m_t4.json').exists() else ExperimentConfig()\nmodel = AdditionTransformer(config.model, rngs=nnx.Rngs(params=42))\nprint(parameter_table(config.model)); print('parameters:', assert_parameter_count(model, 10_000_000))"
        ),
        nbf.v4.new_markdown_cell("## 4. Architecture, tensor shapes, and causal mask"),
        nbf.v4.new_code_cell(
            "architecture=[['model','decoder-only causal'],['layers',config.model.n_layers],['width',config.model.d_model],['heads',config.model.n_heads],['head dimension',config.model.head_dim],['FFN',config.model.d_ff],['normalization',config.model.norm_type],['positions',config.model.position_type],['weight tying',config.model.tie_embeddings]]\nfig,ax=plt.subplots(figsize=(7,3)); ax.axis('off'); ax.table(cellText=architecture,colLabels=['decision','default'],loc='center'); ax.set_title('Frozen default architecture decisions'); plt.show()"
        ),
        nbf.v4.new_code_cell(
            "from jax_addition_transformer.attention import causal_mask\nmask=causal_mask(15); assert bool(mask[14,0]) and not bool(mask[0,14])\nfig, ax = plt.subplots(); ax.imshow(mask, cmap='Blues'); ax.set(title='Causal visibility mask', xlabel='key position', ylabel='query position'); plt.show()"
        ),
        nbf.v4.new_code_cell(
            "names, values = zip(*parameter_table(config.model)); fig, ax=plt.subplots(figsize=(8,3)); ax.barh(names,values); ax.set(title='Trainable parameter breakdown',xlabel='parameters',ylabel='component'); plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "A Pre-LayerNorm block follows `x → norm → causal attention → +x → norm → FFN → +`. Default Q/K/V are `[batch, 15, 5, 64]`; scores and probabilities are `[batch, 5, 15, 15]`."
        ),
        nbf.v4.new_markdown_cell("## 5. Dataset, split, and hybrid sampler"),
        nbf.v4.new_code_cell(
            "from jax_addition_transformer.data import create_split, pair_ids_to_operands, operand_lengths, carry_pattern\nfrom jax_addition_transformer.sampling import HybridSampler\nsplit=create_split(); a,b=pair_ids_to_operands(split.train); print(split.fingerprint, len(split.train),len(split.validation),len(split.test))\nfig,ax=plt.subplots(); ax.hist2d(operand_lengths(a),operand_lengths(b),bins=3); ax.set(title='Training pairs by operand length',xlabel='length(a)',ylabel='length(b)'); plt.show()"
        ),
        nbf.v4.new_code_cell(
            "sampler=HybridSampler.create(split.train,2048,42); natural=carry_pattern(a,b); ids=sampler.sample_ids(); sa,sb=pair_ids_to_operands(ids); sampled=carry_pattern(sa,sb)\nfig,ax=plt.subplots(); x=range(8); ax.bar([i-.2 for i in x],[(natural==i).mean() for i in x],.4,label='natural'); ax.bar([i+.2 for i in x],[(sampled==i).mean() for i in x],.4,label='hybrid'); ax.set(title='Natural versus hybrid carry distribution',xlabel='carry code (units bit is leftmost)',ylabel='fraction'); ax.legend(); plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 6. Optimizer and learning-rate schedule"),
        nbf.v4.new_code_cell(
            "from jax_addition_transformer.optimizers import learning_rate_schedule\nschedule=learning_rate_schedule(config.optimizer); steps=range(config.optimizer.total_steps); fig,ax=plt.subplots(); ax.plot(steps,[float(schedule(s)) for s in steps]); ax.set(title='Warmup-cosine learning-rate schedule',xlabel='step',ylabel='learning rate'); plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 7. Training, checkpointing, and measured curves"),
        nbf.v4.new_code_cell(
            "from jax_addition_transformer.training import train\nRUN_DIR='runs/colab-default'\nif FULL_RUN:\n    require_gpu_for_full_run(); resume_run=(Path(RUN_DIR)/'checkpoints'/'latest').exists(); summary=train(config,RUN_DIR,resume=resume_run)\nelse:\n    print('Training skipped: select a T4 GPU for the full run.')"
        ),
        nbf.v4.new_code_cell(
            "from jax_addition_transformer.checkpointing import restore_checkpoint\nfrom jax_addition_transformer.optimizers import make_optimizer\nif FULL_RUN:\n graphdef, empty_params = nnx.split(model, nnx.Param)\n optimizer, _ = make_optimizer(config.optimizer, empty_params); empty_optimizer_state = optimizer.init(empty_params)\n chosen = Path(RUN_DIR)/'checkpoints'/'best'\n if not chosen.exists(): chosen = Path(RUN_DIR)/'checkpoints'/'latest'\n restored_params, _, checkpoint_metadata = restore_checkpoint(chosen, empty_params, empty_optimizer_state, config.fingerprint)\n model = nnx.merge(graphdef, restored_params)\n print('Restored trained checkpoint at step', checkpoint_metadata['step'])"
        ),
        nbf.v4.new_code_cell(
            "import json, pathlib\np=pathlib.Path(RUN_DIR)/'history.jsonl'\nif p.exists():\n h=[json.loads(x) for x in p.read_text().splitlines()]; fields=[('loss','Training loss'),('answer_token_accuracy','Answer-token accuracy'),('gradient_global_norm','Gradient norm'),('examples_per_second','Examples per second')]\n fig,axs=plt.subplots(2,2,figsize=(10,7))\n for ax,(field,title) in zip(axs.flat,fields): ax.plot([r['step'] for r in h],[r[field] for r in h]); ax.set(title=title,xlabel='step',ylabel=field)\n plt.tight_layout(); plt.show()\nelse: print('No history exists; no curves are invented.')"
        ),
        nbf.v4.new_code_cell(
            "if p.exists():\n vh=[r for r in h if 'validation' in r]\n fig,axs=plt.subplots(1,2,figsize=(10,3))\n if vh:\n  axs[0].plot([r['step'] for r in vh],[r['validation']['teacher_forced_loss'] for r in vh],marker='o'); axs[1].plot([r['step'] for r in vh],[r['validation']['greedy_exact_match'] for r in vh],marker='o')\n for ax,title,ylabel in zip(axs,['Validation loss','Validation greedy exact match'],['loss','exact-match accuracy']): ax.set(title=title,xlabel='step',ylabel=ylabel)\n plt.tight_layout(); plt.show()\nelse: print('No measured validation history exists.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. Greedy validation, unseen test, exhaustive evaluation, and slices"
        ),
        nbf.v4.new_code_cell(
            "from jax_addition_transformer.evaluation import evaluate_ids\nif FULL_RUN:\n validation_metrics, _ = evaluate_ids(model,split.validation,config.training.evaluation_batch_size)\n test_metrics, _ = evaluate_ids(model,split.test,config.training.evaluation_batch_size)\n exhaustive_metrics, failures = evaluate_ids(model,__import__('numpy').arange(1_000_000),config.training.evaluation_batch_size)\n print('validation',validation_metrics); print('unseen test',test_metrics); print('exhaustive',exhaustive_metrics)\nelse: print('Evaluation follows a real trained checkpoint; skipped without a full run.')"
        ),
        nbf.v4.new_code_cell(
            "if FULL_RUN:\n fig,axs=plt.subplots(1,2,figsize=(12,4))\n carry=exhaustive_metrics['slices']['carry_pattern']; lengths=exhaustive_metrics['slices']['operand_length']\n axs[0].bar(carry,[carry[k]['accuracy'] for k in carry]); axs[0].set(title='Accuracy by carry pattern',xlabel='units–tens–hundreds carries',ylabel='exact-match accuracy')\n axs[1].bar(lengths,[lengths[k]['accuracy'] for k in lengths]); axs[1].set(title='Accuracy by operand-length pair',xlabel='length(a) × length(b)',ylabel='exact-match accuracy')\n plt.tight_layout(); plt.show()"
        ),
        nbf.v4.new_code_cell(
            "if FULL_RUN and failures:\n invalid=sum(not row['valid_digit_sequence'] for row in failures); incorrect=len(failures)-invalid\n fig,ax=plt.subplots(); ax.bar(['incorrect digits','invalid token'],[incorrect,invalid]); ax.set(title='Failure summary',xlabel='failure type',ylabel='count'); plt.show()\nelif FULL_RUN: print('No failures in this evaluated split.')"
        ),
        nbf.v4.new_markdown_cell(
            "The CLI evaluation writes every failure to CSV and separately supports the unseen test and exhaustive million-pair domain. Slice plots and failure summaries must be drawn from those generated artifacts; absent results are displayed as not evaluated."
        ),
        nbf.v4.new_code_cell(
            "if FULL_RUN:\n import csv, numpy as np\n fields=['pair_id','a','b','target','prediction','internal_generated_digits','valid_digit_sequence','operand_length_a','operand_length_b','carry_pattern','split']; split_labels=np.empty(1_000_000,dtype='<U10'); split_labels[split.train]='train'; split_labels[split.validation]='validation'; split_labels[split.test]='test'\n with open(pathlib.Path(RUN_DIR)/'failures.csv','w',newline='') as handle:\n  writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()\n  for row in failures: writer.writerow({**row,'split':split_labels[row['pair_id']]})\n summary_path=pathlib.Path(RUN_DIR)/'summary.json'; summary=json.loads(summary_path.read_text()); summary['evaluations']={'validation':validation_metrics,'test':test_metrics,'exhaustive':exhaustive_metrics}; summary_path.write_text(json.dumps(summary,indent=2)+'\\n')\n print('Wrote',len(failures),'failures without inventing results.')"
        ),
        nbf.v4.new_markdown_cell("## 9. Attention inspection, restore, and interactive inference"),
        nbf.v4.new_code_cell(
            "from jax_addition_transformer.tokenizer import encode,format_training_example\nimport jax.numpy as jnp\nif FULL_RUN:\n failed_ids={row['pair_id'] for row in failures}; solved_id=next((i for i in range(1_000_000) if i not in failed_ids),None)\n if solved_id is not None:\n  solved_a,solved_b=divmod(solved_id,1000); logits,maps=model(jnp.asarray([encode(format_training_example(solved_a,solved_b)[:-1])]),return_attention=True)\n  fig,axs=plt.subplots(1,config.model.n_heads,figsize=(15,3))\n  for i,ax in enumerate(axs): ax.imshow(maps[-1][0,i]); ax.set(title=f'head {i}',xlabel='key',ylabel='query')\n  fig.suptitle(f'Final-block attention for solved expression {solved_a:03d} + {solved_b:03d}'); plt.tight_layout(); plt.show()\n else: print('No solved expression exists, so no solved-example attention plot is claimed.')"
        ),
        nbf.v4.new_code_cell(
            "from jax_addition_transformer.generation import ask,predict_pair\nexamples=[(0,0),(1,9),(9,1),(9,991),(99,1),(199,801),(499,501),(500,500),(909,91),(999,1),(999,999)]\nif FULL_RUN:\n sample_text='\\n'.join(f'{a:03d} + {b:03d} = {predict_pair(model,a,b)}' for a,b in examples); print(sample_text); (pathlib.Path(RUN_DIR)/'sample_predictions.txt').write_text(sample_text+'\\n')\nelse: print('Mandatory model examples require a trained checkpoint.')"
        ),
        nbf.v4.new_code_cell(
            "if FULL_RUN:\n second_params,_,_=restore_checkpoint(chosen,empty_params,empty_optimizer_state,config.fingerprint); second_model=nnx.merge(graphdef,second_params)\n before=predict_pair(model,123,456); after=predict_pair(second_model,123,456); assert before==after; print('Checkpoint round trip preserved prediction:',after)"
        ),
        nbf.v4.new_code_cell(
            "from jax_addition_transformer.reporting import generate_report\nif FULL_RUN: print('Generated report:',generate_report(RUN_DIR))"
        ),
        nbf.v4.new_code_cell(
            "if FULL_RUN:\n print(ask(model,input('addition> '),debug=True))\nelse: print('Interactive inference requires a trained checkpoint.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 10. Configuration knobs and artifacts\n\n`ModelConfig` exposes normalization, positions, FFN, attention grouping, dimensions, bias, tying, precision, epsilon, RoPE base, and initialization scale. A completed run writes configuration, environment, split metadata, history, summary, report, failures, samples, and best/latest checkpoints."
        ),
    ]
    OUTPUT.parent.mkdir(exist_ok=True)
    nbf.write(notebook, OUTPUT)


if __name__ == "__main__":
    build()
