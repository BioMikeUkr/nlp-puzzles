# Module 18: Gradio Demos

Build interactive NLP demos with Gradio using real zero-shot models.

## What You'll Learn

- `gr.Interface` vs `gr.Blocks` — when to use each
- Core components: `Textbox`, `CheckboxGroup`, `Slider`, `HighlightedText`, `Label`
- Layout: `gr.Row`, `gr.Column`, `gr.Tabs`
- `gr.Examples` for one-click example loading
- `gr.State` for per-session persistent state
- Integrating GLiNER (NER) and GLiClass (classification) into live UIs

## Models Used

| Model | Task |
|-------|------|
| `knowledgator/gliner-bi-edge-v2.0` | Zero-shot NER |
| `knowledgator/gliclass-edge-v3.0` | Zero-shot text classification |

## Module Structure

```
18_gradio_demos/
├── learning/
│   ├── 01_gradio_basics.ipynb        # Interface, Blocks, components, State
│   └── 02_ner_and_cls_demo.ipynb     # Full demos with real models
├── tasks/
│   ├── task_01_ner_demo.ipynb        # Build NER demo with HighlightedText
│   └── task_02_combined_app.ipynb    # Multi-tab app + gr.State
└── solutions/
    ├── task_01_ner_demo_solution.ipynb
    └── task_02_combined_app_solution.ipynb
```

## Task Overview

### Task 01: NER Demo
1. `run_ner()` → `HighlightedText` format (chunk, label) tuples
2. `gr.Blocks` with `CheckboxGroup` + `Slider` + `HighlightedText`
3. `gr.Examples` with 3+ sample texts

### Task 02: Combined Multi-tab App
1. `run_cls()` → `{label: score}` dict for `gr.Label`
2. `gr.Blocks` + `gr.Tabs` combining both models
3. `gr.State` to persist custom labels between calls

## Key Gradio Patterns

### HighlightedText format
```python
# List of (text_chunk, label_or_None) tuples
[("The ", None), ("LockBit", "malware"), (" exploited ", None), ...]
```

### gr.Label output
```python
# Dict of label → score
{"ransomware": 0.82, "apt_attack": 0.61, "zero_day": 0.45}
```

### gr.State
```python
state = gr.State(value=default)
btn.click(fn, inputs=[..., state], outputs=[output, state])
```

### Load model once
```python
# Outside Gradio callbacks — loaded once at startup
model = GLiNER.from_pretrained("...")

def predict(text):
    return model.predict_entities(text, ...)  # reuses loaded model
```
