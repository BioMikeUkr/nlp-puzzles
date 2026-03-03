# Module 18: Gradio Demos — Q&A

## Architecture & Design

**Q1: What is the difference between `gr.Interface` and `gr.Blocks`?**

`gr.Interface` is a high-level wrapper that automatically creates a UI from a single Python function — you specify inputs, outputs, and it generates the layout. `gr.Blocks` gives full control: you define components, layout (rows/columns/tabs), and wire events manually. Use `Interface` for simple one-function demos; use `Blocks` when you need multiple buttons, tabs, conditional logic, or shared state.

**Q2: Why should ML models be loaded outside Gradio callback functions?**

Gradio callbacks are called on every user request. Loading a model inside a callback (e.g., inside `def predict(text)`) would reload it from disk on every click — taking seconds each time and wasting memory. Loading once at module level (or with `@gr.on` startup) keeps the model in memory and makes callbacks instant. For heavy models like GLiNER/GLiClass this is critical.

**Q3: What format does `gr.HighlightedText` expect?**

Either a list of `(text_chunk, label_or_None)` tuples, or a list of dicts with `{"entity": label, "start": int, "end": int}`. The tuple format is easier to construct from NER output: split the original text at entity boundaries, yielding `(gap, None)` for non-entities and `(span, label)` for entities. The chunks must reconstruct the original text when concatenated.

**Q4: What format does `gr.Label` expect?**

A dict mapping label strings to float scores: `{"ransomware": 0.82, "phishing": 0.31, ...}`. Gradio renders it as a bar chart sorted by score. The `num_top_classes` parameter controls how many bars to show.

**Q5: When would you use `gr.State` vs a Python global variable?**

`gr.State` stores per-session state — each browser tab gets its own independent copy, making it safe for multi-user deployments. A Python global is shared across all sessions, causing race conditions and data leakage between users. Always use `gr.State` for anything that should be per-user (conversation history, selected preferences, intermediate results).

**Q6: How does `gr.Examples` work and why is it useful?**

`gr.Examples` renders a clickable table of example inputs below the demo. When a user clicks a row, it populates the input components. This lowers the barrier to entry — users don't have to think of their own inputs. Specify `inputs=[component]` to link examples to the right fields. Examples can also cache outputs with `cache_examples=True` for fast loading.

**Q7: How do you share a Gradio demo publicly?**

Call `demo.launch(share=True)`. Gradio creates a temporary public URL via its tunnel service (valid for 72 hours). For permanent hosting, deploy to Hugging Face Spaces (push a `app.py` + `requirements.txt`). For production, run `demo.launch(server_name="0.0.0.0", server_port=7860)` behind a reverse proxy.

**Q8: How do you handle slow model inference without blocking the UI?**

Use `gr.Button` with `variant="primary"` and Gradio's built-in async support. For streaming outputs, use Python generators with `yield` — Gradio will stream partial results to the UI. For very long tasks, consider running inference in a background thread and polling with `gr.Progress`.

**Q9: What is `show_legend=True` in `gr.HighlightedText` and when is it needed?**

It renders a legend below the highlighted text showing which color corresponds to which entity type. Essential when you have multiple entity types with different colors — without it users can't decode what the colors mean. Pair it with `color_map={"malware": "#ff6b6b", ...}` to assign specific colors per label.

**Q10: How do `gr.Row` and `gr.Column` control layout?**

`gr.Row` places children side by side horizontally; `gr.Column` stacks them vertically. Use `scale` parameter (e.g., `gr.Column(scale=2)`) to control relative widths — a column with `scale=2` takes twice the space of one with `scale=1`. Nest them freely: a row of two columns is a common two-panel layout for input + output.

---

## Implementation & Coding

**Q11: How do you build the `(chunk, label)` list from GLiNER output?**

```python
entities = sorted(model.predict_entities(text, types), key=lambda e: e["start"])
result, cursor = [], 0
for ent in entities:
    if ent["start"] > cursor:
        result.append((text[cursor:ent["start"]], None))  # gap
    result.append((text[ent["start"]:ent["end"]], ent["label"]))
    cursor = ent["end"]
if cursor < len(text):
    result.append((text[cursor:], None))  # trailing text
```
Sort by start first to handle overlapping/out-of-order entities.

**Q12: How do you parse a comma-separated label string safely?**

```python
labels = [l.strip() for l in labels_str.split(',') if l.strip()]
```
This handles extra spaces, trailing commas, and empty strings. Always validate the result before passing to the model — return `{}` or `[]` early if `labels` is empty.

**Q13: How do you wire a button to a function with multiple inputs and outputs?**

```python
btn.click(
    fn=my_function,
    inputs=[input1, input2, state],
    outputs=[output1, state]
)
```
`inputs` and `outputs` are lists of Gradio components. The function receives their current values and returns values to update them. The order must match.

**Q14: How do you set a default value for a `gr.Textbox`?**

Use the `value` parameter: `gr.Textbox(value="default text")`. For components inside `gr.Blocks`, the initial render will show this value. You can also set it dynamically by returning a new value from a callback.

**Q15: What does `threshold=0.0` in `cls_pipeline(text, labels, threshold=0.0)` do?**

It returns scores for all labels regardless of confidence, rather than filtering low-confidence labels. For UI display this is usually better — the user sees the full ranking. For production use a higher threshold (e.g., 0.3–0.5) to avoid showing irrelevant labels.

**Q16: How do you add a title and description to a Gradio app?**

In `gr.Interface`: use `title=` and `description=` parameters. In `gr.Blocks`: use `gr.Markdown("# Title\nDescription")` at the top — gives full Markdown control. The `title=` parameter in `gr.Blocks(title=...)` sets the browser tab title only.

**Q17: How do you run the same function on both Enter keypress and button click?**

Wire both events to the same function:
```python
text_input.submit(fn, inputs, outputs)  # fires on Enter
btn.click(fn, inputs, outputs)          # fires on button click
```

**Q18: How do you prevent a user from submitting while a request is in flight?**

Set `concurrency_limit` on the event: `btn.click(...).then(...)`. Or use `queue=True` in `demo.launch(queue=True)` — Gradio queues requests and shows a loading spinner automatically. The button is disabled while the function runs.

**Q19: How do you access uploaded file content in Gradio?**

Use `gr.File` or `gr.UploadButton`. The callback receives a file path string. Read it normally:
```python
def process(file_path):
    with open(file_path) as f:
        content = f.read()
```

**Q20: How do you update component properties (not just values) from a callback?**

Return a `gr.update(...)` object:
```python
def toggle_visibility(show):
    return gr.update(visible=show)

btn.click(toggle_visibility, inputs=checkbox, outputs=panel)
```

---

## Debugging & Troubleshooting

**Q21: The `gr.HighlightedText` shows nothing / all text is unlabeled — why?**

Most likely the function returned `None` or an empty list instead of `[(text, None)]`. Always return at least `[(text, None)]` as a fallback. Also check that entity labels match the keys in `color_map` exactly (case-sensitive).

**Q22: `gr.Label` shows nothing after classification — why?**

The function probably returned an empty dict `{}` or `None`. Check that `labels_str` is non-empty and that the pipeline output is correctly parsed into `{label: float}` format. Also verify `float(r['score'])` — some pipelines return numpy scalars that Gradio can't serialize.

**Q23: `gr.State` resets between calls — why?**

If `gr.State` is declared inside a callback function rather than at `gr.Blocks` level, it gets recreated on every call. Declare it at the top level of the `with gr.Blocks()` block.

**Q24: The demo port is already in use — how to fix?**

Pass a different port: `demo.launch(server_port=7861)`. Or close the previous demo: `demo.close()`. In Jupyter, `gr.close_all()` closes all running Gradio servers.

**Q25: GLiNER returns entities out of order — does that affect HighlightedText?**

Yes — if you don't sort entities by `start` before building chunks, overlapping or out-of-order entities cause the cursor to go backwards, producing garbled output. Always sort: `entities = sorted(entities, key=lambda e: e["start"])`.

---

## Trade-offs & Decisions

**Q26: When should you use `gr.Interface` over `gr.Blocks`?**

Use `Interface` for single-function demos with standard layouts — it's faster to write and automatically handles layout, submit button, and flagging. Use `Blocks` when you need: multiple buttons with different logic, tabs, conditional visibility, state, or custom layouts. Almost all production ML demos end up using `Blocks`.

**Q27: `gr.CheckboxGroup` vs `gr.Dropdown` for entity type selection — which is better?**

`CheckboxGroup` for multi-select with a small fixed set of options (≤10) — all options visible at once, easy to toggle. `Dropdown` for single-select or large option sets. For entity types (5 options, multi-select), `CheckboxGroup` is clearer.

**Q28: Should you cache examples with `cache_examples=True`?**

Cache when: inference is slow (>2s) and examples are static. Don't cache when: model output is non-deterministic, examples change frequently, or startup time matters (caching runs all examples at load). For NER/classification demos with edge models, caching is a good UX improvement.

**Q29: How do you balance UI responsiveness vs model latency?**

Edge models (gliner-bi-edge-v2.0, gliclass-edge-v3.0) are designed to be fast on CPU — typically <500ms per text. For heavier models: use `queue=True` + `concurrency_limit=1` to prevent overload; show a spinner; consider batching or async inference. Never block the event loop.

**Q30: Gradio vs Streamlit for ML demos — when to choose each?**

Gradio is purpose-built for ML: native support for model-specific components (HighlightedText, Label, Audio, Image), built-in HuggingFace Spaces integration, and simpler API for input→output demos. Streamlit is better for data dashboards with complex state, charts, and database connections. For showcasing NLP/CV models, Gradio is the standard choice in the ML community.
