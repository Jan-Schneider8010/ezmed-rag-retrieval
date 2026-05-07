# prompts/

Prompt artefacts used by the pipeline. Plain text on purpose: each prompt is
its own file so it diffs cleanly in PRs, can be cited verbatim in the thesis,
and does not require touching Python to iterate.

## Layout

```
prompts/
├── <prompt_name>/
│   ├── system.md     # system message (one string)
│   └── user.md       # user-message template, .format()-able
└── qa_generation/
    └── strategies/
        └── <style>.md   # one strategy = one file
```

Loaded from Python via `ezmed.llm.prompts.load_prompt("<name>")` which returns
a `Prompt(system, user_template)` dataclass. `Prompt.render_user(**kwargs)`
applies `.format` to the user template.

## Adding a new prompt

1. Create `prompts/<name>/system.md` and `prompts/<name>/user.md`.
2. Use `{placeholder}` in `user.md` for runtime values (Python `.format` syntax).
3. Call `load_prompt("<name>")` from the module that needs it.

## Adding a new QA prompting strategy

1. Drop `prompts/qa_generation/strategies/<style>.md` with the instruction text.
2. Extend the `PromptingStrategy` Literal in `src/ezmed/schemas.py`.
3. The dataset builder picks it up automatically via `load_qa_strategy(name)`.
