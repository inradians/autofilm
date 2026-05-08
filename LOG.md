# autofilm — experiment log

A running log of every experiment, in chronological order. The agent appends one
entry per run; the human reads top-to-bottom to see the iteration history.

Format per entry:

```
## exp_NNN — film_loss=X.XX (Δ from prev best)
- Changed: <one-line summary of what was edited in produce.py>
- Critic said: <2-3 bullet summary of metric.json["changes"]>
- Next: <what to try in exp_NNN+1>
```

---

<!-- experiments go here, newest at the bottom -->
