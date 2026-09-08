# GNN Adopt/Hold Decision

## Decision: held back

Protocol: chronological. Branches: temporal-only and fused.

The requested dynamics ablation was not reported because ML2 does not contain
the actual 64-dimensional ML1 temporal latent `z(t)` for the L=30 UCS history.
The existing trainer and metrics code substitute an all-zero `z(t)`, which is
not a valid temporal-only baseline and cannot support an honest fused-versus-
temporal comparison. No adopt/hold decision based on fabricated metrics was
made.

The ML1 deviation export is now available and causally injected into graph slot
10 from window `t-1`. This resolves slot-10 coverage, but it does not replace
the missing temporal latent. The correct fallback for downstream fused-latent
consumers while the GNN is held back is:

```text
z'(t) = z(t)
```

The ablation can be resumed once a real L=30, 64-dimensional `z(t)` artifact or
the reproducible ML1 latent-extraction interface is supplied.