# Cherry-pick provenance

**Repository**: `https://github.com/wxdongg99/crisp`  
**Branch**: `highres-exploration`  
**Source branch**: `origin/crisp-cvpr`  
**Recorded**: 2026-11-23

The high-resolution branch interleaves its own exploration commits with
fixes cherry-picked from the stable `crisp-cvpr` branch.

## Cherry-picked commits

| Stable commit | Highres commit | Subject |
|---|---|---|
| `dd933be` | `459b791` | Fix bg-token device init; add 40k Stage-A/Stage-B configs |
| `95d2a86` | `24eb6c3` | Add configurable inpainting CFG for Stage-A generation |
| `8ac3fdf` | `11c44c1` | Bypass embedding manager directly for unconditional CFG branch |

## Notes

- The content of each pair is the same; hashes differ because the parent
  history of `highres-exploration` is different.
- Future cherry-picks should use `git cherry-pick -x <stable-commit>` so the
  source hash is embedded in the commit message automatically.
- Local high-resolution worktree:
  `/media/cc/U2-Server/crisp-storage/new_crisp_highres`
- Local convenience path:
  `/home/cc/Documents/new_crisp_highres`

## Verification commands

```bash
cd /home/cc/Documents/new_crisp_highres

git show 459b791 --stat
git show 24eb6c3 --stat
git show 11c44c1 --stat

git diff origin/crisp-cvpr..origin/highres-exploration --stat
git log --oneline --graph --decorate --all | head -30
```
